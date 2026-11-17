from __future__ import annotations

import logging
import time
from dataclasses import asdict
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any,Callable,Protocol

import numpy as np

from .frame_pool import ImageBudget
from .models import AlignmentConfig,AlignmentResult,CaptureConfig,Rect
from .pipeline import OcrWorker
from .recognition import RapidOcrAdapter
from .scroll import ScrollPlanner
from .session import MemorySession
from .stability import LayoutStabilityDetector

logger=logging.getLogger('welog.capture')


class Source(Protocol):
    def read(self)->np.ndarray|None: ...
    def scroll(self,units:int)->None: ...
    def close(self)->None: ...


class StableTimeout(RuntimeError):
    def __init__(self,latest,diagnostics=None,temporal_frames=()):
        super().__init__('Frame did not stabilize before timeout')
        self.latest,self.diagnostics=latest,diagnostics or {}
        self.temporal_frames=list(temporal_frames)


def wait_for_stable(source,config,alignment,stop_requested=None,diagnostics=None,budget=None):
    deadline=time.monotonic()+config.stable_timeout_s
    latest=None;workspace=None;metrics={}
    detector=LayoutStabilityDetector(alignment,config.stable_samples,config.stability_max_dynamic_ratio,config.stability_media_max_dynamic_ratio)
    try:
        while time.monotonic()<deadline:
            if stop_requested and stop_requested():raise KeyboardInterrupt
            sample=source.read()
            if sample is not None:
                latest=sample
                if budget is not None and workspace is None:
                    workspace=budget.reserve(sample.shape[0]*sample.shape[1]*20)
                result=detector.observe(sample);metrics=result.to_dict()
                if diagnostics is not None:diagnostics.clear();diagnostics.update(metrics)
                if result.stable:return sample
            time.sleep(config.sample_interval_s)
        logger.warning('stability_timeout reason=%s dynamic_ratio=%.4f media_support=%.4f outside_pixels=%s static_matches=%s static_method=%s',
                       metrics.get('reason'),metrics.get('dynamic_region_ratio',0),metrics.get('media_supported_ratio',0),metrics.get('outside_media_pixels',0),metrics.get('static_matches',0),metrics.get('static_method'))
        temporal=[]
        if config.debug_save_images and config.debug_image_policy=='events' and budget is not None:
            metrics['temporal_media_regions']=[list(boxes) for boxes in list(detector.media_history)[-3:]]
            try:
                for item in list(detector.history)[-3:]:temporal.append(budget.copy(item))
            except Exception:
                for lease in temporal:lease.release()
                temporal=[];metrics['temporal_retention_error']='image_budget'
        raise StableTimeout(latest,metrics,temporal)
    finally:
        detector.history.clear()
        detector.media_history.clear()
        if workspace:workspace.release()


def run_capture(source,output,*,config=None,alignment=None,metadata=None,progress=None,stop_requested=None,ocr_backend=None):
    """Memory-only by default; explicit debug mode retains ROI frames for offline analysis."""
    config,alignment=config or CaptureConfig(),alignment or AlignmentConfig()
    budget=ImageBudget(config.image_memory_limit_bytes)
    worker=None;session=None;accepted=rejected=0;offset=0.;reference=None
    status,reason='needs_attention','not_started'
    planner=ScrollPlanner(min(config.target_step_ratio,(1-alignment.min_overlap_ratio)*.9),config.max_scroll_events)
    sign=1 if config.direction=='down' else -1
    stationary=0;motion_observed=False;last_action={'kind':'initial'};diagnostic_save_error=None
    events_enabled=config.debug_save_images and config.debug_image_policy=='events'
    debug_reference=None;debug_reference_frame=None;diagnostic_leases=[]
    last_attempt_image=None;last_attempt_frame=None
    last_attempt_accepted=False

    def stopped():
        return bool((stop_requested and stop_requested()) or (getattr(source,'interrupted',lambda:False)()))

    def wait_at_boundary():
        deadline=time.monotonic()+config.boundary_wait_s
        while time.monotonic()<deadline:
            if stopped():raise KeyboardInterrupt
            time.sleep(min(.05,max(0.,deadline-time.monotonic())))

    def has_capacity():
        return config.max_frames==0 or session.counter<config.max_frames

    def capture_frame(image,action):
        if config.debug_save_images and not events_enabled:
            with budget.reserve(image.nbytes*2):return session.capture(image,action)
        return session.capture(image,action)

    def attempt(action):
        nonlocal accepted,rejected,offset,reference,last_action
        nonlocal debug_reference,debug_reference_frame,last_attempt_image,last_attempt_frame,last_attempt_accepted
        last_action=action;stability={}
        last_attempt_image=None;last_attempt_frame=None;last_attempt_accepted=False
        image=wait_for_stable(source,config,alignment,stopped,stability,budget)
        frame=capture_frame(image,{**action,'stability':stability})
        if events_enabled:last_attempt_image=image;last_attempt_frame=frame
        lease=budget.copy(image)
        future=worker.submit(lease,frame['frame_id'],tuple(Rect(**b) for b in stability.get('dynamic_masks',[])),config.direction,frame['captured_at'])
        # Only one unverified transition is allowed: this wait supplies backpressure.
        # The model owns the lease until it finishes, even after a stop request.
        stop_latched=False
        while True:
            try:
                outcome=future.result(timeout=.05);break
            except FutureTimeout:
                if future.done():raise
                stop_latched=stop_latched or stopped()
        session.evaluate(frame,outcome)
        result=outcome['alignment']
        last_attempt_accepted=bool(result['accepted'])
        if result['accepted']:
            accepted+=1;offset=outcome['scroll_offset_px'];reference=frame['frame_id']
        else:rejected+=1
        if events_enabled:
            previous=(debug_reference_frame,debug_reference.image) if debug_reference else None
            session.process_event(frame,image,outcome,previous,budget)
            if result['accepted']:
                # Keep one bounded, charged raw reference for the next failure context.
                replacement=budget.copy(image)
                if debug_reference:debug_reference.release()
                debug_reference=replacement;debug_reference_frame={**frame,'scroll_offset_px':offset,'coordinate_status':'confirmed'}
        logger.info('frame=%s action=%s accepted=%s delta=%s offset=%s method=%s stability=%s static_method=%s static_matches=%s',frame['frame_id'],action['kind'],
                    result['accepted'],result['delta_px'],outcome['scroll_offset_px'],outcome['motion_method'],stability['reason'],stability.get('static_method'),stability.get('static_matches'))
        if result.get('feature_rejection_reason'):
            logger.info('feature_rejected frame=%s reason=%s total=%s overlap=%s ratio=%s',frame['frame_id'],
                result['feature_rejection_reason'],result.get('feature_matches_total'),result.get('feature_matches_overlap'),result.get('feature_inlier_ratio'))
        if result.get('edge_rejection_reason'):logger.info('edge_rejected frame=%s reason=%s',frame['frame_id'],result['edge_rejection_reason'])
        if progress:progress({'frame_id':frame['frame_id'],'accepted':result['accepted'],'reason':result['reason'],
                              'delta_px':result['delta_px'],'scroll_offset_px':outcome['scroll_offset_px']})
        if outcome['ocr_error']:raise RuntimeError('ocr_backend_failed:'+outcome['ocr_error'])
        if stop_latched:raise KeyboardInterrupt
        return result

    try:
        session_metadata={'origin_y_px':config.origin_y_px,'capture_config':asdict(config),
                          'alignment_config':alignment.to_dict(),**(metadata or {})}
        if config.debug_save_images:
            from .debug_capture import DebugSession,EventDebugSession
            session=(EventDebugSession if events_enabled else DebugSession)(output,session_metadata,config.debug_max_bytes)
        else:session=MemorySession(output,session_metadata)
        worker=OcrWorker(session.path,budget,ocr_backend or RapidOcrAdapter(),alignment,config)
        logger.info('memory_capture_start direction=%s backend=%s debug_save_images=%s debug_policy=%s output=%s',config.direction,config.capture_backend,config.debug_save_images,config.debug_image_policy,session.path)
        attempt({'kind':'initial'})
        while has_capacity():
            if config.max_frames==0 and stationary:wait_at_boundary()
            events=planner.events
            last_action={'kind':'scroll','events':events,'event_units':sign*config.initial_wheel_units,
                         'wheel_units':sign*config.initial_wheel_units*events}
            for index in range(events):
                if stopped():raise KeyboardInterrupt
                source.scroll(sign*config.initial_wheel_units)
                if index+1<events:time.sleep(config.scroll_event_interval_s)
            time.sleep(config.settle_delay_s)
            result=attempt(last_action)
            recovered=False
            if not result['accepted']:
                for _ in range(config.recovery_attempts):
                    if stopped():raise KeyboardInterrupt
                    if not has_capacity():break
                    time.sleep(config.settle_delay_s)
                    result=attempt({'kind':'retry','wheel_units':0,'events':0})
                    if result['accepted']:break
                planner.recover()
                if not result['accepted']:
                    status,reason='needs_attention','continuity_unconfirmed';break
                recovered=True
            delta=result['delta_px'] or 0.
            stationary=stationary+1 if delta==0 else 0
            motion_observed=motion_observed or delta!=0
            if stationary>=config.bottom_attempts:
                status,reason=('visible_end_candidate','repeated_verified_zero_motion') if motion_observed else ('needs_attention','no_scroll_motion')
                if motion_observed and config.max_frames==0:status='boundary_reached'
                break
            if delta and not recovered:planner.observe(delta,events,session.last_height)
        else:status,reason='limit_reached','max_frames'
    except KeyboardInterrupt:status,reason='stopped','user_interrupt'
    except Exception as error:
        # Do not log vendor exception text that may include recognized chat content.
        status,reason='needs_attention',('ocr_backend_failed' if str(error).startswith('ocr_backend_failed:') else type(error).__name__)
        logger.error('memory_capture_error type=%s',type(error).__name__)
        if session is None:raise
        if isinstance(error,StableTimeout) and error.latest is not None:
            diagnostic_leases=error.temporal_frames
            try:frame=capture_frame(error.latest,{'kind':'unstable_diagnostic','attempted_action':last_action,'stability':error.diagnostics})
            except Exception as save_error:
                logger.error('diagnostic_retention_error type=%s',type(save_error).__name__)
                diagnostic_save_error=type(save_error).__name__
                reason=diagnostic_save_error
                session.append('diagnostics.jsonl',{'kind':'unstable_diagnostic','image_save_error':diagnostic_save_error,
                    'attempted_action':last_action,'stability':error.diagnostics})
                frame=None
            if frame is not None:
                session.evaluate(frame,{'alignment':AlignmentResult(False,'stability_timeout').to_dict(),
                                        'reference_id':reference,'scroll_offset_px':None})
                if events_enabled:
                    try:
                        with budget.reserve(error.latest.nbytes*2):
                            session.retain_event(frame,error.latest,['stability_timeout'],reference=(debug_reference_frame,debug_reference.image) if debug_reference else None,
                                temporal=diagnostic_leases)
                    except Exception as save_error:
                        diagnostic_save_error=type(save_error).__name__
                        session.append('diagnostics.jsonl',{'frame_id':frame['frame_id'],'reasons':['diagnostic_write_failed'],
                            'image_save_error':diagnostic_save_error})
            rejected+=1
        elif events_enabled:
            try:
                image=last_attempt_image
                with budget.reserve(image.nbytes*2 if image is not None else 0):
                    session.retain_event(last_attempt_frame,image,['capture_error',reason],
                        reference=(debug_reference_frame,debug_reference.image) if debug_reference else None,coordinate_eligible=last_attempt_accepted,
                        scroll_offset_px=offset if last_attempt_accepted else None)
            except Exception as save_error:
                diagnostic_save_error=type(save_error).__name__
                session.append('diagnostics.jsonl',{'frame_id':last_attempt_frame['frame_id'] if last_attempt_frame else None,
                    'reasons':['diagnostic_write_failed',reason],'image_save_error':diagnostic_save_error})
    finally:
        try:source.close()
        finally:
            if session:
                if debug_reference:debug_reference.release();debug_reference=None
                for lease in diagnostic_leases:lease.release()
                last_attempt_image=None
                summary={'status':status,'reason':reason,'accepted':accepted,'rejected':rejected,
                         'final_scroll_offset_px':offset,'image_storage':('event_png' if events_enabled else 'debug_png') if config.debug_save_images else 'none',
                         'debug_image_policy':config.debug_image_policy if config.debug_save_images else 'none',
                         'debug_save_images':config.debug_save_images,'debug_image_count':getattr(session,'image_count',0),
                         'debug_image_bytes':getattr(session,'image_bytes',0),
                         'diagnostic_image_save_error':diagnostic_save_error,
                         'capture_mode':'until_boundary' if config.max_frames==0 else 'limited',
                         'capture_direction':config.direction,'timeline_order':'oldest_first',
                         'capture_backend':config.capture_backend,
                         'boundary_reached':status in {'boundary_reached','visible_end_candidate'},
                         'boundary':('top' if config.direction=='up' else 'bottom') if status in {'boundary_reached','visible_end_candidate'} else None}
                if worker:
                    counts=worker.finish(summary);summary.update(counts)
                session.finish(status,**{k:v for k,v in summary.items() if k!='status'})
                logger.info('memory_capture_end status=%s reason=%s accepted=%s rejected=%s offset=%s',status,reason,accepted,rejected,offset)
    return {**summary,'output':str(session.path),'report':str(session.path/'report.html')}
