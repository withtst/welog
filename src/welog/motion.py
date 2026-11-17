"""Text anchors first, followed by validated static visual evidence."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import cv2
import numpy as np

from .alignment import OverlapAligner
from .frame_pool import ImageBudget
from .imageio import gray
from .models import AlignmentConfig, AlignmentResult


class MotionTracker:
    def __init__(self,budget:ImageBudget,config:AlignmentConfig):
        self.budget,self.config=budget,config
        self.previous=None;self.lines=[];self.masks=()
        self.offset=0.;self.reference_id=None

    def accept(self,image,frame_id,lines,direction,masks=()):
        previous_id=self.reference_id
        if self.previous is None:
            result=AlignmentResult(True,'origin',delta_px=0.,overlap_ratio=1.)
            method='origin'
        else:
            if self.previous.image.shape!=image.shape[:2]:
                return AlignmentResult(False,'layout_size_changed'),previous_id,'layout'
            cfg=replace(self.config,masks=self.config.masks+self.masks+tuple(masks))
            aligner=OverlapAligner(cfg)
            current=gray(image);old=self.previous.image
            valid=aligner._valid_mask(current.shape)
            result=None;method='text'
            # Repeated short replies cannot serve as unique anchors.
            counts_a=Counter(line.text for line in self.lines if line.score>=.7)
            counts_b=Counter(line.text for line in lines if line.score>=.7)
            shifts=[]
            for line in lines:
                if line.score<.7 or len(line.text.strip())<2 or counts_a[line.text]!=1 or counts_b[line.text]!=1:
                    continue
                oldline=next((item for item in self.lines if item.text==line.text and abs(item.box[0]-line.box[0])<=8),None)
                if oldline:
                    shifts.append((oldline.box[1]+oldline.box[3]-line.box[1]-line.box[3])/2)
            if len(shifts)>=2 and max(shifts)-min(shifts)<=3:
                delta=int(round(float(np.median(shifts))))
                if self.allowed(delta,current.shape[0],direction):
                    # OCR line boxes jitter. Refine against native pixels rather than accumulating that jitter.
                    candidates=[d for d in range(delta-3,delta+4) if self.allowed(d,current.shape[0],direction)]
                    def cost(d):
                        pa,pb=max(0,d),max(0,-d);length=current.shape[0]-abs(d)
                        allowed=valid[pa:pa+length]&valid[pb:pb+length]
                        difference=np.abs(old[pa:pa+length].astype(np.int16)-current[pb:pb+length].astype(np.int16))
                        return float(difference[allowed].mean()) if allowed.any() else float('inf')
                    delta=min(candidates,key=cost)
                    error,coverage=aligner._validate(old,current,delta,valid)
                    if error<=cfg.max_validation_error and coverage>=cfg.min_validation_coverage:
                        result=AlignmentResult(True,'aligned',float(delta),(current.shape[0]-abs(delta))/current.shape[0],len(shifts),
                                               residual_px=float(max(abs(s-delta) for s in shifts)),validation_error=error,validation_coverage=coverage)
            if result is None:
                method='template';result=aligner.estimate(old,current,direction=direction)
            if not result.accepted:
                old_valid=OverlapAligner(replace(self.config,masks=self.config.masks+self.masks))._valid_mask(current.shape)
                new_valid=OverlapAligner(replace(self.config,masks=self.config.masks+tuple(masks)))._valid_mask(current.shape)
                feature=self.features(old,current,valid,aligner,direction,old_valid,new_valid)
                if feature.accepted:
                    result,method=feature,'static_features'
                else:
                    for key in ['feature_matches_total','feature_matches_overlap','feature_inlier_ratio','feature_rejection_reason']:
                        setattr(result,key,getattr(feature,key))
        if result.accepted:
            replacement=self.budget.copy(gray(image))
            if self.previous:self.previous.release()
            self.previous=replacement;self.lines=list(lines);self.masks=tuple(masks)
            self.offset+=result.delta_px or 0.;self.reference_id=frame_id
        return result,previous_id,method

    def allowed(self,delta,height,direction):
        return abs(delta)<=height*(1-self.config.min_overlap_ratio) and (direction=='both' or (delta<=0 if direction=='up' else delta>=0))

    def features(self,old,current,valid,aligner,direction,previous_valid=None,current_valid=None):
        failed=AlignmentResult(False,'static_features_unconfirmed')
        def reject(reason):failed.feature_rejection_reason=reason;return failed
        detector=cv2.SIFT_create(nfeatures=2500)
        a,da=detector.detectAndCompute(old,(valid if previous_valid is None else previous_valid).astype(np.uint8)*255)
        b,db=detector.detectAndCompute(current,(valid if current_valid is None else current_valid).astype(np.uint8)*255)
        if da is None or db is None or len(a)<12 or len(b)<12:return reject('insufficient_features')
        matches=[pair[0] for pair in cv2.BFMatcher().knnMatch(da,db,k=2) if len(pair)==2 and pair[0].distance<pair[1].distance*.7]
        failed.feature_matches_total=len(matches)
        if len(matches)<12:return reject('insufficient_unique_features')
        src=np.float32([a[m.queryIdx].pt for m in matches]);dst=np.float32([b[m.trainIdx].pt for m in matches])
        remaining=np.ones(len(matches),dtype=bool);accepted=[];height=old.shape[0]
        for _ in range(3):
            indices=np.flatnonzero(remaining)
            if len(indices)<12:break
            matrix,inliers=cv2.estimateAffinePartial2D(src[indices],dst[indices],method=cv2.RANSAC,ransacReprojThreshold=1.5,maxIters=5000)
            if matrix is None:break
            selected=np.zeros(len(matches),dtype=bool);selected[indices[inliers.ravel().astype(bool)]]=True
            remaining[selected]=False
            if selected.sum()<12:reject('insufficient_consensus');continue
            if abs(matrix[0,0]-1)>.003 or abs(matrix[1,1]-1)>.003 or abs(matrix[0,1])>.003:
                reject('non_translation_geometry');continue
            shifts=dst[selected]-src[selected];delta=int(round(-float(np.median(shifts[:,1]))))
            if abs(float(np.median(shifts[:,0])))>1 or not self.allowed(delta,height,direction):
                reject('direction_or_geometry');continue
            # Novel/disappeared content and fixed viewport overlays outside the actual
            # overlap must not dilute the consensus for a valid page translation.
            overlap=(src[:,1]>=max(0,delta))&(src[:,1]<height+min(0,delta))&(dst[:,1]>=max(0,-delta))&(dst[:,1]<height+min(0,-delta))
            selected &= overlap
            count=int(overlap.sum());ratio=float(selected.sum()/max(1,count))
            failed.feature_matches_overlap=count;failed.feature_inlier_ratio=ratio
            if selected.sum()<12 or ratio<.5:reject('insufficient_overlap_consensus');continue
            shifts=dst[selected]-src[selected]
            if max(np.ptp(src[selected,0]),np.ptp(src[selected,1]))<48:reject('insufficient_spatial_support');continue
            error,coverage=aligner._validate(old,current,delta,valid)
            if error>self.config.max_validation_error or coverage<self.config.min_validation_coverage:
                reject('overlap_pixels_changed');continue
            accepted.append(AlignmentResult(True,'aligned',float(delta),(height-abs(delta))/height,int(selected.sum()),candidates=len(matches),
                residual_px=float(np.max(np.abs(shifts[:,1]+delta))),validation_error=error,validation_coverage=coverage,
                feature_matches_total=len(matches),feature_matches_overlap=count,feature_inlier_ratio=ratio))
        if not accepted:return failed
        deltas={item.delta_px for item in accepted}
        if len(deltas)>1 and max(deltas)-min(deltas)>self.config.consensus_tolerance_px:return reject('ambiguous_translations')
        return max(accepted,key=lambda item:(item.inliers,item.feature_inlier_ratio))

    def close(self):
        if self.previous:self.previous.release();self.previous=None
        self.lines=[];self.masks=()
