"""Static anchors at several scales propose motion; overlap layout verifies it."""
import cv2
import numpy as np
from .models import AlignmentResult,Rect


def layout_media_regions(image):
    if image.ndim!=3:return ()
    from .recognition import detect_bubbles
    width=image.shape[1]
    return tuple(Rect(x1,y1,x2-x1,y2-y1) for b in detect_bubbles(image,include_small_edge_media=True)
                 if b.kind in {'image','unknown'} for x1,y1,x2,y2 in [b.box]
                 if x1>=width*.065 and x2<=width*.94)


def without_media(valid,regions):
    result=valid.copy()
    for box in regions:result[max(0,box.y):box.bottom,max(0,box.x):box.right]=False
    return result


def _static_support(old,current,old_valid,new_valid,delta,aligner,ph,pw=16):
    pa,pb=max(0,delta),max(0,-delta);length=len(old)-abs(delta)
    points=[];tested=0
    for y in range(0,length-ph+1,ph):
        for x in range(0,old.shape[1]-pw+1,pw):
            if not old_valid[pa+y:pa+y+ph,x:x+pw].all() or not new_valid[pb+y:pb+y+ph,x:x+pw].all():continue
            a=old[pa+y:pa+y+ph,x:x+pw];b=current[pb+y:pb+y+ph,x:x+pw]
            if a.std()<aligner.config.min_texture_std:continue
            if (np.abs(cv2.Laplacian(a,cv2.CV_32F))>25).mean()<.05:continue
            tested+=1
            if np.abs(a.astype(np.int16)-b.astype(np.int16)).mean()<=2:
                if float(cv2.matchTemplate(a,b,cv2.TM_CCOEFF_NORMED)[0,0])>=.97:points.append((x,pa+y))
    return points,tested


def estimate_edge_layout(old,current,old_valid,new_valid,aligner,direction,motion_bounds=None):
    large=_estimate_scale(old,current,old_valid,new_valid,aligner,direction,motion_bounds,12)
    if large.accepted:return large
    accepted=[];failures=[large]
    # A clipped bubble may leave fewer than 12 rows. Small fragments can resolve
    # repeated avatar aliases, but must be corroborated by full-size patches.
    for height in (8,6):
        result=_estimate_scale(old,current,old_valid,new_valid,aligner,direction,motion_bounds,height)
        if result.accepted:accepted.append(result)
        else:failures.append(result)
    if accepted:
        if max(r.delta_px for r in accepted)-min(r.delta_px for r in accepted)>2:
            return AlignmentResult(False,'ambiguous_multiscale_layout')
        return accepted[0]  # Prefer the larger successful native patch.
    for reason in ('edge_layout_needs_motion_bounds','edge_layout_insufficient_large_support'):
        for result in failures:
            if result.reason==reason:return result
    return large


def _estimate_scale(old,current,old_valid,new_valid,aligner,direction,motion_bounds,patch_height):
    failed=AlignmentResult(False,'edge_layout_unconfirmed')
    h,w=old.shape;ph,pw=patch_height,16
    old_valid=old_valid.copy();new_valid=new_valid.copy()
    for valid in (old_valid,new_valid):
        valid[:4]=False;valid[-4:]=False;valid[:,:4]=False;valid[:,-4:]=False
    maximum=int(h*(1-aligner.config.min_overlap_ratio))
    low,high={'up':(-maximum,0),'down':(0,maximum),'both':(-maximum,maximum)}[direction]
    if motion_bounds is not None:
        low=max(low,int(motion_bounds['minimum']));high=min(high,int(motion_bounds['maximum']))
    seeds=[]
    # A repeated-avatar page can have no texture at either viewport edge.
    for start in range(0,h,112):
        band=range(start,min(start+112,h-ph+1),8)
        candidates=[]
        for y in band:
            for x in range(0,w-pw+1,8):
                if not new_valid[y:y+ph,x:x+pw].all():continue
                patch=current[y:y+ph,x:x+pw]
                texture=float(patch.std())
                if texture<aligner.config.min_texture_std or (np.abs(cv2.Laplacian(patch,cv2.CV_32F))>25).mean()<.05:continue
                candidates.append((texture,x,y))
        chosen=[]
        for item in sorted(candidates,reverse=True):
            _,x,y=item
            if any(abs(x-px)<pw and abs(y-py)<ph for _,px,py in chosen):continue
            chosen.append(item)
            if len(chosen)==12:break
        seeds.extend(chosen)
    votes={}
    for _,x,y in seeds:
        y0,y1=max(0,y+low),min(h-ph,y+high)
        x0,x1=max(0,x-1),min(w-pw,x+1)
        if y1<y0:continue
        search=old[y0:y1+ph,x0:x1+pw]
        scores=cv2.matchTemplate(search,current[y:y+ph,x:x+pw],cv2.TM_CCOEFF_NORMED)
        invalid=(~old_valid[y0:y1+ph,x0:x1+pw]).astype(np.uint8)
        integral=cv2.integral(invalid)
        blocked=integral[ph:,pw:]-integral[:-ph,pw:]-integral[ph:,:-pw]+integral[:-ph,:-pw]>0
        scores[blocked]=-1
        for _ in range(8):
            _,score,_,point=cv2.minMaxLoc(scores)
            if score<.97:break
            bx,by=point;delta=y0+by-y
            votes[delta]=votes.get(delta,0)+1
            scores[max(0,by-2):by+3,:]=-1
        if float(scores.max())>=.97:
            failed.reason='edge_layout_repetitive_anchors';return failed
    centers=[]
    for delta in sorted(votes,key=lambda d:votes[d],reverse=True):
        if not any(abs(delta-other)<=2 for other in centers):centers.append(delta)
    if not centers:
        failed.reason='edge_layout_no_anchors';return failed
    if len(centers)>24:
        failed.reason='edge_layout_too_many_candidates';return failed
    accepted=[];needs_bounds=False
    joint=old_valid&new_valid
    for delta in centers:
        error,coverage=aligner._validate(old,current,delta,joint)
        if error>aligner.config.max_validation_error or coverage<aligner.config.min_validation_coverage:
            failed.reason='edge_layout_pixels_changed';continue
        pa,pb=max(0,delta),max(0,-delta);length=h-abs(delta)
        points,tested=_static_support(old,current,old_valid,new_valid,delta,aligner,ph)
        if len(points)<6 or len(points)/max(1,tested)<.85:
            failed.reason='edge_layout_insufficient_support';continue
        spread=np.ptp(np.array(points),axis=0)
        if ph<12:
            robust,total=_static_support(old,current,old_valid,new_valid,delta,aligner,12)
            if len(robust)<6 or len(robust)/max(1,total)<.85 or max(np.ptp(np.array(robust),axis=0))<48:
                failed.reason='edge_layout_insufficient_large_support';continue
        rows=sorted({y for _,y in points})
        clusters=1+sum(b-a>24 for a,b in zip(rows,rows[1:]))
        broad=min(spread)>=48
        # Narrow layouts need measured scroll bounds and several independent
        # avatar rows. Multiple translations inside those bounds still fail.
        narrow=motion_bounds is not None and spread[0]>=16 and spread[1]>=48 and clusters>=3
        if motion_bounds is None and spread[0]>=16 and spread[1]>=48 and clusters>=3:
            needs_bounds=True
        if not broad and not narrow:
            failed.reason='edge_layout_narrow_support';continue
        reason='edge_layout_aligned' if ph==12 else f'edge_layout_aligned_{ph}px'
        accepted.append(AlignmentResult(True,reason,float(delta),length/h,len(points),candidates=len(centers),
            validation_error=error,validation_coverage=coverage))
    if not accepted:
        if needs_bounds:failed.reason='edge_layout_needs_motion_bounds'
        return failed
    if max(r.delta_px for r in accepted)-min(r.delta_px for r in accepted)>2:
        return AlignmentResult(False,'ambiguous_edge_layout')
    return max(accepted,key=lambda r:r.inliers)
