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
            keep=old_valid[pa+y:pa+y+ph,x:x+pw]&new_valid[pb+y:pb+y+ph,x:x+pw]
            if ph>=12 and not keep.all():continue
            if not keep.all() and (keep.mean()<.75 or keep.sum()<64):continue
            a=old[pa+y:pa+y+ph,x:x+pw];b=current[pb+y:pb+y+ph,x:x+pw]
            if a[keep].std()<aligner.config.min_texture_std:continue
            if (np.abs(cv2.Laplacian(a,cv2.CV_32F))>25).mean()<.05:continue
            tested+=1
            aa=a[keep].astype(np.float32);bb=b[keep].astype(np.float32)
            if np.abs(aa-bb).mean()<=2:
                aa-=aa.mean();bb-=bb.mean();norm=float(np.linalg.norm(aa)*np.linalg.norm(bb))
                if norm>0 and float(np.dot(aa,bb)/norm)>=.97:points.append((x,pa+y))
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


def propose_bounded_static_layout(old,current,old_valid,new_valid,aligner,direction,motion_bounds):
    """A unique narrow static translation is a proposal, never an acceptance.

    Keep the full-size patch, ratio, pixel and ambiguity checks. The caller
    must independently corroborate the same delta with broad spatial support.
    """
    if motion_bounds is None:return AlignmentResult(False,'proposal_needs_motion_bounds')
    return _estimate_scale(old,current,old_valid,new_valid,aligner,direction,motion_bounds,12,proposal_only=True)


def corroborate_bounded_surface(old,current,old_valid,new_valid,aligner,direction,motion_bounds):
    """Full-size pixel agreement with broad support on both axes, uniquely."""
    if motion_bounds is None:return AlignmentResult(False,'surface_needs_motion_bounds')
    return _estimate_scale(old,current,old_valid,new_valid,aligner,direction,motion_bounds,12,require_broad=True)


def _estimate_scale(old,current,old_valid,new_valid,aligner,direction,motion_bounds,patch_height,*,proposal_only=False,require_broad=False):
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
            if len(chosen)==24:break
        seeds.extend(chosen)
    votes={};repetitive=0
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
        local=[]
        for _ in range(8):
            _,score,_,point=cv2.minMaxLoc(scores)
            if score<.97:break
            bx,by=point;delta=y0+by-y
            local.append(delta)
            scores[max(0,by-2):by+3,:]=-1
        if float(scores.max())>=.97:
            # One textureless border/repeated word must not veto independent
            # anchors. Do not keep the truncated votes from this weak seed.
            repetitive+=1;continue
        for delta in local:votes[delta]=votes.get(delta,0)+1
    centers=[]
    for delta in sorted(votes,key=lambda d:votes[d],reverse=True):
        if not any(abs(delta-other)<=2 for other in centers):centers.append(delta)
    if not centers:
        failed.reason='edge_layout_repetitive_anchors' if repetitive else 'edge_layout_no_anchors';return failed
    # Candidate count is bounded by native viewport height, not evidence
    # quality. Common glyphs can propose many offsets; validate every one so
    # neither a low-ranked real match nor a competing translation is hidden.
    failed.candidates=len(centers)
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
        starts=[rows[0]]+[b for a,b in zip(rows,rows[1:]) if b-a>24]
        clusters=len(starts)
        broad=min(spread)>=48
        if require_broad and not broad:
            failed.reason='edge_layout_broad_support_required';continue
        # Narrow layouts need measured scroll bounds and several independent
        # avatar rows. Multiple translations inside those bounds still fail.
        tight_pair=motion_bounds is not None and clusters==2 and motion_bounds['maximum']-motion_bounds['minimum']<min(np.diff(starts))
        narrow=motion_bounds is not None and spread[0]>=16 and spread[1]>=48 and (clusters>=3 or tight_pair)
        if motion_bounds is None and spread[0]>=16 and spread[1]>=48 and clusters>=2:
            needs_bounds=True
        if proposal_only and min(spread)<16:
            failed.reason='edge_layout_insufficient_proposal_span';continue
        if not proposal_only and not broad and not narrow:
            failed.reason='edge_layout_narrow_support';continue
        reason='edge_layout_aligned' if ph==12 else f'edge_layout_aligned_{ph}px'
        accepted.append(AlignmentResult(True,reason,float(delta),length/h,len(points),candidates=len(centers),
            validation_error=error,validation_coverage=coverage))
    if not accepted:
        if needs_bounds:failed.reason='edge_layout_needs_motion_bounds'
        return failed
    if max(r.delta_px for r in accepted)-min(r.delta_px for r in accepted)>2:
        # A repeated timestamp can align old texture against a mostly empty
        # overlap. Test texture in the current frame as well before calling
        # two candidate offsets equivalent.
        supported = []
        for result in accepted:
            reverse_delta = -int(result.delta_px)
            points, tested = _static_support(current, old, new_valid, old_valid, reverse_delta, aligner, 12)
            if len(points) >= 6 and len(points) / max(1, tested) >= .85:
                supported.append(result)
        if supported and max(r.delta_px for r in supported)-min(r.delta_px for r in supported)<=2:
            accepted = supported
        else:
            # Repeated bubbles can pass the patch ratio in both directions
            # while a single distinct bubble disproves the shifted copies.
            # Allow faint noise on at most 0.1% of the static overlap.
            # Every rival offset must have substantial contradictory pixels.
            consistent = []
            contradicted = 0
            for candidate in accepted:
                delta = int(candidate.delta_px)
                pa, pb = max(0, delta), max(0, -delta)
                length = h - abs(delta)
                keep = old_valid[pa:pa+length] & new_valid[pb:pb+length]
                difference = cv2.absdiff(old[pa:pa+length], current[pb:pb+length])[keep]
                if difference.max() <= 8 and np.count_nonzero(difference) * 1000 <= difference.size:
                    consistent.append(candidate)
                elif np.count_nonzero(difference > 20) >= 12:
                    contradicted += 1
            if len(consistent) != 1 or contradicted != len(accepted) - 1:
                return AlignmentResult(False,'ambiguous_edge_layout',candidates=len(centers))
            consistent[0].reason = 'edge_layout_aligned_static_overlap'
            accepted = consistent
    result=max(accepted,key=lambda r:r.inliers)
    if proposal_only:
        result.accepted=False;result.reason='bounded_static_proposal'
    return result
