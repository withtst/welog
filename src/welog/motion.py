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
                feature=self.features(old,current,valid,aligner,direction)
                if feature.accepted:
                    result,method=feature,'static_features'
        if result.accepted:
            replacement=self.budget.copy(gray(image))
            if self.previous:self.previous.release()
            self.previous=replacement;self.lines=list(lines);self.masks=tuple(masks)
            self.offset+=result.delta_px or 0.;self.reference_id=frame_id
        return result,previous_id,method

    def allowed(self,delta,height,direction):
        return abs(delta)<=height*(1-self.config.min_overlap_ratio) and (direction=='both' or (delta<=0 if direction=='up' else delta>=0))

    def features(self,old,current,valid,aligner,direction):
        failed=AlignmentResult(False,'static_features_unconfirmed')
        detector=cv2.SIFT_create(nfeatures=2500)
        a,da=detector.detectAndCompute(old,valid.astype(np.uint8)*255)
        b,db=detector.detectAndCompute(current,valid.astype(np.uint8)*255)
        if da is None or db is None or len(a)<12 or len(b)<12:return failed
        matches=[pair[0] for pair in cv2.BFMatcher().knnMatch(da,db,k=2) if len(pair)==2 and pair[0].distance<pair[1].distance*.7]
        if len(matches)<12:return failed
        src=np.float32([a[m.queryIdx].pt for m in matches]);dst=np.float32([b[m.trainIdx].pt for m in matches])
        matrix,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=1.5,maxIters=5000)
        if matrix is None:return failed
        selected=inliers.ravel().astype(bool)
        if selected.sum()<12 or selected.mean()<.5:return failed
        if abs(matrix[0,0]-1)>.003 or abs(matrix[1,1]-1)>.003 or abs(matrix[0,1])>.003:return failed
        shifts=dst[selected]-src[selected];delta=int(round(-float(np.median(shifts[:,1]))))
        if abs(float(np.median(shifts[:,0])))>1 or not self.allowed(delta,current.shape[0],direction):return failed
        if max(np.ptp(src[selected,0]),np.ptp(src[selected,1]))<48:return failed
        error,coverage=aligner._validate(old,current,delta,valid)
        if error>self.config.max_validation_error or coverage<self.config.min_validation_coverage:return failed
        return AlignmentResult(True,'aligned',float(delta),(old.shape[0]-abs(delta))/old.shape[0],int(selected.sum()),
                               residual_px=float(np.max(np.abs(shifts[:,1]+delta))),validation_error=error,validation_coverage=coverage)

    def close(self):
        if self.previous:self.previous.release();self.previous=None
        self.lines=[];self.masks=()
