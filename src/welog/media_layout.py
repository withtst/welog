"""Conservative screenshot evidence from dense small UI/text components, not background color."""
import cv2
import numpy as np


def screenshot_structure(patch):
    h,w=patch.shape[:2]
    if w<80 or h<55:return None
    gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
    if float((gray>=230).mean())<.5:return None
    _,_,stats,centers=cv2.connectedComponentsWithStats((gray<200).astype(np.uint8))
    stats,centers=stats[1:],centers[1:]
    valid=(stats[:,4]>=2)&(stats[:,3]>=2)
    stats,centers=stats[valid],centers[valid]
    if len(stats)<16:return None
    tiny_height=max(5,min(10,round(min(h,w)*.08)))
    tiny=(stats[:,3]<=tiny_height)&(stats[:,2]<=max(60,w*.25))
    if tiny.sum()<16 or tiny.mean()<.7:return None
    spread=np.ptp(centers[tiny],axis=0)/[w,h]
    if spread[0]<.35 or spread[1]<.25:return None
    return {'method':'dense_small_components','components':len(stats),'tiny_components':int(tiny.sum()),
            'tiny_ratio':float(tiny.mean()),'spread_x':float(spread[0]),'spread_y':float(spread[1])}
