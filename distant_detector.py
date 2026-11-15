"""Whole-field detections plus overlapping magnified passes over the far field."""
import numpy as np
from evaluate_detector import iou


def merge_detections(rows, overlap=.5):
    kept=[]
    for row in sorted(rows,key=lambda r:r['confidence'],reverse=True):
        if not any(iou(row['box'],other['box'])>overlap for other in kept):
            kept.append(row)
    return kept


def predict_robots(model, image, device='mps', confidence=.25, tile_confidence=.4, scale=.65):
    h,w=image.shape[:2]
    rows=[]
    windows=[(0,0,w,h,confidence)]
    tile_w=round(w*scale);tile_h=round(h*.65)
    for left in (() if scale>=1 else (0,(w-tile_w)//2,w-tile_w)):
        windows.append((left,0,left+tile_w,tile_h,tile_confidence))
    for left,top,right,bottom,threshold in windows:
        result=model.predict(image[top:bottom,left:right],imgsz=960,conf=threshold,device=device,verbose=False,max_det=30)[0]
        for box,score in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist()):
            # Reject truncated tile-edge boxes; another overlapping tile can see them.
            if (left or right<w) and (box[0]<4 or box[2]>right-left-4 or box[3]>bottom-top-4):
                continue
            rows.append({'box':[box[0]+left,box[1]+top,box[2]+left,box[3]+top], 'confidence':score,'class_id':0})
    return merge_detections(rows)


def as_tracker_boxes(rows):
    from ultralytics.engine.results import Boxes
    return np.array([r['box']+[r['confidence'],0] for r in rows],dtype=np.float32).reshape(-1,6)
