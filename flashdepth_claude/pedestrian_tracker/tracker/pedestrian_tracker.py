"""
YOLO 11m-seg Instance Segmentation + Tracking Wrapper

Provides per-frame pedestrian detection with instance segmentation masks
and multi-object tracking (BoTSORT).
"""

import numpy as np
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class PedestrianTracker:
    """
    YOLO 11m-seg based pedestrian tracker.

    Uses instance segmentation for pixel-level masks and
    BoTSORT for persistent tracking across frames.
    """

    def __init__(self, model_path='yolo11m-seg.pt', confidence=0.5,
                 iou_threshold=0.5, tracker='botsort.yaml', person_only=True,
                 imgsz=None):
        """
        Args:
            model_path: Path to YOLO model weights
            confidence: Detection confidence threshold
            iou_threshold: NMS IoU threshold
            tracker: Tracker config file name
            person_only: If True, only track person class (class 0)
            imgsz: YOLO input resolution [W, H] or int. None = use frame size as-is.
        """
        logger.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.tracker = tracker
        self.classes = [0] if person_only else None
        self.imgsz = imgsz

    def track(self, frame_bgr):
        """
        Run detection + tracking on a single frame.

        Args:
            frame_bgr: [H, W, 3] uint8 BGR numpy array

        Returns:
            list of dicts, each with:
                track_id: int
                class_id: int
                bbox: [x1, y1, x2, y2] numpy array
                mask_points: list of polygon points (float coords)
                confidence: float
            Returns empty list if no detections.
        """
        track_kwargs = dict(
            persist=True,
            classes=self.classes,
            tracker=self.tracker,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False,
        )
        if self.imgsz is not None:
            track_kwargs['imgsz'] = self.imgsz

        results = self.model.track(frame_bgr, **track_kwargs)

        result = results[0]

        # No detections or no tracking IDs
        if result.boxes.id is None or len(result.boxes.id) == 0:
            return []

        if result.masks is None:
            return []

        detections = []
        track_ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        masks_xy = result.masks.xy  # list of polygon points per detection

        for track_id, cls, box, conf, mask_pts in zip(
            track_ids, classes, boxes, confs, masks_xy
        ):
            detections.append({
                'track_id': track_id,
                'class_id': cls,
                'bbox': box,
                'mask_points': mask_pts,
                'confidence': float(conf),
            })

        return detections
