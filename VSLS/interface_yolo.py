
import os
import cv2
import os.path as osp
# from mmengine.config import Config
# from mmengine.dataset import Compose
# from mmdet.apis import init_detector
# from mmdet.utils import get_test_pipeline_cfg
# # from mmengine.runner.amp import autocast
from torch.amp import autocast
import torch
# import supervision as sv
from typing import Dict, Optional, Sequence, List

import supervision as sv
from supervision.draw.color import Color, ColorPalette

class LabelAnnotator(sv.LabelAnnotator):

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


class YoloInterface:
    def __init__(self):
        """
        Initialize the YOLO-World model with the given configuration and checkpoint.

        Args:
        """
        
     
        pass
    def set_BBoxAnnotator(self):
        self.BOUNDING_BOX_ANNOTATOR = sv.BoxAnnotator(thickness=1)
        # MASK_ANNOTATOR = sv.MaskAnnotator()
        self.LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                        text_scale=0.5,
                                        text_thickness=1,
                                        smart_position=True,
                                        color=ColorPalette.LEGACY)

# class YoloWorldInterface(YoloInterface):
#     def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda:0"):
#         """
#         Initialize the YOLO-World model with the given configuration and checkpoint.

#         Args:
#             config_path (str): Path to the model configuration file.
#             checkpoint_path (str): Path to the model checkpoint.
#             device (str): Device to run the model on (e.g., 'cuda:0', 'cpu').
#         """
#         self.config_path = config_path
#         self.checkpoint_path = checkpoint_path
#         self.device = device

#         # Load configuration
#         cfg = Config.fromfile(config_path)
#         cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config_path))[0])
#         cfg.load_from = checkpoint_path

#         # Initialize the model
#         self.model = init_detector(cfg, checkpoint=checkpoint_path, device=device)
#         self.set_BBoxAnnotator()

#         # Initialize the test pipeline
#         # build test pipeline
#         self.model.cfg.test_dataloader.dataset.pipeline[
#             0].type = 'mmdet.LoadImageFromNDArray'
#         self.test_pipeline = Compose(self.model.cfg.test_dataloader.dataset.pipeline)

        

#     def reparameterize_object_list(self, target_objects: List[str], cue_objects: List[str]):
#         """
#         Reparameterize the detect object list to be used by the YOLO model.

#         Args:
#             target_objects (List[str]): List of target object names.
#             cue_objects (List[str]): List of cue object names.
#         """
#         # Combine target objects and cue objects into the final text format
#         combined_texts = target_objects + cue_objects

#         # Format the text prompts for the YOLO model
#         self.texts = [[obj.strip()] for obj in combined_texts] + [[' ']]

#         # Reparameterize the YOLO model with the provided text prompts
#         self.model.reparameterize(self.texts)


#     def inference(self, image: str, max_dets: int = 100, score_threshold: float = 0.3, use_amp: bool = False):
#         """
#         Run inference on a single image.

#         Args:
#             image (str): Path to the image.
#             max_dets (int): Maximum number of detections to keep.
#             score_threshold (float): Score threshold for filtering detections.
#             use_amp (bool): Whether to use mixed precision for inference.

#         Returns:
#             sv.Detections: Detection results.
#         """
#         # Prepare data for inference
#         data_info = dict(img_id=0, img_path=image, texts=self.texts)
#         data_info = self.test_pipeline(data_info)
#         data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
#                           data_samples=[data_info['data_samples']])

#         # Run inference
#         with autocast(enabled=use_amp), torch.no_grad():
#             output = self.model.test_step(data_batch)[0]
#             pred_instances = output.pred_instances
#             pred_instances = pred_instances[pred_instances.scores.float() > score_threshold]

#         if len(pred_instances.scores) > max_dets:
#             indices = pred_instances.scores.float().topk(max_dets)[1]
#             pred_instances = pred_instances[indices]

#         pred_instances = pred_instances.cpu().numpy()

#         # Process detections
#         detections = sv.Detections(
#             xyxy=pred_instances['bboxes'],
#             class_id=pred_instances['labels'],
#             confidence=pred_instances['scores'],
#             mask=pred_instances.get('masks', None)
#         )
#         return detections
    
#     def inference_detector(self, images, max_dets=50, score_threshold=0.2, use_amp: bool = False):
#         data_info = dict(img_id=0, img=images[0], texts=self.texts) #TBD for batch searching
#         data_info = self.test_pipeline(data_info)
#         data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
#                         data_samples=[data_info['data_samples']])
#         detections_inbatch = []
#         with torch.no_grad():
#             outputs = self.model.test_step(data_batch)
#             # cover to searcher interface format
            
#             for output in outputs:
#                 pred_instances = output.pred_instances
#                 pred_instances = pred_instances[pred_instances.scores.float() >
#                                                 score_threshold]
#                 if len(pred_instances.scores) > max_dets:
#                     indices = pred_instances.scores.float().topk(max_dets)[1]
#                     pred_instances = pred_instances[indices]

#                 output.pred_instances = pred_instances

#                 if 'masks' in pred_instances:
#                     masks = pred_instances['masks']
#                 else:
#                     masks = None
#                 pred_instances = pred_instances.cpu().numpy()
#                 detections = sv.Detections(xyxy=pred_instances['bboxes'],
#                     class_id=pred_instances['labels'],
#                     confidence=pred_instances['scores'],
#                     mask=masks)
#                 detections_inbatch.append(detections)
#         self.detect_outputs_raw = outputs
#         self.detections_inbatch = detections_inbatch
#         return detections_inbatch

#     def bbox_visualization(self, images, detections_inbatch):
#         anno_images = []
#         # detections_inbatch = self.detections_inbatch
#         for b, detections in enumerate(detections_inbatch):
#             texts = self.texts
#             labels = [
#                 f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
#                 zip(detections.class_id, detections.confidence)
#             ]

        
#             index = len(detections_inbatch) -1 
#             image = images[index]
#             anno_image = image.copy()
  
    
#             anno_image = self.BOUNDING_BOX_ANNOTATOR.annotate(anno_image, detections)
#             anno_image = self.LABEL_ANNOTATOR.annotate(anno_image, detections, labels=labels)
#             anno_images.append(anno_image)
        
#         return anno_images

# ============================================================
# ADD THIS CLASS TO: VSLS/interface_yolo.py
#
# Requirements:
#   pip install ultralytics supervision
#
# Usage (replaces YoloWorldInterface):
#   from VSLS.interface_yolo import UltralyticsYOLOWorldInterface
#   yolo_interface = UltralyticsYOLOWorldInterface(
#       checkpoint_path="yolov8x-worldv2.pt",
#       device="cuda:0"
#   )
# ============================================================

from ultralytics import YOLOWorld as UltralyticsYOLOWorld


class UltralyticsYOLOWorldInterface(YoloInterface):
    def __init__(self, checkpoint_path: str = "yolov8x-worldv2.pt", device: str = "cuda:0"):
        """
        Initialize the Ultralytics YOLO-World model.
        Drop-in replacement for YoloWorldInterface — no mmcv required.

        Args:
            checkpoint_path (str): Path to .pt checkpoint, or model name for auto-download.
                                   E.g. "yolov8x-worldv2.pt" (~100MB, auto-downloaded on first run).
            device (str): Device for inference (e.g., "cuda:0", "cpu").
        """
        self.checkpoint_path = checkpoint_path
        self.device = device

        self.model = UltralyticsYOLOWorld(checkpoint_path)
        self.model.to(device)

        # texts mirrors YoloWorldInterface.texts: list of [class_name] per index, plus a trailing [' ']
        # populated by reparameterize_object_list()
        self.texts = []

        self.set_BBoxAnnotator()

    # ------------------------------------------------------------------
    # Public interface — identical signatures to YoloWorldInterface
    # ------------------------------------------------------------------

    def reparameterize_object_list(self, target_objects: List[str], cue_objects: List[str]):
        """
        Set the open-vocabulary class list for the YOLO-World model.

        Args:
            target_objects (List[str]): Primary objects to detect.
            cue_objects    (List[str]): Contextual/cue objects.
        """
        combined = target_objects + cue_objects
        # Keep same nested-list format as YoloWorldInterface so that
        # interface_searcher.py can index: self.yolo.texts[label][0]
        self.texts = [[obj.strip()] for obj in combined] + [[' ']]

        # Ultralytics API: set_classes accepts a flat list of strings
        self.model.set_classes([obj.strip() for obj in combined])

    def inference_detector(self, images, max_dets: int = 50, score_threshold: float = 0.2, use_amp: bool = False):
        """
        Run detection on a batch of images (numpy arrays, BGR or RGB).
        Returns the same format as YoloWorldInterface.inference_detector():
            List[sv.Detections]  — one entry per image.

        Args:
            images          : List of np.ndarray images (H, W, 3).
            max_dets        : Maximum detections to keep per image.
            score_threshold : Confidence threshold.
            use_amp         : Ignored (kept for API compatibility).
        """
        detections_inbatch = []

        for image in images:
            results = self.model.predict(
                source=image,
                conf=score_threshold,
                max_det=max_dets,
                device=self.device,
                verbose=False,
            )[0]

            boxes = results.boxes
            if boxes is None or len(boxes) == 0:
                detections = sv.Detections.empty()
            else:
                xyxy       = boxes.xyxy.cpu().numpy()       
                confidence = boxes.conf.cpu().numpy()        
                class_id   = boxes.cls.cpu().numpy().astype(int)  

                detections = sv.Detections(
                    xyxy=xyxy,
                    confidence=confidence,
                    class_id=class_id,
                )

            detections_inbatch.append(detections)

        self.detections_inbatch = detections_inbatch
        return detections_inbatch

    def inference(self, image, max_dets: int = 100, score_threshold: float = 0.3, use_amp: bool = False):
        """Single-image wrapper (mirrors YoloWorldInterface.inference)."""
        return self.inference_detector([image], max_dets=max_dets, score_threshold=score_threshold)[0]

    def bbox_visualization(self, images, detections_inbatch):
        """Identical to YoloWorldInterface.bbox_visualization."""
        anno_images = []
        for b, detections in enumerate(detections_inbatch):
            labels = [
                f"{self.texts[class_id][0]} {confidence:.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]
            index = len(detections_inbatch) - 1
            image = images[index]
            anno_image = image.copy()
            anno_image = self.BOUNDING_BOX_ANNOTATOR.annotate(anno_image, detections)
            anno_image = self.LABEL_ANNOTATOR.annotate(anno_image, detections, labels=labels)
            anno_images.append(anno_image)
        return anno_images



