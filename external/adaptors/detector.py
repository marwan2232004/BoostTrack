import os
import pickle

import torch

from external.adaptors import yolox_adaptor
from ultralytics.utils.nms import non_max_suppression
from ultralytics import YOLO
class Detector(torch.nn.Module):
    K_MODELS = {"yolox", "yolov26"}

    def __init__(self, model_type, path, dataset, size):
        super().__init__()
        if model_type not in self.K_MODELS:
            raise RuntimeError(f"{model_type} detector not supported")

        self.model_type = model_type
        self.path = path
        self.dataset = dataset
        self.model = None
        self.size = size

        os.makedirs("./cache", exist_ok=True)
        self.cache_path = os.path.join(
            "./cache", f"det_{os.path.basename(path).split('.')[0]}.pkl"
        )
        self.cache = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as fp:
                self.cache = pickle.load(fp)
        else:
            self.initialize_model()

    def initialize_model(self):
        """Wait until needed."""
        if self.model_type == "yolox":
            self.model = yolox_adaptor.get_model(self.path, self.dataset, self.size)
            
        elif self.model_type == "yolov26":
            full_model = YOLO(self.path)
            self.model = full_model.model.eval().half()

    def forward(self, batch, tag=None):
        if tag in self.cache:
            return self.cache[tag]
        
        if self.model is None:
            self.initialize_model()

        with torch.no_grad():
            batch = batch.half()
            self.model.to(batch.device)
            output = self.model(batch)
            
            if self.model_type == "yolov26" and isinstance(output, tuple):
                
                
                # Extract the prediction tensor if it's a tuple
                preds = output[0] if isinstance(output, tuple) else output
                
                # Define the class ID for 'Worker'. 
                # (Change this to 0 or 1 if your data.yaml defines them in a different order)
                WORKER_CLASS_ID = 0
                
                # Apply NMS. This automatically filters boxes and converts 
                # coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).
                # NMS returns a list of tensors (one per batch item). We take [0].
                nms_predictions = non_max_suppression(
                    preds, 
                    conf_thres=0.1,  
                    iou_thres=0.7,
                    classes=[WORKER_CLASS_ID]
                )
                
                # Safeguard against empty batch outputs
                output = nms_predictions[0] if len(nms_predictions) > 0 else torch.empty((0, 6), device=preds.device)

        if output is not None:
            self.cache[tag] = output.cpu().detach()

        return output

    def dump_cache(self):
        with open(self.cache_path, "wb") as fp:
            pickle.dump(self.cache, fp)