import os
import pickle

import torch

from external.adaptors import yolox_adaptor

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
            try:
                from ultralytics import YOLO
            except ImportError:
                raise ImportError("The 'ultralytics' package is required to load YOLOv26 .pt files. Run: pip install ultralytics")
            
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
                if isinstance(output, tuple):
                    output = output[0]
                
                # Strip the batch dimension: (1, 300, 6) -> (300, 6)
                if output.ndim == 3:
                    output = output.squeeze(0)
                
                # Ultralytics raw tensors are sometimes transposed to (6, 300).
                # If rows are fewer than columns, transpose it so detections are the rows.
                if len(output.shape) == 2 and output.shape[0] < output.shape[1]:
                    output = output.transpose(0, 1)

        if output is not None:
            self.cache[tag] = output.cpu().detach()

        return output

    def dump_cache(self):
        with open(self.cache_path, "wb") as fp:
            pickle.dump(self.cache, fp)