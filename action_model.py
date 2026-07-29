import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

def get_mean_std(model_name):
    for import_path, cls_name in [
        ("transformers", "AutoVideoProcessor"),
        ("transformers", "AutoImageProcessor"),
        ("transformers", "VideoMAEImageProcessor"),
    ]:
        try:
            module = __import__(import_path, fromlist=[cls_name])
            proc_cls = getattr(module, cls_name)
            proc = proc_cls.from_pretrained(model_name)
            return np.array(proc.image_mean, dtype=np.float32), np.array(proc.image_std, dtype=np.float32)
        except Exception:
            continue
    print(f"  [warning] no processor found for {model_name}, using generic (0.5,0.5,0.5) normalization.")
    return np.array([0.5, 0.5, 0.5], dtype=np.float32), np.array([0.5, 0.5, 0.5], dtype=np.float32)


class VideoMAEAdapter(nn.Module):
    def __init__(self, model_name, input_size=224, num_frames=16, patch_size=16, tubelet_size=2):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=config, low_cpu_mem_usage=False)

        if hasattr(self.backbone, "predictor") and self.backbone.predictor is not None:
            for p in self.backbone.predictor.parameters():
                p.requires_grad = False

        self.patch_size = getattr(config, "patch_size", patch_size)
        self.tubelet_size = getattr(config, "tubelet_size", tubelet_size)
        self.grid_size = input_size // self.patch_size
        self.input_size = input_size
        self.num_frames = num_frames
        self.mean, self.std = get_mean_std(model_name)

        expected = self.grid_size * self.grid_size * (num_frames // self.tubelet_size)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, num_frames, input_size, input_size)
            tokens, self.pixel_key, self.layout = self._probe_forward(dummy)
            n_tokens = tokens.shape[1]
            if n_tokens == expected:
                self.drop_cls = False
            elif n_tokens == expected + 1:
                self.drop_cls = True
            else:
                raise ValueError(
                    f"{model_name}: got {n_tokens} tokens, expected {expected} "
                    f"(or {expected + 1} with a CLS token) given grid_size={self.grid_size}, "
                    f"tubelet_size={self.tubelet_size}, num_frames={num_frames}."
                )
            self.out_channels = tokens.shape[-1]

        self.spatial_scale = self.grid_size / input_size
        print(f"  [{model_name}] pixel_key={self.pixel_key!r} layout={self.layout} "
              f"drop_cls={self.drop_cls} out_channels={self.out_channels} grid={self.grid_size}")

    def _probe_forward(self, dummy):
        candidates = [
            ("pixel_values", "BCTHW"),
            ("pixel_values", "BTCHW"),
            ("pixel_values_videos", "BCTHW"),
            ("pixel_values_videos", "BTCHW"),
        ]
        errors = []
        for key, layout in candidates:
            try:
                x = dummy if layout == "BCTHW" else dummy.permute(0, 2, 1, 3, 4)
                out = self.backbone(**{key: x})
                tokens = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                if tokens.dim() == 3:
                    return tokens, key, layout
            except Exception as e:
                errors.append(f"  {key}/{layout} -> {type(e).__name__}: {e}")
        raise RuntimeError(
            "Could not find a working (pixel_key, layout) combination for this backbone.\n"
            + "\n".join(errors)
        )

    def forward(self, clips):
        x = clips if self.layout == "BCTHW" else clips.permute(0, 2, 1, 3, 4)
        out = self.backbone(**{self.pixel_key: x})
        tokens = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        if self.drop_cls:
            tokens = tokens[:, 1:, :]

        B, N, C = tokens.shape
        T_ = N // (self.grid_size * self.grid_size)
        feat = tokens.view(B, T_, self.grid_size, self.grid_size, C)
        return feat.permute(0, 4, 1, 2, 3)


class RoIActionModel(nn.Module):
    def __init__(self, num_classes, model_name, input_size=224, num_frames=16,
                 patch_size=16, tubelet_size=2, temporal_hidden_dim=256):
        super().__init__()
        self.adapter = VideoMAEAdapter(
            model_name, input_size=input_size, num_frames=num_frames,
            patch_size=patch_size, tubelet_size=tubelet_size,
        )
        self.out_channels = self.adapter.out_channels
        self.num_frames = self.adapter.num_frames
        self.input_size = self.adapter.input_size
        self.mean = self.adapter.mean
        self.std = self.adapter.std

        self.temporal = nn.GRU(self.out_channels, temporal_hidden_dim, dropout=0.3, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(temporal_hidden_dim, num_classes),
        )

    def forward(self, clips):
        feat = self.adapter(clips)
        per_frame = feat.mean(dim=[3, 4])
        per_frame = per_frame.permute(0, 2, 1)
        _, h = self.temporal(per_frame)
        return self.classifier(h[-1])


def load_action_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    action_cfg = ckpt["cfg"]

    action_classes = action_cfg["CLASSES"]
    model = RoIActionModel(
        num_classes=len(action_classes),
        model_name=action_cfg["MODEL_NAME"],
        input_size=action_cfg["INPUT_SIZE"],
        num_frames=action_cfg["NUM_FRAMES"],
        patch_size=action_cfg["PATCH_SIZE"],
        tubelet_size=action_cfg["TUBELET_SIZE"],
        temporal_hidden_dim=action_cfg["TEMPORAL_HIDDEN_DIM"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    meta = {
        "classes": action_classes,
        "padding_ratio": action_cfg.get("PADDING_RATIO", 0.0),
        "num_frames": model.num_frames,
        "input_size": model.input_size,
        "mean": model.mean,
        "std": model.std,
    }
    return model, meta