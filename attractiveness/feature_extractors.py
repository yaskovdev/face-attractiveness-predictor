"""Frozen pretrained image encoders that turn an aligned face into a feature vector."""
import cv2
import numpy as np
import torch

BACKBONES = ("clip_vit_l14", "dinov2_vit_l14")


class FeatureExtractor:
    def __init__(self, backbone, device=None):
        self.backbone = backbone
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if backbone == "clip_vit_l14":
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
            self._encode = model.encode_image
            mean, std = open_clip.OPENAI_DATASET_MEAN, open_clip.OPENAI_DATASET_STD
            self._model = model
        elif backbone == "dinov2_vit_l14":
            import timm
            model = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)
            self._encode = model
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            self._model = model
        else:
            raise ValueError(f"Unknown backbone {backbone}, expected one of {BACKBONES}")
        self._model.eval().to(self.device)
        self._mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)
        self._input_size = 224

    def _to_tensor(self, aligned_faces_bgr):
        batch = np.stack([
            cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), (self._input_size, self._input_size), interpolation=cv2.INTER_AREA)
            for face in aligned_faces_bgr
        ])
        tensor = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        return (tensor - self._mean) / self._std

    @torch.inference_mode()
    def extract(self, aligned_faces_bgr, batch_size=64):
        """Returns an (N, D) float32 array; features of the face and its mirror image are averaged."""
        features = []
        for start in range(0, len(aligned_faces_bgr), batch_size):
            tensor = self._to_tensor(aligned_faces_bgr[start:start + batch_size])
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
                embedding = (self._encode(tensor) + self._encode(torch.flip(tensor, dims=[3]))) / 2
            features.append(embedding.float().cpu().numpy())
        return np.concatenate(features)
