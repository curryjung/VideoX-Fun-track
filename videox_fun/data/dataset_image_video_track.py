import os
import random
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .dataset_image_video import ImageVideoDataset


class ImageVideoDatasetTrack(ImageVideoDataset):
    def __init__(
        self,
        ann_path,
        data_root=None,
        video_sample_size=512,
        video_sample_stride=4,
        video_sample_n_frames=16,
        image_sample_size=512,
        video_repeat=0,
        text_drop_ratio=0.1,
        enable_bucket=False,
        video_length_drop_start=0.0,
        video_length_drop_end=1.0,
        enable_inpaint=False,
        return_file_name=False,
        track_condition_key: str = "track_condition",
        track_file_key: str = "track_file_path",
        track_required: bool = True,
        root_map: Optional[Dict[str, str]] = None,
        root_id_key: str = "root_id",
    ):
        super().__init__(
            ann_path=ann_path,
            data_root=data_root,
            video_sample_size=video_sample_size,
            video_sample_stride=video_sample_stride,
            video_sample_n_frames=video_sample_n_frames,
            image_sample_size=image_sample_size,
            video_repeat=video_repeat,
            text_drop_ratio=text_drop_ratio,
            enable_bucket=enable_bucket,
            video_length_drop_start=video_length_drop_start,
            video_length_drop_end=video_length_drop_end,
            enable_inpaint=enable_inpaint,
            return_file_name=return_file_name,
        )
        self.track_condition_key = track_condition_key
        self.track_file_key = track_file_key
        self.track_required = track_required
        self.root_id_key = root_id_key
        self.root_map = self._normalize_root_map(root_map)

    def _normalize_root_map(self, root_map: Optional[Dict[str, str]]) -> Dict[str, str]:
        if root_map is None:
            return {}
        normalized: Dict[str, str] = {}
        for key, value in root_map.items():
            if key is None:
                continue
            key_str = str(key).strip()
            value_str = str(value).strip()
            if key_str == "" or value_str == "":
                continue
            normalized[key_str] = os.path.abspath(value_str)
        return normalized

    def _resolve_data_path_with_root(
        self,
        data_path: str,
        data_info: Optional[Dict] = None,
    ) -> str:
        if os.path.isabs(data_path):
            return data_path
        if data_info is not None:
            root_id = data_info.get(self.root_id_key, None)
            if root_id is not None:
                root_key = str(root_id).strip()
                if root_key != "" and root_key in self.root_map:
                    return os.path.join(self.root_map[root_key], data_path)
        if self.data_root is None:
            return data_path
        return os.path.join(self.data_root, data_path)

    def _resolve_track_path(self, track_file_path: str, data_info: Optional[Dict] = None) -> str:
        return self._resolve_data_path_with_root(track_file_path, data_info=data_info)

    def _load_track_condition(self, track_file_path: str) -> Dict[str, torch.Tensor]:
        data = np.load(track_file_path)

        if "tracks_compressed" in data:
            tracks = data["tracks_compressed"]
        elif "tracks" in data:
            tracks = data["tracks"]
        else:
            raise KeyError(f"Track file missing tracks key: {track_file_path}")

        if "visibility_compressed" in data:
            visibility = data["visibility_compressed"]
        elif "visibility" in data:
            visibility = data["visibility"]
        else:
            visibility = np.ones(tracks.shape[:2], dtype=np.float32)

        if tracks.ndim == 4 and tracks.shape[0] == 1:
            tracks = tracks[0]
        if visibility.ndim == 3 and visibility.shape[0] == 1:
            visibility = visibility[0]

        if tracks.ndim != 3 or tracks.shape[-1] != 2:
            raise ValueError(f"Unexpected tracks shape {tracks.shape} in {track_file_path}")
        if visibility.ndim != 2:
            raise ValueError(f"Unexpected visibility shape {visibility.shape} in {track_file_path}")

        return {
            "tracks": torch.as_tensor(tracks, dtype=torch.float32),
            "visibility": torch.as_tensor(visibility, dtype=torch.float32),
        }

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        data_info = self.dataset[sample["idx"] % len(self.dataset)]
        track_file_path: Optional[str] = data_info.get(self.track_file_key, None)

        if track_file_path is None or track_file_path == "":
            if self.track_required:
                raise KeyError(
                    f"Missing `{self.track_file_key}` in metadata row index={sample['idx']}."
                )
            sample[self.track_condition_key] = None
            return sample

        resolved_path = self._resolve_track_path(track_file_path, data_info=data_info)
        if not os.path.exists(resolved_path):
            if self.track_required:
                raise FileNotFoundError(f"Track file not found: {resolved_path}")
            sample[self.track_condition_key] = None
            return sample

        sample[self.track_condition_key] = self._load_track_condition(resolved_path)
        return sample


class ImageVideoLatentTrackDataset(ImageVideoDatasetTrack):
    """Track dataset for precomputed VAE latents.

    Metadata keys:
    - text
    - track_file_path
    - latent_file_path (preferred) or file_path (fallback)
    - text_feature_file_path (optional)
    - clip_feature_file_path (optional)
    - first_frame_vae_latent_file_path (optional)
    """

    def __init__(
        self,
        ann_path,
        data_root=None,
        text_drop_ratio: float = 0.1,
        track_condition_key: str = "track_condition",
        track_file_key: str = "track_file_path",
        latent_file_key: str = "latent_file_path",
        text_feature_file_key: str = "text_feature_file_path",
        clip_feature_file_key: str = "clip_feature_file_path",
        first_frame_file_key: str = "first_frame_file_path",
        first_frame_vae_latent_file_key: str = "first_frame_vae_latent_file_path",
        track_required: bool = True,
        root_map: Optional[Dict[str, str]] = None,
        root_id_key: str = "root_id",
    ):
        # Keep parent constructor for annotation loading and retry behavior.
        super().__init__(
            ann_path=ann_path,
            data_root=data_root,
            video_sample_size=512,
            video_sample_stride=1,
            video_sample_n_frames=1,
            image_sample_size=512,
            video_repeat=0,
            text_drop_ratio=text_drop_ratio,
            enable_bucket=False,
            video_length_drop_start=0.0,
            video_length_drop_end=1.0,
            enable_inpaint=False,
            return_file_name=False,
            track_condition_key=track_condition_key,
            track_file_key=track_file_key,
            track_required=track_required,
            root_map=root_map,
            root_id_key=root_id_key,
        )
        self.latent_file_key = latent_file_key
        self.text_feature_file_key = text_feature_file_key
        self.clip_feature_file_key = clip_feature_file_key
        self.first_frame_file_key = first_frame_file_key
        self.first_frame_vae_latent_file_key = first_frame_vae_latent_file_key

    def _resolve_data_path(self, data_path: str, data_info: Optional[Dict] = None) -> str:
        return self._resolve_data_path_with_root(data_path, data_info=data_info)

    def _load_latents(self, latent_path: str) -> torch.Tensor:
        latents = torch.load(latent_path, map_location="cpu")
        if not isinstance(latents, torch.Tensor):
            raise ValueError(f"Expected Tensor in latent file: {latent_path}")
        if latents.ndim == 5 and latents.shape[0] == 1:
            latents = latents[0]
        if latents.ndim != 4:
            raise ValueError(f"Unexpected latent shape {tuple(latents.shape)} in {latent_path}")
        return latents.float()

    def _load_text_feature(self, text_feature_path: str) -> Optional[Dict[str, torch.Tensor]]:
        if not os.path.isfile(text_feature_path):
            return None
        data = np.load(text_feature_path, allow_pickle=True)
        if "prompt_embeds" not in data:
            return None

        prompt_embeds = torch.as_tensor(data["prompt_embeds"], dtype=torch.float32)
        if prompt_embeds.ndim != 2:
            return None

        attention_mask = None
        if "attention_mask" in data:
            attention_mask = torch.as_tensor(data["attention_mask"], dtype=torch.long).view(-1)
            # prompt_embeds is trimmed [L, D], so keep first L tokens.
            attention_mask = attention_mask[: prompt_embeds.shape[0]]
        else:
            attention_mask = torch.ones((prompt_embeds.shape[0],), dtype=torch.long)
        return {
            "prompt_embeds": prompt_embeds,
            "attention_mask": attention_mask,
        }

    def _load_clip_feature(self, clip_feature_path: str) -> Optional[torch.Tensor]:
        if not os.path.isfile(clip_feature_path):
            return None
        data = np.load(clip_feature_path, allow_pickle=True)
        if "clip_feature" not in data:
            return None
        clip_feature = torch.as_tensor(data["clip_feature"], dtype=torch.float32)
        if clip_feature.ndim != 2:
            return None
        return clip_feature

    def _load_first_frame(self, first_frame_path: str) -> Optional[torch.Tensor]:
        if not os.path.isfile(first_frame_path):
            return None
        try:
            image = Image.open(first_frame_path).convert("RGB")
            arr = np.asarray(image, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        except Exception:
            return None

    def _load_first_frame_vae_latent(self, first_frame_vae_latent_path: str) -> Optional[torch.Tensor]:
        if not os.path.isfile(first_frame_vae_latent_path):
            return None
        try:
            latent = torch.load(first_frame_vae_latent_path, map_location="cpu")
        except Exception:
            return None
        if not isinstance(latent, torch.Tensor):
            return None
        if latent.ndim == 5 and latent.shape[0] == 1:
            latent = latent[0]
        if latent.ndim != 4:
            return None
        return latent.float()

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]

        while True:
            sample = {}
            try:
                data_info = self.dataset[idx % len(self.dataset)]
                text = data_info.get("text", "")
                if random.random() < self.text_drop_ratio:
                    text = ""

                latent_file_path = data_info.get(self.latent_file_key, data_info.get("file_path", None))
                if latent_file_path is None or latent_file_path == "":
                    raise KeyError(
                        f"Missing `{self.latent_file_key}` or `file_path` in metadata row index={idx}."
                    )
                resolved_latent_path = self._resolve_data_path(latent_file_path, data_info=data_info)
                if not os.path.exists(resolved_latent_path):
                    raise FileNotFoundError(f"Latent file not found: {resolved_latent_path}")

                sample["latents"] = self._load_latents(resolved_latent_path)
                sample["text"] = text
                sample["data_type"] = "latent"
                sample["idx"] = idx

                latent_dir = os.path.dirname(resolved_latent_path)
                text_feature_path = data_info.get(
                    self.text_feature_file_key, os.path.join(latent_dir, "text_feature_wan_t5.npz")
                )
                clip_feature_path = data_info.get(
                    self.clip_feature_file_key, os.path.join(latent_dir, "first_frame_clip_feature.npz")
                )
                first_frame_path = data_info.get(
                    self.first_frame_file_key, os.path.join(latent_dir, "first_frame.png")
                )
                first_frame_vae_latent_path = data_info.get(
                    self.first_frame_vae_latent_file_key,
                    os.path.join(latent_dir, "first_frame_vae_latent.pt"),
                )
                text_feature_path = self._resolve_data_path(text_feature_path, data_info=data_info)
                clip_feature_path = self._resolve_data_path(clip_feature_path, data_info=data_info)
                first_frame_path = self._resolve_data_path(first_frame_path, data_info=data_info)
                first_frame_vae_latent_path = self._resolve_data_path(
                    first_frame_vae_latent_path,
                    data_info=data_info,
                )

                text_feature = self._load_text_feature(text_feature_path)
                if text_feature is not None:
                    sample["precomputed_prompt_embeds"] = text_feature["prompt_embeds"]
                    sample["precomputed_attention_mask"] = text_feature["attention_mask"]
                clip_feature = self._load_clip_feature(clip_feature_path)
                if clip_feature is not None:
                    sample["precomputed_clip_feature"] = clip_feature
                first_frame = self._load_first_frame(first_frame_path)
                if first_frame is not None:
                    sample["first_frame_pixel_values"] = first_frame
                first_frame_vae_latent = self._load_first_frame_vae_latent(first_frame_vae_latent_path)
                if first_frame_vae_latent is not None:
                    sample["first_frame_vae_latent"] = first_frame_vae_latent

                track_file_path: Optional[str] = data_info.get(self.track_file_key, None)
                if track_file_path is None or track_file_path == "":
                    if self.track_required:
                        raise KeyError(
                            f"Missing `{self.track_file_key}` in metadata row index={idx}."
                        )
                    sample[self.track_condition_key] = None
                else:
                    resolved_track_path = self._resolve_track_path(track_file_path, data_info=data_info)
                    if not os.path.exists(resolved_track_path):
                        if self.track_required:
                            raise FileNotFoundError(f"Track file not found: {resolved_track_path}")
                        sample[self.track_condition_key] = None
                    else:
                        sample[self.track_condition_key] = self._load_track_condition(resolved_track_path)
                break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length - 1)
        return sample


class DummyTrackLatentDataset(Dataset):
    """Dummy dataset for dataloader smoke tests.

    This dataset emits random latent tensors and random track conditions
    compatible with train_track.py latent mode.
    """

    def __init__(
        self,
        length: int = 128,
        latent_shape=(16, 21, 60, 104),
        n_frames: int = 81,
        n_points: int = 50,
        text: str = "dummy track prompt",
    ):
        self.length = int(length)
        self.latent_shape = latent_shape
        self.n_frames = int(n_frames)
        self.n_points = int(n_points)
        self.text = text

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        del idx
        latents = torch.randn(self.latent_shape, dtype=torch.float32)
        tracks = torch.rand((self.n_frames, self.n_points, 2), dtype=torch.float32)
        visibility = torch.ones((self.n_frames, self.n_points), dtype=torch.float32)
        prompt_embeds = torch.randn((151, 4096), dtype=torch.float32)
        attention_mask = torch.ones((151,), dtype=torch.long)
        clip_feature = torch.randn((257, 1280), dtype=torch.float32)
        return {
            "latents": latents,
            "text": self.text,
            "data_type": "latent",
            "idx": 0,
            "precomputed_prompt_embeds": prompt_embeds,
            "precomputed_attention_mask": attention_mask,
            "precomputed_clip_feature": clip_feature,
            "track_condition": {
                "tracks": tracks,
                "visibility": visibility,
            },
        }
