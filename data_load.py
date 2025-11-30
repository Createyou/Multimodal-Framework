import os
import glob
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset
from PIL import Image


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


class MultiModalImagePtDataset(Dataset):


    def __init__(
        self,
        images_path: List[str],
        images_class: List[int],
        signal_pt_paths: Union[Sequence[str], Dict[int, str]],
        transform=None,
        signal_transform=None,
        dtype: torch.dtype = torch.float32,
        per_class_start_index: Optional[Dict[int, int]] = None,
        preload_signals: bool = True,
    ):


        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        self.signal_transform = signal_transform
        self.dtype = dtype

        if isinstance(signal_pt_paths, (list, tuple)):
            self.signal_pt_paths = {i: p for i, p in enumerate(signal_pt_paths)}
        else:
            self.signal_pt_paths = dict(signal_pt_paths)


        self.signals_by_class: Dict[int, torch.Tensor] = {}
        if preload_signals:
            for cls_idx, pt_path in self.signal_pt_paths.items():
                self.signals_by_class[cls_idx] = self._load_signals_from_pt(pt_path)
        else:
            self.signals_by_class = {}

        self._occurrence_index: List[int] = []
        occurrence_count: Dict[int, int] = {}
        self._start_offset = per_class_start_index or {}

        for lbl in self.images_class:
            j = occurrence_count.get(lbl, 0)
            self._occurrence_index.append(j + self._start_offset.get(lbl, 0))
            occurrence_count[lbl] = j + 1


        self.per_class_img_count: Dict[int, int] = {}
        for lbl in self.images_class:
            self.per_class_img_count[lbl] = self.per_class_img_count.get(lbl, 0) + 1


    def _to_tensor_float(self, obj) -> torch.Tensor:
        if isinstance(obj, (list, tuple)):
            ten = torch.stack([torch.as_tensor(x) for x in obj], dim=0)
        else:
            ten = torch.as_tensor(obj)
        if ten.dtype.is_floating_point:
            return ten.to(self.dtype)
        else:
            return ten.to(torch.float32).to(self.dtype)


    def _load_signals_from_pt(self, pt_path: str) -> torch.Tensor:

        obj = torch.load(pt_path, map_location="cpu")


        if isinstance(obj, (list, tuple)) or torch.is_tensor(obj):
            return self._to_tensor_float(obj)

        if isinstance(obj, dict):

            if "samples" in obj:
                return self._to_tensor_float(obj["samples"])


            for top_key in ("train_dataset", "val_dataset", "dataset", "data"):
                if top_key in obj and isinstance(obj[top_key], dict) and "samples" in obj[top_key]:
                    return self._to_tensor_float(obj[top_key]["samples"])


            for key in ("data", "x", "signals"):
                if key in obj:
                    return self._to_tensor_float(obj[key])

            raise ValueError(f"Unable to find 'samples' (or compatibility key) in the dictionary {pt_path}")

        raise TypeError(f"Unsupported pt content type: {type(obj)} from {pt_path}")

    def __len__(self) -> int:
        return len(self.images_path)

    def _get_signal_tensor(self, cls_idx: int, idx_in_class: int) -> torch.Tensor:
        # 惰性加载
        if cls_idx not in self.signals_by_class:
            self.signals_by_class[cls_idx] = self._load_signals_from_pt(self.signal_pt_paths[cls_idx])

        sigs = self.signals_by_class[cls_idx]


        if isinstance(sigs, torch.Tensor):
            n = sigs.size(0)
        else:
            n = len(sigs)


        if idx_in_class < 0 or idx_in_class >= n:
            raise IndexError(
                f"[get_signal_tensor] cls_idx={cls_idx}, "
                f"idx_in_class={idx_in_class}, valid range=0..{n - 1}"
            )

        sig_tensor = sigs[idx_in_class]

        if self.signal_transform is not None:
            sig_tensor = self.signal_transform(sig_tensor)
        return sig_tensor

    def __getitem__(self, index: int):
        img_path = self.images_path[index]
        label = int(self.images_class[index])

        # 读图
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)


        idx_in_class = self._occurrence_index[index]
        signal = self._get_signal_tensor(label, idx_in_class)

        return img, signal, label

    @staticmethod
    def collate_fn(batch):
        imgs, sigs, labels = tuple(zip(*batch))
        imgs = torch.stack(imgs, dim=0)
        sigs = torch.stack(sigs, dim=0)
        labels = torch.as_tensor(labels, dtype=torch.long)
        return imgs, sigs, labels


    @classmethod
    def from_folders(
        cls,
        img_root: str,
        sig_pt_dir: str,
        transform=None,
        signal_transform=None,
        dtype: torch.dtype = torch.float32,
        per_class_start_index: Optional[Dict[int, int]] = None,
        preload_signals: bool = True,
    ):

        classes = sorted([d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))])
        class_to_idx = {c: i for i, c in enumerate(classes)}

        # 收集所有图像路径与标签
        images_path: List[str] = []
        images_class: List[int] = []

        for cname in classes:
            cdir = os.path.join(img_root, cname)
            img_files = []
            for ext in IMG_EXTENSIONS:
                img_files += glob.glob(os.path.join(cdir, f"*{ext}"))
            img_files = sorted(img_files)
            images_path.extend(img_files)
            images_class.extend([class_to_idx[cname]] * len(img_files))


        signal_pt_paths: Dict[int, str] = {}
        for cname, idx in class_to_idx.items():
            p = os.path.join(sig_pt_dir, f"{cname}.pt")
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Signal file not found: {p}")
            signal_pt_paths[idx] = p

        return cls(
            images_path=images_path,
            images_class=images_class,
            signal_pt_paths=signal_pt_paths,
            transform=transform,
            signal_transform=signal_transform,
            dtype=dtype,
            per_class_start_index=per_class_start_index,
            preload_signals=preload_signals,
        )



