import torch
import os
import json
import copy
import numpy as np
from PIL import Image
from random import randint
from tqdm import tqdm
from dataloader.dynamic_dataset.helpers import params2rendervar
import cv2
from typing import Any, Dict, List, Optional
import imageio.v2 as imageio


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        data_path: str = "/home/anurag/datasets/dynamic/data",
        seq: str = "basketball",
        t: int = 0,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_seg: bool = False,
    ):
        self.data_path = data_path
        self.seq = seq
        self.split = split
        self.patch_size = patch_size
        self.load_seg = load_seg
        if split == "train":
            self.md = json.load(open(os.path.join(data_path, seq, "train_meta.json"), 'r'))
        else:
            self.md = json.load(open(os.path.join(data_path, seq, "test_meta.json"), 'r'))
        self.t = t
        self.file_names = self.md['fn'][t]
        self.indices = list(range(len(self.md['fn'][self.t])))

    def __len__(self):
        return len(self.md['fn'][self.t])
    
    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]

        image_filename = self.file_names[index]
        image = imageio.imread(os.path.join(self.data_path, self.seq, "ims", image_filename))[..., :3]
        w, h, k, w2c = self.md['w'], self.md['h'], self.md['k'][self.t][index], self.md['w2c'][self.t][index]
        data = {
            "K": torch.from_numpy(np.array(k)).float(),
            "camtoworld": torch.from_numpy(np.linalg.inv(w2c)).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }
        

        if self.load_seg and index > 0:
            # projected points to image plane to get depths
            image_filename = self.file_names[index].replace('.jpg', '.png')
            seg = imageio.imread(os.path.join(self.data_path, self.seq, "seg", image_filename))
            data["seg"] = torch.from_numpy(seg) > 0.5

        return data
    