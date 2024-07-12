import os
import glob
import torch
import numpy as np
import cv2
from PIL import Image

class segmentation_dataset(torch.utils.data.Dataset):

    def __init__(self, root, mode="train"):
        assert mode in {"train", "val", "test"}
        self.root = root
        self.mode = mode
        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "labels")
        assert self._check_unique()
        self.filenames = self._read_name()

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):

        # filename = self.filenames[idx]
        # image_path = os.path.join(self.images_directory, filename + ".jpg")
        # mask_path = os.path.join(self.masks_directory, filename + ".png")
        image_path_ls = sorted(glob.glob(self.images_directory + '/*'))
        mask_path_ls = sorted(glob.glob(self.masks_directory + '/*'))
        image_path = image_path_ls[idx]
        mask_path = mask_path_ls[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        mask = (mask // 255).astype(np.uint8)
        sample = dict(image=image, mask=mask)
        return sample
    
    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _check_unique(self):
        image_path_ls = sorted(glob.glob(self.images_directory + '/*'))
        img_filenames = [path.split('/')[-1][:-len('.jpg')] for path in image_path_ls]
        mask_path_ls = sorted(glob.glob(self.masks_directory + '/*'))
        mask_filenames = [path.split('/')[-1][:-len('.jpg')] for path in mask_path_ls]
        for img_name,mask_name in zip(img_filenames,mask_filenames):
            if img_name != mask_name:
                return False
        return True

    def _read_name(self):
        image_path_ls = sorted(glob.glob(self.images_directory + '/*'))
        filenames = [path.split('/')[-1][:-len('.jpg')] for path in image_path_ls]
        return filenames