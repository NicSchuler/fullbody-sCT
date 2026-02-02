"""
Paired NIfTI Dataset for cyclegan folder format.

Loads paired MR/CT images from separate trainA/trainB folders,
matching files by filename.
"""

import os
import random
import numpy as np
import nibabel as nib
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class PairedNiftiDataset(Dataset):
    """Dataset for paired MR-CT images in cyclegan folder format.

    Expected structure:
        dataroot/
        ├── train/
        │   ├── trainA/  (MR images)
        │   └── trainB/  (CT images)
        ├── val/
        │   ├── valA/
        │   └── valB/
        └── test/
            ├── testA/
            └── testB/

    Files are matched by filename (e.g., trainA/img001.nii <-> trainB/img001.nii)
    """

    def __init__(self, opt):
        """Initialize dataset.

        Args:
            opt: Options object with:
                - dataroot: Path to dataset root
                - phase: 'train', 'val', or 'test'
                - load_size: Size to load images (default 256)
                - crop_size: Size to crop (default 256)
                - no_flip: If True, disable random horizontal flip
                - max_dataset_size: Maximum number of samples
        """
        self.opt = opt
        self.root = opt.dataroot

        # Set up paths based on phase
        phase = getattr(opt, 'phase', 'train')
        if phase == 'train':
            self.dir_A = os.path.join(opt.dataroot, 'train', 'trainA')
            self.dir_B = os.path.join(opt.dataroot, 'train', 'trainB')
        elif phase == 'val':
            self.dir_A = os.path.join(opt.dataroot, 'val', 'valA')
            self.dir_B = os.path.join(opt.dataroot, 'val', 'valB')
        else:  # test
            self.dir_A = os.path.join(opt.dataroot, 'test', 'testA')
            self.dir_B = os.path.join(opt.dataroot, 'test', 'testB')

        # Get list of files in A directory
        self.A_paths = sorted([
            f for f in os.listdir(self.dir_A)
            if f.endswith('.nii') or f.endswith('.nii.gz')
        ])

        # Verify B files exist for each A file
        self.valid_files = []
        for fname in self.A_paths:
            if os.path.exists(os.path.join(self.dir_B, fname)):
                self.valid_files.append(fname)

        # Limit dataset size if specified
        max_size = getattr(opt, 'max_dataset_size', float('inf'))
        if max_size < float('inf'):
            self.valid_files = self.valid_files[:int(max_size)]

        print(f'[PairedNiftiDataset] Found {len(self.valid_files)} paired images in {phase} split')

        # Image properties
        self.load_size = getattr(opt, 'load_size', 256)
        self.crop_size = getattr(opt, 'crop_size', 256)
        self.no_flip = getattr(opt, 'no_flip', False)

    def _load_nifti(self, path):
        """Load NIfTI file and convert to PIL Image.

        Args:
            path: Path to .nii or .nii.gz file

        Returns:
            PIL.Image: Grayscale image
        """
        nifti = nib.load(path)
        data = nifti.get_fdata(caching='unchanged')
        data = np.squeeze(data)

        # Data is assumed to be normalized to [0, 1] from preprocessing
        # Convert to [0, 255] for PIL
        data = (data * 255).astype(np.uint8)

        return Image.fromarray(data)

    def _get_transform_params(self, size):
        """Get random transform parameters.

        Args:
            size: (width, height) tuple

        Returns:
            dict with 'crop_pos' and 'flip' keys
        """
        w, h = size

        # Random crop position
        x = random.randint(0, max(0, w - self.crop_size))
        y = random.randint(0, max(0, h - self.crop_size))

        # Random flip
        flip = not self.no_flip and random.random() > 0.5

        return {'crop_pos': (x, y), 'flip': flip}

    def _apply_transform(self, img, params):
        """Apply transforms to image.

        Args:
            img: PIL Image
            params: Transform parameters from _get_transform_params

        Returns:
            torch.Tensor: Transformed image in [-1, 1] range
        """
        # Resize if needed
        if img.size[0] != self.load_size or img.size[1] != self.load_size:
            img = img.resize((self.load_size, self.load_size), Image.BICUBIC)

        # Crop
        if self.load_size > self.crop_size:
            x, y = params['crop_pos']
            img = img.crop((x, y, x + self.crop_size, y + self.crop_size))

        # Flip
        if params['flip']:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Convert to tensor and normalize to [-1, 1]
        img = transforms.ToTensor()(img)
        img = transforms.Normalize((0.5,), (0.5,))(img)

        return img

    def __getitem__(self, index):
        """Get a paired sample.

        Args:
            index: Sample index

        Returns:
            dict with:
                - A: MR image tensor [1, H, W]
                - B: CT image tensor [1, H, W]
                - A_paths: Path to MR image
                - B_paths: Path to CT image
        """
        fname = self.valid_files[index]

        # Load images
        A_path = os.path.join(self.dir_A, fname)
        B_path = os.path.join(self.dir_B, fname)

        A_img = self._load_nifti(A_path)
        B_img = self._load_nifti(B_path)

        # Get shared transform parameters
        params = self._get_transform_params(A_img.size)

        # Apply same transforms to both
        A = self._apply_transform(A_img, params)
        B = self._apply_transform(B_img, params)

        return {
            'A': A,
            'B': B,
            'A_paths': A_path,
            'B_paths': B_path
        }

    def __len__(self):
        return len(self.valid_files)


def create_dataloader(opt):
    """Create DataLoader for PairedNiftiDataset.

    Args:
        opt: Options object

    Returns:
        torch.utils.data.DataLoader
    """
    dataset = PairedNiftiDataset(opt)

    shuffle = getattr(opt, 'phase', 'train') == 'train'
    num_workers = getattr(opt, 'num_threads', 4)
    batch_size = getattr(opt, 'batch_size', 1)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True
    )

    return dataloader
