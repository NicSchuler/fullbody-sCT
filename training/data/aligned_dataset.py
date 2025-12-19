import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import nibabel as nib
import numpy as np
import re


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def _extract_center_id(self, filepath):
        """Extract center ID from filename.

        Parses filenames with pattern 'AB_[numbers][A|B|C][numbers]-*.nii'
        where the last letter (A, B, or C) before the final number determines the center.
        Examples: 'AB_1ABA005-1.nii' -> A (center 0), 'AB_1ABC122-48.nii' -> C (center 2)

        Parameters:
            filepath (str): Path to the image file

        Returns:
            int: Center ID (0 for A, 1 for B, 2 for C), defaults to 0 if pattern not found
        """
        filename = os.path.basename(filepath)
        # Match pattern: AB_[numbers][A|B|C][numbers]-[numbers].nii
        # We want to capture the letter (A, B, or C) that appears before the last set of digits before the hyphen
        match = re.search(r'([ABC])\d+-\d+\.nii', filename)
        if match:
            center_letter = match.group(1)
            # Convert A->0, B->1, C->2
            return ord(center_letter) - ord('A')
        return 0  # Default to center 0 (A) if pattern not found

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]

        if  AB_path.endswith(".nii"):
            # print("TRaining on NII images aligned")
            AB_img_nifti = nib.load(AB_path)
            AB_img_numpy = AB_img_nifti.get_fdata(caching = "unchanged")
            # tbd debug sizing and numpy array acceptance
            AB_img_numpy = np.squeeze(AB_img_numpy)
            AB_img_numpy = AB_img_numpy * 255
            AB_img_numpy = AB_img_numpy.astype(np.uint8)
            AB = Image.fromarray(AB_img_numpy)

        #      check PIL library
        else:
            # print("Training NOT on nii images, aligned dataset")
            AB = Image.open(AB_path).convert('RGB')

        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        # Extract center ID from filename
        center_id = self._extract_center_id(AB_path)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path, 'center_id': center_id}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)