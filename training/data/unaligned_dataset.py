import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import nibabel as nib
import numpy as np
import re


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def _extract_center_id(self, filepath):
        """Extract center ID from filename.

        Parses filenames with pattern 'AB_[numbers][A|B|C][numbers]-*.nii'
        where the last letter (A, B, or C) before the final number determines the center.
        Examples: 'AB_1ABA005-1.nii' -> A (center 0), 'AB_1ABC122-48.nii' -> C (center 2)

        Parameters:
            filepath (str): Path to the image file

        Returns:
            int or None: Center ID (0 for A, 1 for B, 2 for C), or None if pattern not found
        """
        filename = os.path.basename(filepath)
        # Check if filename starts with 'AB_' and matches the expected pattern
        # Important: downstream applications cannot handle other body regions
        # because of varying number of treatment centers and names!
        if filename.startswith('AB_'):
            # Match pattern: AB_[numbers][A|B|C][numbers]-[numbers].nii
            match = re.search(r'AB_\dAB*([ABC])\d+-\d+\.nii', filename)
            if match:
                center_letter = match.group(1)
                # Convert A->0, B->1, C->2
                return ord(center_letter) - ord('A')
        return None

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        #  TBD? Crop on patches (there is default function, check it)
        if A_path.endswith(".nii") and B_path.endswith(".nii"):
            # print("TRaining on NII images")
            A_img_nifti = nib.load(A_path)
            A_img_numpy = A_img_nifti.get_fdata(caching = "unchanged")
            # tbd debug sizing and numpy array acceptance
            A_img_numpy=A_img_numpy*255
            A_img_numpy = np.squeeze(A_img_numpy)
            A_img_numpy = A_img_numpy.astype(np.uint8)
            A_img = Image.fromarray(A_img_numpy)

            B_img_nifti = nib.load(B_path)
            B_img_numpy = B_img_nifti.get_fdata(caching = "unchanged")
            # tbd debug sizing and numpy array acceptance
            B_img_numpy = np.squeeze(B_img_numpy)
            B_img_numpy = B_img_numpy * 255
            B_img_numpy = B_img_numpy.astype(np.uint8)
            B_img = Image.fromarray(B_img_numpy)

        # to modifY
        else:
            # print("Training NOT on nii images")
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')


        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        result = {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

        # Extract center IDs independently for each domain
        center_id_A = self._extract_center_id(A_path)
        center_id_B = self._extract_center_id(B_path)

        # Only add center_ids if they are not None
        if center_id_A is not None:
            result['center_id_A'] = center_id_A
        if center_id_B is not None:
            result['center_id_B'] = center_id_B

        return result

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)