"""Single NIfTI dataset for inference with medical imaging files."""
import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import nibabel as nib
import numpy as np


class SingleNiftiDataset(BaseDataset):
    """Dataset class for single-domain NIfTI images (inference mode).

    This is similar to SingleDataset but supports NIfTI (.nii) files
    commonly used in medical imaging.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A (tensor) -- an image in one domain
            A_paths (str) -- the path of the image
        """
        A_path = self.A_paths[index]

        if A_path.endswith(".nii") or A_path.endswith(".nii.gz"):
            # Load NIfTI file using nibabel
            A_img_nifti = nib.load(A_path)
            A_img_numpy = A_img_nifti.get_fdata(caching="unchanged")

            # Convert normalized [0, 1] data to [0, 255] uint8 for PIL
            A_img_numpy = A_img_numpy * 255
            A_img_numpy = np.squeeze(A_img_numpy)
            A_img_numpy = A_img_numpy.astype(np.uint8)
            A_img = Image.fromarray(A_img_numpy)
        else:
            # Fallback to standard image loading
            A_img = Image.open(A_path).convert('RGB')

        A = self.transform(A_img)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
