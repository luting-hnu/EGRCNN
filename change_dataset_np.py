"# -- coding: UTF-8 --"
from torch.utils.data import Dataset
import numpy as np
import torchvision
import helper_augmentations
from PIL import Image
import glob

class ChangeDatasetNumpy(Dataset):
    """ChangeDataset Numpy Pickle Dataset"""

    def __init__(self, files, transform=None):

        image_path1 = glob.glob(files + '/A' + '/*.png')
        image_path1.sort()
        self.image_path1 = image_path1
        image_path2 = glob.glob(files + '/B' + '/*.png')
        image_path2.sort()
        self.image_path2 = image_path2
        target = glob.glob(files + '/label' + '/*.png')
        target.sort()
        self.target = target
        #label_edge
        target_edge = glob.glob(files + '/label_edge' + '/*.png')
        target_edge.sort()
        self.target_edge = target_edge
    
        self.transform = transform

    def __len__(self):
        # return len(self.data_dict)
        assert len(self.image_path1) == len(self.image_path2)
        return len(self.image_path1)

    def __getitem__(self, idx):
        images1 = Image.open(self.image_path1[idx])
        images2 = Image.open(self.image_path2[idx])
        mask = Image.open(self.target[idx])
        mask_edge = Image.open(self.target_edge[idx])
        sample = {'reference': images1, 'test': images2, 'label': mask, 'label_edge': mask_edge}
        # Handle Augmentations
        if self.transform:
            trf_reference = sample['reference']
            trf_test = sample['test']          
            trf_label = sample['label']
            trf_label_edge = sample['label_edge']
            # Dont do Normalize on label, all the other transformations apply...
            for t in self.transform.transforms:
                if (isinstance(t, helper_augmentations.SwapReferenceTest)) or (isinstance(t, helper_augmentations.JitterGamma)):
                    trf_reference, trf_test = t(sample)
                else:
                    # All other type of augmentations
                    trf_reference = t(trf_reference)
                    trf_test = t(trf_test)
                
                # Don't Normalize or Swap
                if not isinstance(t, torchvision.transforms.transforms.Normalize):
                    # ToTensor divide every result by 255
                    # https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#to_tensor
                    if isinstance(t, torchvision.transforms.transforms.ToTensor):
                        trf_label = t(trf_label) * 255.0
                        trf_label_edge = t(trf_label_edge) * 255.0
                    else:
                        if not isinstance(t, helper_augmentations.SwapReferenceTest):
                            if not isinstance(t, torchvision.transforms.transforms.ColorJitter):
                                if not isinstance(t, torchvision.transforms.transforms.RandomGrayscale):
                                    if not isinstance(t, helper_augmentations.JitterGamma):
                                        trf_label = t(trf_label)
                                        trf_label_edge = t(trf_label_edge)
                              
            sample = {'reference': trf_reference, 'test': trf_test, 'label': trf_label, 'label_edge': trf_label_edge}

        return sample