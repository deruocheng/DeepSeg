from torch.utils.data.dataset import Dataset
import PIL.Image as Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import random


class segmentation_dataset(Dataset):
    def __init__(self, folder_path, folder_name, img_list, file_format, transform_options=None, model_phase=None, seed=0):
        self.data_path = folder_path
        self.folder_name = folder_name
        self.img_list = img_list
        self.data_ext = file_format
        self.phase = model_phase
        self.transform_list = transform_options
        self.random_state = np.random.RandomState(seed=seed)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        image_name_only = self.img_list[index].rstrip('\n')
        image_name = self.data_path / self.folder_name[0] / f"{image_name_only}{self.data_ext[0]}"
        image = Image.open(image_name).convert('RGB')
        img_transforms = transforms.ToTensor()
        # define basic transformations for the image
        if 'normalize' in self.transform_list:
            img_transforms = transforms.Compose([
                img_transforms,
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        # return data according to model phase
        if self.phase == 'predict':
            # return image, image_name
            image = img_transforms(image)
            return image, image_name
        else:
            # all other cases will require mask
            mask_name = self.data_path / self.folder_name[1] /f"{image_name_only}{self.data_ext[1]}"
            mask = Image.open(mask_name)
            if self.phase == 'test':
                # return image, mask, image_name
                image = img_transforms(image)
                mask = TF.to_tensor(mask)
                mask = mask.squeeze().long()
                return image, mask, image_name
            elif self.phase == 'val':
                # return image, mask
                image = img_transforms(image)
                mask = TF.to_tensor(mask)
                mask = mask.squeeze().long()
                return image, mask
            elif self.phase == 'train':
                # return simultaneously augmented image and mask
                if 'hflip' in self.transform_list and self.random_state.rand(1) > 0.5:
                    image = TF.hflip(image)
                    mask = TF.hflip(mask)
                if 'vflip' in self.transform_list and self.random_state.rand(1) > 0.5:
                    image = TF.vflip(image)
                    mask = TF.vflip(mask)
                if 'rotate' in self.transform_list and self.random_state.rand(1) > 0.5:
                    image = TF.rotate(image, 90)
                    mask = TF.rotate(mask, 90)
                image = img_transforms(image)
                mask = TF.to_tensor(mask)
                mask = mask.squeeze().long()
                return image, mask


def split_train_val(all_imgs, val_ratio, r_seed):
    random.seed(r_seed)
    val_num = int(len(all_imgs) * val_ratio)
    val_imgs = random.sample(all_imgs, val_num)
    train_imgs = [element for _, element in enumerate(all_imgs) if element not in val_imgs]
    return train_imgs, val_imgs