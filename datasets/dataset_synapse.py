import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from PIL import Image

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y,_ = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            #image = np.transpose(image, (2, 0, 1))  # 调整维度顺序
            image = Image.fromarray(image.astype(np.uint8)).resize((self.output_size[1], self.output_size[0]), Image.BICUBIC) #双三次插值
            image = np.array(image, dtype=np.float32)
            #image = np.transpose(image, (1, 2, 0))  # 调整维度顺序
            label = Image.fromarray(label.astype(np.uint8)).resize((self.output_size[1], self.output_size[0]), Image.NEAREST) #最邻近插值：不改变像素值大小
            label = np.array(label, dtype=np.float32)

        image = torch.from_numpy(image)
        image=image.permute(2,0,1)
        label = torch.from_numpy(label)
        sample = {'image': image, 'label': label.long()}
        return sample


class AcdcDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.transform = transform  # using transform in torch!
        self.ids=sorted(os.listdir(images_dir))
        print(self.ids)
        self.ids2=sorted(os.listdir(masks_dir))
        print(self.ids2)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids2]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images_fps[idx]).convert('RGB'))
        mask = np.array(Image.open(self.masks_fps[idx]))
        sample = {'image':image , 'label': mask}
        id={'image_id':self.ids[idx],"label_id":self.ids2[idx]}
        sample = self.transform(sample)

        return sample,id
