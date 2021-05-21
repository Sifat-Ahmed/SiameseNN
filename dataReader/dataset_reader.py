import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

class CreateDataset(Dataset):
    def __init__(self,
                 images,
                 labels,
                 transform=None,
                 train=True):

        self._images = images
        self._labels = labels
        self._transform = transform
        self._train = train

        #print("Classes" , self._classes)

    def __len__(self):
        return len(self._images)


    def __getitem__(self, idx):


        img1 = cv2.imread(self._images[idx][0])
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        #img1 = img1/255.0


        img2 = cv2.imread(self._images[idx][1])
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        #img2 = img2/255.0

        if self._transform:
            img1 = self._transform(img1)
            img2 = self._transform(img2)

        label = self._labels[idx]

        return img1, img2, torch.tensor(label, dtype=torch.float32)



if __name__ == "__main__":
    dataset = CreateDataset(r'H:/Research/JTEKT/Human reid/json_helper/dataset.json')

    for i in dataset:
        print(i)