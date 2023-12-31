import pandas as pd
from torch.utils import data
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
# In[]
def custom_transform(mode):
    if mode=="train":
        transformer=transforms.Compose([
            transforms.RandomRotation(degrees=45),
            transforms.Resize((512,512)),
            transforms.ToTensor(),#transform to [C, H, W]
            transforms.Normalize(mean=(0.3749, 0.2602, 0.1857), std=(0.2526, 0.1780, 0.1291))
            ])
    else:
        transformer=transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),#transform to [C, H, W]
            transforms.Normalize(mean=(0.3749, 0.2602, 0.1857), std=(0.2526, 0.1780, 0.1291))
            ])
    return transformer
def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        self.transform = custom_transform(mode=mode)
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        image_path=self.root+"/"+self.img_name[index]+".jpeg"
        img=Image.open(image_path)
        img=self.transform(img)
        label=self.label[index]
        return img, label


