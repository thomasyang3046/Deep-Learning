from fileinput import filename
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import json

# In[]
def custom_transform(mode):
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
    # Transformer
    transformer = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        normalize
        ])
    return transformer
# In[]
#test 先讀test的label(轉成整數)並生成image，之後把image和label(one hot)丟到resnet跑
def getData(mode, root, test_file='test.json'):
    label_map = json.load(open(os.path.join(root, 'objects.json')))
    if mode =='test':
        data_json = json.load(open(os.path.join(root, test_file)))
        labels_list = data_json
        one_hot_vector_list = []
        for i in range(len(labels_list)):
            one_hot_vector = np.zeros(24, dtype=int)
            for j in labels_list[i]:
                one_hot_vector[label_map[j]] = 1
            one_hot_vector_list.append(one_hot_vector)
        labels = np.array(one_hot_vector_list)
    else:
        data_json = json.load(open(os.path.join(root, 'new_test.json')))
        labels_list = data_json
        one_hot_vector_list = []
        for i in range(len(labels_list)):
            one_hot_vector = np.zeros(24, dtype=int)
            for j in labels_list[i]:
                one_hot_vector[label_map[j]] = 1
            one_hot_vector_list.append(one_hot_vector)
        labels = np.array(one_hot_vector_list)

    return labels
# In[]
class testDataset(Dataset):
    def __init__(self, mode, root, test_file='test.json'):
        self.filenames = None
        self.labels = None
        self.mode = mode
        self.root = root
        self.labels = getData(mode, root, test_file=test_file)
        self.transform = custom_transform(mode)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.mode == 'test':
            label = torch.Tensor(self.labels[idx])
            return label
        else:
            label = torch.Tensor(self.labels[idx])
            return label



