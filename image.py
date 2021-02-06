import cv2
import torch
from torch.utils.data.dataset import Dataset
import torch

class CustomDataSet(Dataset):
    # 데이터 초기화 시키는 작업
    def __init__(self, file_path):
        f = open(".\HS\images.txt", 'r')
        total_line = f.readlines()

        self.img_path = []
        self.label_list = []

        for line in total_line:
            # img part
            img_path = line.split(',')[0]
            self.img_path.append(img_path)

            # label part
            label = line.split(',')[1].split('\n')[0]
            self.label_list.append(label)
        self.data_len = len(self.label_list)

    # 경로를 통해 실제 데이터와 접근해서 데이터를 돌려줌
    def __getitem__(self, index):

        img = cv2.imread(self.img_path[index])
        resize_img = cv2.resize(img, (244, 244))

        label = self.label_list[index]

        return img, label

    # 데이터의 전체길이 구함
    def __len__(self):
        return self.data_len

path = "HS/images.txt"
custom_dataset = CustomDataSet(path)
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=1, shuffle=False)

for img, label in train_loader:
    print(img.shape, label)

'''
f = open("HS/images.txt", 'r')
total_line = f.readlines()
img_list = []
label_list = []

for line in total_line:
    # img part
    print(line)
    img_path = line.split(',')[0]
    img = cv2.imread(img_path)
    print(img.shape)
    resize_img = cv2.resize(img, (244, 244))
    print(resize_img.shape)
    img_list.append(resize_img)

    # label part
    label = line.split(',')[1].split('\n')[0]
    label_list.append(label)

img = cv2.imread("HS/a.jpg")
label = 1
'''