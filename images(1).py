import cv2
import torch
from torch.utils.data.dataset import Dataset
import torch
import glob

class CustomDataSet(Dataset):
    # 데이터 초기화 시키는 작업
    def __init__(self, file_path):
        f = open("./images.txt", 'r')
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
        print(self.img_path[index])
        print(img.shape)
        resize_img = cv2.resize(img, (244, 244))

        label = self.label_list[index]

        return resize_img, label

    # 데이터의 전체길이 구함
    def __len__(self):
        return self.data_len

path = "/images.txt"
custom_dataset = CustomDataSet(path)
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=1, shuffle=False)

for img, label in train_loader:
    print(img.shape, label)

find_jpg = glob.glob("/imgData/cat.jpg")
print(find_jpg)

f = open("label.txt", 'w')

for img_name in find_jpg:
    if "cat" in img_name:
        f.write(img_name + ", 0 \n")
    else:
        f.write(img_name + ", 1 \n")