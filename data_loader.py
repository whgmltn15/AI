import cv2
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch

class CustomDataSet(Dataset):

    # 데이터 초기화 시키는 작업
    def __init__(self, file_path):

        self.to_tensor = transforms.ToTensor()
        f = open("dot_cat_label.txt", 'r')
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
        resize_img = cv2.resize(img, (256, 256))
        img = self.to_tensor(resize_img)


        label = float(self.label_list[index])

        return img, label

    # 데이터의 전체길이 구함
    def __len__(self):
        return self.data_len

def load_data():

    path = "dot_cat_label.txt"
    custom_dataset = CustomDataSet(path)
    train_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=10, shuffle=False)

    return train_loader

#
# for img, label in train_loader:
#     print(img.shape, label)