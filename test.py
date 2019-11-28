import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from unet import Unet
from dataset import CardDataset
import yaml
import cv2
from PIL import Image
import glob
import time
import numpy as np

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()

Model_path = "checkpoints/weights_9.pth"

names = glob.glob("data/val/im*.jpg")

def cv2img_process(img):
    ####分类图片预处理，
    #img = one_img(img)#add pad white resize 512
    assert (img.shape[0]==512 and img.shape[1]==512),"img not resize 512"
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    img = np.float32(img)
    img = np.ascontiguousarray(img[..., ::-1])
    img = img.transpose(2, 0, 1)# Convert Img from BGR to RGB
    for channel, _ in enumerate(img):
        # Normalization 
        img[channel] /= 255
        img[channel] -= mean[channel]
        img[channel] /= std[channel]
    # img = (img/255 - mean) / std 
    # Convert to float tensor
    img = torch.from_numpy(img).float().unsqueeze(0)  #chw  to bchw,加一个维度，并将numpy 变成torch
    #print (type(img))
    #print (img.shape)
    # Convert to Pytorch variable
    # img = Variable(img, requires_grad=false)
    return img


#显示模型的输出结果
def test():
    model = Unet(3, 1)
    model.load_state_dict(torch.load(Model_path,map_location='cpu'))
    #card_dataset = CardDataset("data/val", transform=x_transforms,target_transform=y_transforms)
    #dataloaders = DataLoader(card_dataset, batch_size=1)
    model.eval()
    
    with torch.no_grad():
        for name in names:
            img = cv2.imread(name,1)
            x = cv2img_process(img)
            
            y=model(x)
            img_y = (torch.squeeze(y).numpy()*-0.4*40/255.0-0.3)/0.7
            img_y = np.where(img_y < 0.3,0,img_y)
            img_y = np.where(img_y > 0.3,1,img_y)

            cv2.imshow("x",img)
            cv2.imshow("predict",img_y)
            #print(img.shape)
            #print(img_y.shape)
            #print("max ",img_y.max())
            #print("min ",img_y.min())
            print(img_y[250][250])
            cv2.waitKey(10)


#显示模型的输出结果
def test_video():
    cap = cv2.VideoCapture(2)
    
    
    model = Unet(3, 1)
    model.load_state_dict(torch.load(Model_path))

    model.to(device)
    #card_dataset = CardDataset("data/val", transform=x_transforms,target_transform=y_transforms)
    #dataloaders = DataLoader(card_dataset, batch_size=1)
    model.eval()
    
    with torch.no_grad():
        while True:
            ret,frame = cap.read()
            if ret is None:
                print("camera is not ready")
                exit(0)
                

            frame = frame[0:480,0:480]
            img = cv2.resize(frame,(512,512))
            #img = cv2.imread(name,1)
            
            x = cv2img_process(img)

            x_cuda = x.cuda()

            y=model(x_cuda)
            y_cpu = y.cpu()
            img_y = (torch.squeeze(y_cpu).numpy()*-0.4*40/255.0-0.3)/0.7
            img_y = np.where(img_y < 0.3,0,img_y)
            img_y = np.where(img_y > 0.3,1,img_y)

            cv2.imshow("x",img)
            cv2.imshow("predict",img_y)
            #print(img.shape)
            #print(img_y.shape)
            #print("max ",img_y.max())
            #print("min ",img_y.min())
            #print(img_y[250][250])
            cv2.waitKey(1)


if __name__ == '__main__':
    # 用摄像头
    #test_video()
    # 用测试图片
    test()
