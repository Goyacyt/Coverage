from operator import ne
from pyexpat import model
from tkinter import Y
from turtle import forward
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets,transforms
import matplotlib.pyplot as plt

import torch.optim as optim
from torch.autograd import Variable

from torch.nn import Module
from torch import nn

from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import visualization as viz
import cv2

import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'

RANDOM_SEED=42
LEARNING_RATE=0.001
BATCH_SIZE=32
NUM_EPOCHS=15
IMG_SIZE=32
NUM_CLASSES=10
activate_range=0.2

classes=['0','1','2','3','4','5','6','7','8','9']

transforms=transforms.Compose([transforms.ToTensor()
])

train_dataset=datasets.MNIST(root='mnist_data',
train=True,
transform=transforms,
download=True)

test_dataset=datasets.MNIST(root='mnist_data',
train=False,
transform=transforms)


train_loader=DataLoader(dataset=train_dataset,
batch_size=BATCH_SIZE,
shuffle=True)

test_loader=DataLoader(dataset=test_dataset,
batch_size=BATCH_SIZE,
shuffle=False)

def traverse(tensor,activate_gate):
    res=0
    active=0
    sub=tensor.reshape(-1)
    for i in sub:
        res+=1
        element=i.item()
        #print(element)
        if element>=activate_gate:
            active+=1
    return (res,active)

class LeNet1(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet1, self).__init__()
        
        # input is Nx1x28x28
        model_list = [
            # params: 4*(5*5*1 + 1) = 104
            # output is (28 - 5) + 1 = 24 => Nx4x24x24
            nn.Conv2d(1, 4, 5),
            nn.Tanh(),
            # output is 24/2 = 12 => Nx4x12x12
            nn.AvgPool2d(2),
            # params: (5*5*4 + 1) * 12 = 1212
            # output: 12 - 5 + 1 = 8 => Nx12x8x8
            nn.Conv2d(4, 12, 5),
            nn.Tanh(),
            # output: 8/2 = 4 => Nx12x4x4
            nn.AvgPool2d(2)
        ]
        
        self.model = nn.Sequential(*model_list)
        # params: (12*4*4 + 1) * 10 = 1930
        self.fc = nn.Linear(12*4*4, num_classes)
        # Total number of parameters = 104 + 1212 + 1930 = 3246
    
    def forward(self, x):
        out = self.model(x)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out

net=LeNet1()
optimizer=optim.SGD(net.parameters(),lr=0.1)
loss=nn.CrossEntropyLoss()

USE_PRETRAINEDMODEL=True
if USE_PRETRAINEDMODEL:
    print("use pretrained model")
    net.load_state_dict(torch.load('./mnist_torchvision_Lenet1.pt'))
else:
    for epoch in range(NUM_EPOCHS):
        net.train()
        for id,(train_x,train_label) in enumerate(train_loader):
            optimizer.zero_grad()
            #print("train_label:",train_label)
            predict_y=net(train_x.float())
            #print("predict:",predict_y.shape)
            #print("predict_value:",predict_y)
            va_loss=loss(predict_y,train_label.long())
            if id%100==0:
                print('id:{},loss:{}'.format(id,va_loss.sum().item()))
            va_loss.backward()
            optimizer.step()

    print("Finish Training!")
    torch.save(net.state_dict(),'./mnist_torchvision_Lenet1.pt')
    

def imshow(img,transpose=True):
    npimg=img.numpy()
    print(type(npimg))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

def attributes_image_features(algorithm,input,**kwargs):
    net.zero_grad()
    tensor_attribution=algorithm.attribute(input,target=labels[ind],**kwargs)
    return tensor_attribution

def attributes_image_features2(algorithm,input,**kwargs):
    net.zero_grad()
    tensor_attribution=algorithm.attribute(input,target=test_label[ind],**kwargs)
    return tensor_attribution

net.eval()
saliency=Saliency(net)

num_image=0

"""for dataiter in iter(test_loader):
    images,labels=dataiter
    outputs=net(images)
    _,predicted=torch.max(outputs,1)
    for ind in range(32):
        input=images[ind].unsqueeze(0)
        input.requires_grad=True
        grads=saliency.attribute(input,target=labels[ind].item())
        grads = np.transpose(grads.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        dl = DeepLift(net)
        attr_dl = attributes_image_features(dl, input, baselines=input * 0)
        attr_dl = attr_dl.squeeze(0).cpu().detach().numpy()
        th,closing,opening=utils.attr2concept(attr_dl)
        #blank=np.zeros([28,28],dtype=np.uint8)
        img_h=cv2.hconcat([th,closing,opening])
        #img_h=cv2.hconcat([th,closing,opening])
        plt.imshow(img_h)
        plt.title("True label:"+str(classes[labels[ind]])+"   Predict label"+str(classes[predicted[ind]]))
        plt.savefig("./mnist_Lenet1/img"+str(num_image)+"_"+str(ind)+"_"+str(classes[labels[ind]])+".jpg")
    num_image+=1"""
num_image=0

for id,(test_image,test_label) in enumerate(test_loader):
    for k in range(32):
        ing=test_image[k,:].reshape(28,28)
        img_scale=cv2.resize(ing.numpy(),(14,14),interpolation=cv2.INTER_AREA)
        img_scale=cv2.copyMakeBorder(img_scale,7,7,7,7,cv2.BORDER_CONSTANT,0)
        test_image[k,:]=torch.from_numpy(img_scale)
    outputs=net(test_image)
    _,predicted=torch.max(outputs,1)
    for ind in range(32):
        input=test_image[ind].unsqueeze(0)
        input.requires_grad=True
        grads=saliency.attribute(input,target=test_label[ind].item())
        grads=np.transpose(grads.squeeze(0).cpu().detach().numpy(),(1,2,0))
        dl=DeepLift(net)
        attr_dl=attributes_image_features2(dl,input,baselines=input*0)
        attr_dl=attr_dl.squeeze(0).cpu().detach().numpy()
        th,closing,opening=utils.attr2concept(attr_dl)
        img_h=cv2.hconcat([th,opening,closing])
        plt.imshow(img_h)
        plt.title("After Scale\n True label:"+str(classes[test_label[ind]])+"   Predict label"+str(classes[predicted[ind]]))
        plt.savefig("./mnist_Lenet1/mutate_img"+str(num_image)+"_"+str(ind)+"_"+str(classes[test_label[ind]])+".jpg")
    num_image+=1
    if num_image>15:
        break

