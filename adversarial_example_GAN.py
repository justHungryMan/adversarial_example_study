#!/usr/bin/env python
# coding: utf-8

# In[73]:


import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import argparse
import glob
import os
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt


# In[64]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 64
num_workers = 2
epochs = 256
lr = 0.0002
betas = (0.5, 0.999)
PATH = '../data/saved_model/GAN.pth'


# In[65]:


def get_indices(dataset,class_name):
    indices =  []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == class_name:
            indices.append(i)
    return indices


transform = transforms.Compose([
    transforms.Scale(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root = './data', train=True,
                                       download=True, transform=transform)
cat_idx = get_indices(trainset, 3)
cat_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(cat_idx))

dog_idx = get_indices(trainset, 5)
dog_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(cat_idx))


#classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
#          'horse', 'ship', 'truck')

def imshow(img):
    npimg = img.numpy()
    plt.imshow(npimg)
    plt.show


# In[66]:


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            # 3 x 64 x 64)
            nn.Conv2d(3, 64, 4,2,1,bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64*2, 64*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        output = self.conv(input)
        return output.view(-1, 1).squeeze(1)
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.tconv = nn.Sequential(
            nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        output = self.tconv(input)
        return output


# In[67]:


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: 
            torch.nn.init.xavier_uniform_(m.bias)


# In[68]:


cat_G = Generator()
cat_G.apply(weight_init)

cat_D = Discriminator()
cat_D.apply(weight_init)

if torch.cuda.is_available():
    cat_G = cat_G.cuda()
    cat_D = cat_D.cuda()

criterion = nn.BCELoss()

input = torch.FloatTensor(batch_size, 3, 64, 64)
noise = torch.FloatTensor(batch_size, 100, 1, 1)
fixed_noise = torch.FloatTensor(batch_size, 100, 1, 1).normal_(0, 1)
fixed_noise = Variable(fixed_noise)


optimizerD_cat = optim.Adam(cat_D.parameters(), lr=lr, betas=betas)
optimizerG_cat = optim.Adam(cat_G.parameters(), lr=lr, betas=betas)

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


# In[71]:


for epoch in tqdm(range(epochs)):
    for i, (images, labels) in enumerate(cat_loader):
        
        noise = to_variable(torch.FloatTensor(images.shape[0], 100, 1, 1).normal_(0, 1))
        
        true_label = to_variable(torch.ones(images.shape[0]))
        false_label = to_variable(torch.zeros(images.shape[0]))
        
        true_images = to_variable(images)
        fake_images = cat_G(noise)
        
        # D
        optimizerD_cat.zero_grad()
        D_loss = criterion(cat_D(true_images), true_label) + criterion(cat_D(fake_images), false_label)
        D_loss.backward(retain_graph=True)
        optimizerD_cat.step()
        
        # G
        optimizerG_cat.zero_grad()
        G_loss = criterion(cat_D(fake_images), true_label)
        G_loss.backward(retain_graph=True)
        optimizerG_cat.step()
        
print('Done')


# In[119]:


def denorm(x):
    
    return ((x + 1) / 2).clamp(0, 1)

torch.save({
    'epoch': epoch,
    'cat_G': cat_G.state_dict(),
    'optimizerG_cat': optimizerG_cat.state_dict(),
    'cat_D': cat_D.state_dict(),
    'optimizerD_cat': optimizerD_cat.state_dict(),
}, './data/model_{epoch}'.format(epoch=epochs))

for i in range(100):
    noise = to_variable(torch.FloatTensor(1, 100, 1, 1).normal_(0, 1))
    #cat = denorm(cat_G(noise)).transpose(1, 2).transpose(2, 3)
    #generated_image = cat.view(64, 64, 3).cpu().detach()
    #imshow(generated_image)

    torchvision.utils.save_image(denorm(cat_G(noise).cpu().detach()), './data/cat_{result}'.format(result=i))


# In[ ]:




