#RESNET-----------------------

import wandb
WANDB_CONFIG_DIR = '/home/whale/Desktop/results-RA'
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

name = 'RESNET-RA-30-split-full'
PROJECT_NAME = name
cats = 30
n_actions = 30
input_size = 200

config = {
  'LR' : 0.0001,
  'NAME' : name,
  'num_episodes' : 50
}



def initialize_model(n_actions, use_pretrained=True):
    model_ft = models.resnet18(pretrained=use_pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, n_actions)
    return model_ft, input_size

def reward_fx(cat_i, action):
    r = 0
    if cat_i == action:
        r += 1
            # print("Got a treat!!")
    return r

def getSpikesInNextTimeslot(category):
    inputArray = np.empty((200,400))
    for j in range(8):
        for i in range(25):
            ModulationProbabilityFactor = [0, 0, 0, 0, 0, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 12.8, 6.4, 3.2, 1.6, 0.8, 0.4, 0.2, 0, 0, 0, 0, 0] #len = 25
            IntegerCollectionForInputStateGeneration = np.arange(10_000)
            category_size = category.shape[0]
            for c in range(category_size):
                currentInputSpikeProbability =  (category[c] * ModulationProbabilityFactor[i])
                if currentInputSpikeProbability > np.random.choice(IntegerCollectionForInputStateGeneration,1):
                    inputArray[i+(j*25),c] = 1.
                else:
                    inputArray[i+(j*25),c] = 0.
    return torch.from_numpy(inputArray)[None,...][None,...].float()

def defineObjectCategories():
    availableProbabilities = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 42, 43, 43, 43, 43, 44, 44, 44, 44, 45, 45, 45, 45, 46, 46, 46, 46, 47, 47, 47, 47, 48, 48, 48, 48, 49, 49, 49, 49, 50, 50, 50, 50, 51, 51, 51, 51, 52, 52, 52, 52, 53, 53, 53, 53, 54, 54, 54, 54, 55, 55, 55, 55, 56, 56, 56, 56, 57, 57, 57, 57, 58, 58, 58, 58, 59, 59, 59, 59, 60, 60, 60, 60, 61, 61, 61, 61, 62, 62, 62, 62, 63, 63, 63, 63, 64, 64, 64, 64, 65, 65, 65, 65, 66, 66, 66, 66, 67, 67, 67, 67, 68, 68, 68, 68, 69, 69, 69, 69, 70, 70, 70, 70, 71, 71, 71, 71, 72, 72, 72, 72, 73, 73, 73, 73, 74, 74, 74, 74, 75, 75, 75, 75, 76, 76, 76, 76, 77, 77, 77, 77, 78, 78, 78, 78, 79, 79, 79, 79, 80, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101, 102, 102, 102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106, 107, 107, 107, 108, 108, 108, 109, 109, 109, 110, 110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 116, 116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 126, 126, 127, 127, 128, 128, 129, 129, 130, 130, 131, 131, 132, 132, 133, 133, 134, 134, 135, 135, 136, 136, 137, 137, 138, 138, 139, 139, 140, 140, 141, 141, 142, 142, 143, 143, 144, 144, 145, 145, 146, 146, 147, 147, 148, 148, 149, 149, 150, 150, 151, 151, 152, 152, 153, 153, 154, 154, 155, 155, 156, 156, 157, 157, 158, 158, 159, 159, 160, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200]
    CategorySpikeProbabilities = np.empty((30,400))
    for j in range(30):
        for i in range(400):
            CategorySpikeProbabilities[j,i] = np.random.choice(availableProbabilities, 1)
    return CategorySpikeProbabilities



wandb.init(project=PROJECT_NAME, config=config, entity="justkittenaround")
wbconfig = wandb.config
print(wbconfig)

CategorySpikeProbabilities = defineObjectCategories()

model, input_size = initialize_model(n_actions, use_pretrained=False)
model = model.to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total # Trainable Parameters = ', pytorch_total_params,  '!!!!!!!')

optimizer = optim.Adam(model.parameters(), lr=wbconfig.LR)
criterion = nn.CrossEntropyLoss()
   
# TALL_REWARDS = []
# for epoch in range(wbconfig.num_episodes):
#     for phase in ['train']:
#         if phase == 'train':
#             model.train() 
#         rewards = 0
#         for cat_i in range(cats):
#             inputs = getSpikesInNextTimeslot(CategorySpikeProbabilities[cat_i,:])
#             inputs = torch.cat((inputs,inputs,inputs),1).to(device)
#             labels = torch.tensor(cat_i)[None,...].to(device)
#             optimizer.zero_grad()
#             with torch.set_grad_enabled(True):
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 _, preds = torch.max(outputs, 1)
#                 loss.backward()
#                 optimizer.step()
#             reward = reward_fx(cat_i, preds)
#             rewards += reward
#         wandb.log({phase+'-episode_reward' : rewards})
#         TALL_REWARDS.append(rewards)
#     wandb.log({phase+'total_reward':np.asarray(TALL_REWARDS).sum()/wbconfig.num_episodes})




# VALL_REWARDS = []
# for epoch in range(10):
#     for phase in ['val']:
#         model.eval() 
#         rewards = 0
#         for cat_i in range(cats):
#             inputs = getSpikesInNextTimeslot(CategorySpikeProbabilities[cat_i,:])
#             inputs = torch.cat((inputs,inputs,inputs),1).to(device)
#             labels = torch.tensor(cat_i)[None,...].to(device)
#             optimizer.zero_grad()
#             with torch.set_grad_enabled(False):
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 _, preds = torch.max(outputs, 1)
#             reward = reward_fx(cat_i, preds)
#             rewards += reward
#         wandb.log({phase+'-episode_reward' : rewards})
#         VALL_REWARDS.append(rewards)
#     wandb.log({phase+'total_reward':np.asarray(VALL_REWARDS).sum()/10})

# print(np.asarray(VALL_REWARDS).sum()/10)



print("BEGGINING FIRST 15 CATEGORY TRAINING")
TALL_REWARDS = []
for epoch in range(wbconfig.num_episodes):
    for phase in ['first-train']:
        model.train() 
        rewards = 0
        for cat_i in range(15):
            inputs = getSpikesInNextTimeslot(CategorySpikeProbabilities[cat_i,:])
            inputs = torch.cat((inputs,inputs,inputs),1).to(device)
            labels = torch.tensor(cat_i)[None,...].to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()
            reward = reward_fx(cat_i, preds)
            rewards += reward
        wandb.log({phase+'-episode_reward' : rewards})
        TALL_REWARDS.append(rewards)
    wandb.log({phase+'total_reward':np.asarray(TALL_REWARDS).sum()/wbconfig.num_episodes})


print("BEGGINING SECOND 15 CATEGORY TRAINING")
TALL_REWARDS = []
for epoch in range(wbconfig.num_episodes):
    for phase in ['second-train']:
        model.train() 
        rewards = 0
        for cat_i in range(15,30):
            inputs = getSpikesInNextTimeslot(CategorySpikeProbabilities[cat_i,:])
            inputs = torch.cat((inputs,inputs,inputs),1).to(device)
            labels = torch.tensor(cat_i)[None,...].to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()
            reward = reward_fx(cat_i, preds)
            rewards += reward
        wandb.log({phase+'-episode_reward' : rewards})
        TALL_REWARDS.append(rewards)
    wandb.log({phase+'total_reward':np.asarray(TALL_REWARDS).sum()/wbconfig.num_episodes})

print("Beggining FULL!!!")
TALL_REWARDS = []
for epoch in range(1):
    for phase in ['full-train']:
        model.train() 
        rewards = 0
        for cat_i in range(cats):
            inputs = getSpikesInNextTimeslot(CategorySpikeProbabilities[cat_i,:])
            inputs = torch.cat((inputs,inputs,inputs),1).to(device)
            labels = torch.tensor(cat_i)[None,...].to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()
            reward = reward_fx(cat_i, preds)
            rewards += reward
        wandb.log({phase+'-episode_reward' : rewards})
        TALL_REWARDS.append(rewards)
    wandb.log({phase+'total_reward':np.asarray(TALL_REWARDS).sum()/1})




print("Doing Test!")
VALL_REWARDS = []
for epoch in range(10):
    for phase in ['val']:
        model.eval() 
        rewards = 0
        for cat_i in range(cats):
            inputs = getSpikesInNextTimeslot(CategorySpikeProbabilities[cat_i,:])
            inputs = torch.cat((inputs,inputs,inputs),1).to(device)
            labels = torch.tensor(cat_i)[None,...].to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
            reward = reward_fx(cat_i, preds)
            rewards += reward
        wandb.log({phase+'-episode_reward' : rewards})
        VALL_REWARDS.append(rewards)
    wandb.log({phase+'total_reward':np.asarray(VALL_REWARDS).sum()/10})

print(np.asarray(VALL_REWARDS).sum()/10)    