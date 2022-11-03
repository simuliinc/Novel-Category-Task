###https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
#DQN-----------------------

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


name = 'DQN-test-RA-30'
PROJECT_NAME = name
n_actions = 30
cats = 30


config = {
  'LR' : 0.0005,
  'BATCH_SIZE' : 64, 
  'GAMMA' : 0.3,
  'EPS_START' : 0.9,
  'EPS_END' : 0.01,
  'EPS_DECAY' : 100,
  'NAME' : name,
  "FC1" : 64,
  'FC2' : 64,
  'TARGET_UPDATE': 20,
  'num_episodes' : 900
}




##actual reward
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


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, wbconfig.FC1, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(wbconfig.FC1)
        self.conv2 = nn.Conv2d(wbconfig.FC1, wbconfig.FC2, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(wbconfig.FC2)
        self.conv3 = nn.Conv2d(wbconfig.FC2, wbconfig.FC2, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(wbconfig.FC2)
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * wbconfig.FC2
        self.head = nn.Linear(linear_input_size, outputs)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # print("OUT SHAPE", x.view(x.size(0), -1).shape)
        return self.head(x.view(x.size(0), -1))



def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = wbconfig.EPS_END + (wbconfig.EPS_START - wbconfig.EPS_END) * \
        math.exp(-1. * steps_done / wbconfig.EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < wbconfig.BATCH_SIZE:
        return
    transitions = memory.sample(wbconfig.BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(wbconfig.BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * wbconfig.GAMMA) + reward_batch
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated


wandb.init(project=PROJECT_NAME, config=config, entity="justkittenaround")
wbconfig = wandb.config
print(wbconfig)


CategorySpikeProbabilities = defineObjectCategories()


policy_net = DQN(200, 400, n_actions).to(device)
target_net = DQN(200, 400, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)
steps_done = 0
episode_durations = []


rewards = []
inp_shape = 400



pytorch_total_params = sum(p.numel() for p in policy_net.parameters() if p.requires_grad)
print('Total # Trainable Parameters = ', pytorch_total_params*2,  '!!!!!!!')



ALL_REWARDS = []
for i_episode in range(wbconfig.num_episodes):
    done = False
    rewards = 0
    currentTimeslot = 0
    cat_i = 0
    state  = getSpikesInNextTimeslot(CategorySpikeProbabilities[cat_i,:])
    for t in count():
        # Select and perform an action
        action = select_action(state)
        # print("SPIKES: ", state.sum())
        reward = reward_fx(cat_i, action)
        rewards += reward
        reward = torch.tensor([reward], device=device)
        cat_i += 1
        next_state = getSpikesInNextTimeslot(CategorySpikeProbabilities[cat_i,:])
        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        # Move to the next state
        state = next_state
        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if cat_i == 29:
        	done = True
        if done:
            episode_durations.append(t + 1)
            # plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % wbconfig.TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    wandb.log({'episode_reward':rewards})
    ALL_REWARDS.append(rewards)
wandb.log({'total_reward':np.asarray(ALL_REWARDS).sum()/wbconfig.num_episodes})
		


steps_done = 0

### TEST
print('TEST!!')
ALL_REWARDS = []
for i_episode in range(10):
    done = False
    rewards = 0
    currentTimeslot = 0
    cat_i = 0
    state  = getSpikesInNextTimeslot(CategorySpikeProbabilities[cat_i,:])
    for t in count():
        with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
            action = policy_net(state).max(1)[1].view(1, 1)
        # print("SPIKES: ", state.sum())
        reward = reward_fx(cat_i, action)
        rewards += reward
        reward = torch.tensor([reward], device=device)
        cat_i += 1
        next_state = getSpikesInNextTimeslot(CategorySpikeProbabilities[cat_i,:])
        # Move to the next state
        state = next_state
        if cat_i == 29:
            done = True
        if done:
            episode_durations.append(t + 1)
            # plot_durations()
            break
    wandb.log({'val-episode_reward':rewards})
    ALL_REWARDS.append(rewards)
wandb.log({'vall-total_reward':np.asarray(ALL_REWARDS).sum()/10})
        
print(np.asarray(ALL_REWARDS).sum()/10)








##############VESTIGUAL CODE#######################################
# def getSpikesInNextTimeslot(category, secondCategory, thirdCategory, currentTimeslot):
# 	phaseAtInitialTimeslot = 23
# 	ModulationProbabilityFactor = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 12.8, 6.4, 3.2, 1.6, 0.8, 0.4, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 	IntegerCollectionForInputStateGeneration = np.arange(10000)
# 	category_size = category.shape[0]
# 	currentTimeslot += 1
# 	currentPhase = phaseAtInitialTimeslot + (currentTimeslot -1)
# 	currentPhase -= 75*(currentPhase//75)
# 	if currentPhase == 0:
# 		currentPhase = 75
# 	secondPhase = currentPhase + 25
# 	if secondPhase > 75:
# 		secondPhase -= 75
# 	thirdPhase = secondPhase + 25
# 	if thirdPhase > 75:
# 		thirdphase -= 75
# 	inputArray = np.empty(400)
# 	for c in range(category_size):
# 		currentInputSpikeProbability =  (category[c] * ModulationProbabilityFactor[currentPhase]) + (secondCategory[c] * ModulationProbabilityFactor[secondPhase]) + (thirdCategory[c] * ModulationProbabilityFactor[thirdPhase])
# 		if currentInputSpikeProbability > np.random.choice(IntegerCollectionForInputStateGeneration,1):
# 			inputArray[c] = 1.
# 		else:
# 			inputArray[c] = 0.
# 	return inputArray


##chance reward
# def reward_fx(cat_i, action,i_exs, apc_dict, cat_id):
# 	i_ex = i_exs[cat_id[cat_i]]
# 	act_ex = apc_dict[action.item()]
# 	r=0
# 	for item in act_ex:
# 		ran = random.randint(0, 1)
# 		if ran == 1:
# 			r += .33
# 			# print("Got a treat!!")
# 	return r

## whith modulation
# def getSpikesInNextTimeslot(category, currentTimeslot):
# 	global currentTimeslot
# 	phaseAtInitialTimeslot = 23
# 	ModulationProbabilityFactor = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 12.8, 6.4, 3.2, 1.6, 0.8, 0.4, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #len = 75
# 	IntegerCollectionForInputStateGeneration = np.arange(10000)
# 	category_size = category.shape[0]
# 	inputArray = np.empty(400)
# 	currentTimeslot += 1
# 	currentPhase = phaseAtInitialTimeslot + (currentTimeslot-1)
# 	currentPhase -= 75*(currentPhase//75)
# 	if currentPhase == 0:
# 		currentPhase = 74
# 	for c in range(category_size):
# 		currentInputSpikeProbability =  (category[c] * ModulationProbabilityFactor[currentPhase-1])
# 		if currentInputSpikeProbability > np.random.choice(IntegerCollectionForInputStateGeneration,1):
# 			inputArray[c] = 1.
# 		else:
# 			inputArray[c] = 0.
# 	return inputArray

## w/1 category modulation
# def getSpikesInNextTimeslot(category, currentTimeslot):
# 	inputArray = np.empty((25,400))
# 	for i in range(25):
# 		phaseAtInitialTimeslot = 0
# 		ModulationProbabilityFactor = [0., 0., 0., 0., 0., 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 12.8, 6.4, 3.2, 1.6, 0.8, 0.4, 0.2, 0., 0., 0., 0., 0.]
# 		IntegerCollectionForInputStateGeneration = np.arange(10000)
# 		category_size = category.shape[0]
# 		# currentPhase = phaseAtInitialTimeslot + (currentTimeslot)
# 		# currentPhase -= 25*(currentPhase//25)
# 		# if currentPhase == 0:
# 		# 	currentPhase = 24
# 		currentTimeslot += 1
# 		for c in range(category_size):
# 			currentInputSpikeProbability =  (category[c] * ModulationProbabilityFactor[i])
# 			if currentInputSpikeProbability > np.random.choice(IntegerCollectionForInputStateGeneration,1):
# 				inputArray[i,c] = 1.
# 			else:
# 				inputArray[i,c] = 0.
# 	return inputArray



# def reward_fx(cat_i, action,i_exs, apc_dict, cat_id):
# 	i_ex = i_exs[cat_id[cat_i]]
# 	act_ex = apc_dict[action.item()]
# 	r=0
# 	for item in act_ex:
# 		if item in i_ex:
# 			r += .33
# 			# print("Got a treat!!")
# 	return r




# def reward_fx(action,i):
	# if i == 0:
	# 	correct_action = 0
	# if i == 3:
	# 	correct_action = 19
	# if correct_action == action:
	# 	reward = 1.0
	# if action == 1 and i==0:
	# 	action_ex = [0,1,3]
	# 	reward = .66
	# if action == 2 and i==0:
	# 	action_ex = [0,1,4]
	# 	reward = .66
	# if action == 3 and i==0:
	# 	action_ex = [0,1,5]
	# 	reward = .66
	# if action == 4 and i==0:
	# 	action_ex = [0,2,3]
	# 	reward = .66
	# if action == 5 and i==0:
	# 	action_ex = [0,2,4]
	# 	reward = .66
	# if action == 6 and i==0:
	# 	action_ex = [0,2,5]
	# 	reward = .66
	# if action == 7 and i==0:
	# 	action_ex = [0,3,4]
	# 	reward = .33
	# if action == 8 and i==0:
	# 	action_ex = [0,3,5]
	# 	reward = .33
	# if action == 9 and i==0:
	# 	action_ex = [0,4,5]
	# 	reward = .33
	# if action == 10 and i==0:
	# 	action_ex = [1,2,3]
	# 	reward = .66
	# if action == 11 and i==0:
	# 	action_ex = [1,2,4]
	# 	reward = .66
	# if action == 12 and i==0:
	# 	action_ex = [1,2,5]
	# 	reward = .66
	# if action == 13 and i==0:
	# 	action_ex = [1,3,4]
	# 	reward = .33
	# if action == 14 and i==0:
	# 	action_ex = [1,3,5]
	# 	reward = .33
	# if action == 15 and i==0:
	# 	action_ex = [1,4,5]
	# 	reward = .33
	# if action == 16 and i==0:
	# 	action_ex = [2,3,4]
	# 	reward = .33
	# if action == 17 and i==0:
	# 	action_ex = [2,3,5]
	# 	reward = .33
	# if action == 18 and i==0:
	# 	action_ex = [2,4,5]
	# 	reward = .33
	# if action == 19 and i==0:
	# 	reward = 0.
	# if action == 1 and i==3:
	# 	action_ex = [0,1,3]
	# 	reward = .33
	# if action == 2 and i==3:
	# 	action_ex = [0,1,4]
	# 	reward = .33
	# if action == 3 and i==3:
	# 	action_ex = [0,1,5]
	# 	reward = .33
	# if action == 4 and i==3:
	# 	action_ex = [0,2,3]
	# 	reward = .33
	# if action == 5 and i==3:
	# 	action_ex = [0,2,4]
	# 	reward = .33
	# if action == 6 and i==3:
	# 	action_ex = [0,2,5]
	# 	reward = .33
	# if action == 7 and i==3:
	# 	action_ex = [0,3,4]
	# 	reward = .66
	# if action == 8 and i==3:
	# 	action_ex = [0,3,5]
	# 	reward = .66
	# if action == 9 and i==3:
	# 	action_ex = [0,4,5]
	# 	reward = .66
	# if action == 10 and i==3:
	# 	action_ex = [1,2,3]
	# 	reward = .33
	# if action == 11 and i==3:
	# 	action_ex = [1,2,4]
	# 	reward = .33
	# if action == 12 and i==3:
	# 	action_ex = [1,2,5]
	# 	reward = .33
	# if action == 13 and i==3:
	# 	action_ex = [1,3,4]
	# 	reward = .66
	# if action == 14 and i==3:
	# 	action_ex = [1,3,5]
	# 	reward = .66
	# if action == 15 and i==3:
	# 	action_ex = [1,4,5]
	# 	reward = .66
	# if action == 16 and i==3:
	# 	action_ex = [2,3,4]
	# 	reward = .66
	# if action == 17 and i==3:
	# 	action_ex = [2,3,5]
	# 	reward = .66
	# if action == 18 and i==3:
	# 	action_ex = [2,4,5]
	# 	reward = .66
	# if action == 0 and i==3:
	# 	reward = 0.
	# return reward

# apc = []
# a = tuple(np.sort(np.random.choice(range(30),3,replace=False)))
# while len(apc) != 4060:
#      a = tuple(np.sort(np.random.choice(range(30),3,replace=False)))
#      while a not in apc:
#              apc.append(a)
#              a = tuple(np.sort(np.random.choice(range(30), 3, replace=False)))

# apc_dict = {}
# for v in range(4060):
# 	apc_dict.update({v:apc[v]})

# i_exs = []
# for i in np.arange(0,cats,3):
# 	i_exs.append(tuple([i, i+1, i+2]))

# cat_id = {}
# j = 0
# for cat_i in np.arange(0,cats,3):
# 	cat_id.update({cat_i: j})
# 	j += 1