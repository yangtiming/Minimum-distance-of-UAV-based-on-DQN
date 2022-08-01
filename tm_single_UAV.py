# coding = utf-8

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import time
import math
import cv2
import matplotlib.pyplot as plt
import wandb

# parameters
Batch_size = 128
Lr = 0.0002
Epsilon = 0.9  # greedy policy
Gamma = 0.9  # reward discount
Target_replace_iter = 100  # target update frequency
Memory_capacity = 20000

num_UAV = 1
num_DEV = 50
location_UAV = 2  # xy
Height = 50
Speed = 3000
location_DEV = 2  # xy
N_actions = 4 * num_UAV  # xy 无人机左右前后 ; 设备状态:前后左右
N_states = location_UAV * num_UAV + location_DEV * num_DEV  # 无人机，设备所有的状态集合
area_size = 400 * 400  # 面积


def step(states, num_UAV, actions, num_left_right):
    stt = False
    states__ = list(np.zeros_like(states))
    for i in range(len(states)):
        states__[i] = states[i]

        # print(actions)
    if actions == 0:
        states__[0] = states__[0] + 3
        states__[1] = states__[1] + 3

    elif actions == 1:
        states__[0] = states__[0] - 3
        states__[1] = states__[1] - 3


    elif actions == 2:
        states__[0] = states__[0] - 3
        states__[1] = states__[1] + 3

    elif actions == 3:
        states__[0] = states__[0] + 3
        states__[1] = states__[1] - 3

    # device 左右

    for i in range(num_UAV * 2, num_UAV * 2 + 2 * num_left_right, 2):
        states__[i] = states__[i] + random.randint(0, 2)
        if states__[i] > 400:
            states__[i] = 1

    # device 上下
    for i in range(num_UAV * 2 + 2 * num_left_right, len(states__) - 1, 2):
        states__[i + 1] = states__[i + 1] - random.randint(0, 1)

        if states__[i + 1] < 0:
            states__[i + 1] = 399

    for i in range(num_UAV * 2):
        if states__[i] < 0:
            states__[i] = 0
        if states__[i] > 400:
            states__[i] = 400

    return states__


def reward(states, num_DEV, Speed, Height):
    uav = []
    uav_x = states[0]
    uav_y = states[1]
    r = []
    del states[0]
    del states[0]
    for i in range(num_DEV):
        uav.append(uav_x)
        uav.append(uav_y)

    for i in range(num_DEV * 2):
        temp = 0
        temp = math.sqrt((uav[i] - states[i]) ** 2)
        r.append(temp)

    states.insert(0, uav_x)
    states.insert(1, uav_y)

    return -sum(r) / Speed


def xuxian(plot, thickness, lineType):
    point_color = (150, 100, 0)  # BGR
    for i in range(0, 10, 2):
        ptStart = (0 + 40 * i, 200)
        ptEnd = (50 + 40 * i, 200)
        cv2.line(plot, ptStart, ptEnd, point_color, thickness, lineType)


def xuxian1(plot, thickness, lineType):
    point_color = (0, 50, 100)  # BGR
    for i in range(0, 10, 2):
        ptStart = (200, 0 + 40 * i)
        ptEnd = (200, 50 + 40 * i)
        cv2.line(plot, ptStart, ptEnd, point_color, thickness, lineType)


def plot(states, num_UAV, num_left_right):
    # print(states)

    plot = cv2.imread('1.png')
    plot = cv2.resize(plot, (400, 400), interpolation=cv2.INTER_AREA)
    # 画直线
    point_color = (150, 100, 0)  # BGR
    thickness = 2
    lineType = 4
    ptStart = (0, 165)
    ptEnd = (400, 165)
    cv2.line(plot, ptStart, ptEnd, point_color, thickness, lineType)

    ptStart = (0, 235)
    ptEnd = (400, 235)
    cv2.line(plot, ptStart, ptEnd, point_color, thickness, lineType)

    # 画虚线
    xuxian(plot, thickness, lineType)

    point_color = (0, 50, 100)  # BGR
    ptStart = (165, 0)
    ptEnd = (165, 400)
    cv2.line(plot, ptStart, ptEnd, point_color, thickness, lineType)

    ptStart = (235, 0)
    ptEnd = (235, 400)
    cv2.line(plot, ptStart, ptEnd, point_color, thickness, lineType)
    xuxian1(plot, thickness, lineType)

    for i in range(num_UAV):
        cv2.circle(plot, (states[i], states[i + 1]), 10, (255, 0, 0), 2)
    for i in range(num_UAV + 1, num_UAV * 2 + 2 * num_left_right, 2):
        cv2.circle(plot, (states[i], states[i + 1]), 10, (0, 0, 255), 2)
    for i in range(num_UAV * 2 + 2 * num_left_right, len(states) - 1, 2):
        cv2.circle(plot, (states[i], states[i + 1]), 10, (255, 0, 255), 2)

    cv2.imshow('img', plot)
    cv2.waitKey(1)
    return plot


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_states, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, N_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((Memory_capacity, N_states * 2 + 2))  # innitialize memory
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=Lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        if np.random.uniform() < Epsilon:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.numpy()
            # action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:
            action = np.random.randint(0, N_actions)
            # action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % Memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net update
        if self.learn_step_counter % Target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(Memory_capacity, Batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_states]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_states:N_states + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_states + 1:N_states + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_states:]))

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + Gamma * q_next.max(1)[0].view(Batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()
states = []
import random
random.seed(0)
# 初始化UAV
for i in range((num_UAV)):
    states.append(100)
    states.append(100)
num_left_right = 30
# 初始化DEV 左右走
for i in range(num_left_right):
    states.append(random.randint(0, 400))
    states.append(random.randint(175, 225))

# 初始化DEV 上下走
for i in range(num_DEV - num_left_right):
    states.append(random.randint(350, 400))
    states.append(random.randint(0, 400))

states_ = []
total_r = []
no_wandb = False
if no_wandb:
    wandb.init(project='dqn')
#save video
fps = 30
size = (400,400)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
videoWrite = cv2.VideoWriter('MySaveVideo.mp4', fourcc, fps, size)

for i_episode in range(100):
    # reset

    for i in range((num_UAV) * 2):
        states[i] = random.randint(0, 400)

    count_done = 0

    '''
    if i_episode <50:
        Epsilon = 0.1+(0.8*i_episode)/50

    else:
        Epsilon = 0.9
    '''
    print(i_episode)
    while True:
        #if i_episode>80:
        frame=plot(states, num_UAV, num_left_right)
        #if (count_done%10==0):
        #    videoWrite.write(frame)

        ep_r = 0
        actions = int(dqn.choose_action(states))

        # take action
        # print("states",states)
        states_ = step(states, num_UAV, actions, num_left_right)
        # print("states_",states_)

        # time.sleep(5)
        r = reward(states_, num_DEV, Speed, Height)
        dqn.store_transition(states, actions, r, states_)
        ep_r += r
        if dqn.memory_counter > Memory_capacity:
            dqn.learn()

        count_done = count_done + 1

        if count_done > 1000:
            total_r.append(ep_r)
            if no_wandb:
                wandb.log({'i_episode': i_episode, 'reward': ep_r})
            break

        for i in range(len(states)):
            states[i] = states_[i]

plt.plot(total_r, linewidth=1, label='reward_total')
print(total_r)
plt.show()
plt.close()

