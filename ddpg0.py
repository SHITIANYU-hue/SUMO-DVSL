#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 01:36:24 2018

@author: wuyuankai
"""

from __future__ import division
#from Priority_Replay import SumTree, Memory
# import tensorflow as tf
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import os
import sys

tools = '/usr/share/sumo/tools'
sys.path.append(tools)
import traci
from networks0 import rm_vsl_co

EP_MAX = 600
LR_A = 0.0002    # learning rate for actor
LR_C = 0.0005    # learning rate for critic
GAMMA = 0.9      # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 64
BATCH_SIZE = 32

RENDER = False

###############################  DDPG  ####################################




# Constants
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
LR_A = 0.001
LR_C = 0.002
TAU = 0.01
GAMMA = 0.9

class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(s_dim, 60)
        self.fc2 = nn.Linear(60, a_dim)

    def forward(self, s):
        neta = F.relu(self.fc1(s))
        a = torch.sigmoid(self.fc2(neta)) * 8
        return a

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Critic, self).__init__()
        self.fc1_s = nn.Linear(s_dim, 50)
        self.fc1_a = nn.Linear(a_dim, 50)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, s, a):
        h_s = F.relu(self.fc1_s(s))
        h_a = F.relu(self.fc1_a(a))
        netc = torch.cat([h_s, h_a], dim=1)
        return self.fc2(netc)


class VSL_DDPG_PR(object):
    def __init__(self, a_dim, s_dim):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0

        self.a_dim, self.s_dim = a_dim, s_dim

        self.actor = Actor(s_dim, a_dim)
        self.actor_target = Actor(s_dim, a_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(s_dim, a_dim)
        self.critic_target = Critic(s_dim, a_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_A)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_C)

    def choose_action(self, s):
        s = torch.FloatTensor(s).unsqueeze(0)
        return self.actor(s).detach().numpy().flatten()

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = torch.FloatTensor(self.memory[indices, :])
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        a_target = self.actor_target(bs_)
        q_ = self.critic_target(bs_, a_target).detach()

        q_target = br + GAMMA * q_

        q = self.critic(bs, ba)
        critic_loss = F.mse_loss(q, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        a_loss = -self.critic(bs, self.actor(bs)).mean()

        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        # Soft update for target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_((1 - TAU) * target_param.data + TAU * param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_((1 - TAU) * target_param.data + TAU * param.data)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def savemodel(self):
        save_dir = 'ddpg_networkss_withoutexplore2/'
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), f'{save_dir}ddpg_actor.pth')
        torch.save(self.critic.state_dict(), f'{save_dir}ddpg_critic.pth')

    def loadmodel(self):
        self.actor.load_state_dict(torch.load('ddpg_networkss_withoutexplore/ddpg_actor.pth'))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic.load_state_dict(torch.load('ddpg_networkss_withoutexplore/ddpg_critic.pth'))
        self.critic_target.load_state_dict(self.critic.state_dict())

        
def from_a_to_mlv(a):
    return 17.8816 + 2.2352*np.floor(a)


vsl_controller = VSL_DDPG_PR(s_dim = 13, a_dim = 5)
net = rm_vsl_co(visualization = False)
total_step = 0
var = 1.5
att = []
all_ep_r = []
att = []
all_co = []
all_hc = []
all_nox = []
all_pmx = []
all_oflow = []
all_bspeed = []
stime = np.zeros(13,)
co = 0
hc = 0
nox = 0
pmx = 0
oflow = 0
bspeed = 0
traveltime='meanTravelTime='
for ep in range(EP_MAX):
    time_start=time.time()
    co = 0
    hc = 0
    nox = 0
    pmx = 0
    ep_r = 0
    oflow = 0
    bspeed = 0
    v = 29.06*np.ones(5,)
    net.start_new_simulation(write_newtrips = False)
    s, r, simulationSteps, oflow_temp, bspeed_temp, co_temp, hc_temp, nox_temp, pmx_temp = net.run_step(v)
    co = co + co_temp
    hc = hc + hc_temp
    nox = nox + nox_temp
    pmx = pmx + pmx_temp
    oflow = oflow + oflow_temp
    bspeed_temp = bspeed + bspeed_temp
    stime[0:12] = s
    stime[12] = 0
    while simulationSteps < 18000:
        a = vsl_controller.choose_action(stime)
        #a = np.clip(np.random.laplace(a, var), 0, 7.99) The exploration is not very useful
        v = from_a_to_mlv(a)
        stime_ = np.zeros(13,)
        s_, r, simulationSteps, oflow_temp, bspeed_temp, co_temp, hc_temp, nox_temp, pmx_temp = net.run_step(v)

#        vid_list = traci.lane.getLastStepVehicleIDs('m7_5') + traci.lane.getLastStepVehicleIDs('m7_4') + traci.lane.getLastStepVehicleIDs('m6_4') + traci.lane.getLastStepVehicleIDs('m6_3')
#        for i in range(len(vid_list)):
#            traci.vehicle.setLaneChangeMode(vid_list[i], 0b001000000000)
        co = co + co_temp
        hc = hc + hc_temp
        nox = nox + nox_temp
        pmx = pmx + pmx_temp
        oflow = oflow + oflow_temp
        bspeed = bspeed + bspeed_temp
        stime_[0:12] = s_
        stime_[12] = simulationSteps/18000
        vsl_controller.store_transition(stime, a, r, stime_)
        total_step = total_step + 1
        if total_step > MEMORY_CAPACITY:
            #var = abs(1.5 - 1.5/600*ep)    # decay the action randomness
            vsl_controller.learn()
        stime = stime_
        ep_r += r
    all_ep_r.append(ep_r)
    all_co.append(co/1000)
    all_hc.append(hc/1000)
    all_nox.append(nox/1000)
    all_pmx.append(pmx/1000)
    all_oflow.append(oflow)
    all_bspeed.append(bspeed/300)
    net.close()
    fname = 'output_sumo.xml'
    with open(fname, 'r') as f:  # 打开文件
        lines = f.readlines()  # 读取所有行
        last_line = lines[-2]  # 取最后一行
    nPos=last_line.index(traveltime)
    aat_tempo = float(last_line[nPos+16:nPos+21])
    print('Episode:', ep, ' Rewards: %.4f' % ep_r, 'CO(g): %.4f' % co,\
          'HC(g): %.4f' % hc, 'NOX(g): %.4f' % nox, 'PMX(g): %.4f' % pmx, 'Out-in flow: %.4f' % oflow, \
          'Bottleneck speed: %.4f' % bspeed, 'Average travel time: %.4f' % aat_tempo)
    if all_ep_r[ep] == max(all_ep_r) and ep > 15:
        vsl_controller.savemodel()
    time_end=time.time()
    print('totally cost',time_end-time_start)
    
    
'''
Comparison with no VSL control
'''

#time_start=time.time()
#vsl_controller = VSL_DDPG_PR(s_dim = 13, a_dim = 5)
#net = rm_vsl_co(visualization = False, incidents = False)
##net.writenewtrips()
#traveltime='meanTravelTime='
#co = 0
#hc = 0
#nox = 0
#pmx = 0
#ep_r = 0
#oflow = 0
#bspeed = 0
#v = 29.06*np.ones(5,)
#net.start_new_simulation(write_newtrips = False)
#s, r, simulationSteps, oflow_temp, bspeed_temp, co_temp, hc_temp, nox_temp, pmx_temp = net.run_step(v)
#co = co + co_temp
#hc = hc + hc_temp
#nox = nox + nox_temp
#pmx = pmx + pmx_temp
#oflow = oflow + oflow_temp
#bspeed_temp = bspeed + bspeed_temp
#while simulationSteps < 18000:
#    s_, r, simulationSteps, oflow_temp, bspeed_temp, co_temp, hc_temp, nox_temp, pmx_temp = net.run_step(v)
#    co = co + co_temp
#    hc = hc + hc_temp
#    nox = nox + nox_temp
#    pmx = pmx + pmx_temp
#    oflow = oflow + oflow_temp
#    bspeed = bspeed + bspeed_temp
#    ep_r += r
#net.close()
#fname = 'output_sumo.xml'
#with open(fname, 'r') as f:  # 打开文件
#    lines = f.readlines()  # 读取所有行
#    last_line = lines[-2]  # 取最后一行
#nPos=last_line.index(traveltime)
#aat_tempo = float(last_line[nPos+16:nPos+21])
#print( 'Average Travel Time: %.4f' % aat_tempo, ' Rewards: %.4f' % ep_r, 'CO(g): %.4f' % co,\
#      'HC(g): %.4f' % hc, 'NOX(g): %.4f' % nox, 'PMX(g): %.4f' % pmx, 'Out-in flow: %.4f' % oflow, \
#      'Bottleneck speed: %.4f' % bspeed)
#time_end=time.time()
#print('totally cost',time_end-time_start)
#
#time_start=time.time()
#vsl_controller.loadmodel()
#co = 0
#hc = 0
#nox = 0
#pmx = 0
#ep_r = 0
#oflow = 0
#bspeed = 0
#v = 29.06*np.ones(5,)
#net.start_new_simulation(write_newtrips = False)
#s, r, simulationSteps, oflow_temp, bspeed_temp, co_temp, hc_temp, nox_temp, pmx_temp = net.run_step(v)
#co = co + co_temp
#hc = hc + hc_temp
#nox = nox + nox_temp
#pmx = pmx + pmx_temp
#oflow = oflow + oflow_temp
#bspeed_temp = bspeed + bspeed_temp
#stime = np.zeros(13,)
#stime[0:12] = s
#stime[12] = 0
#while simulationSteps < 18000:
#    a = vsl_controller.choose_action(stime)
#    #a = np.clip(np.random.laplace(a, var), 0, 7.99)
#    v = from_a_to_mlv(a)
#    stime_ = np.zeros(13,)
#    s_, r, simulationSteps, oflow_temp, bspeed_temp, co_temp, hc_temp, nox_temp, pmx_temp = net.run_step(v)
#    co = co + co_temp
#    hc = hc + hc_temp
#    nox = nox + nox_temp
#    pmx = pmx + pmx_temp
#    oflow = oflow + oflow_temp
#    bspeed = bspeed + bspeed_temp
#    stime_[0:12] = s_
#    stime_[12] = simulationSteps/18000
#    stime = stime_
#    ep_r += r
#net.close()
#fname = 'output_sumo.xml'
#with open(fname, 'r') as f:  # 打开文件
#    lines = f.readlines()  # 读取所有行
#    last_line = lines[-2]  # 取最后一行
#nPos=last_line.index(traveltime)
#aat_tempo = float(last_line[nPos+16:nPos+21])
#print( 'Average Travel Time: %.4f' % aat_tempo, ' Rewards: %.4f' % ep_r, 'CO(g): %.4f' % co,\
#      'HC(g): %.4f' % hc, 'NOX(g): %.4f' % nox, 'PMX(g): %.4f' % pmx, 'Out-in flow: %.4f' % oflow, \
#      'Bottleneck speed: %.4f' % bspeed)
#time_end=time.time()
#print('totally cost',time_end-time_start)