#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#-*- coding: utf-8 -*-

import numpy as np
import paddle.fluid as fluid #导入PaddlePaddle的流式API。
import parl                  #导入PARL库，这是一个强化学习库。
from parl import layers      #导入PARL的层模块，用于构建神经网络。
import random
from env import ContainerNumber  # 从环境模块中导入容器数量ContainerNumber和节点数量NodeNumber。
from env import NodeNumber
flag=[]                 #初始化一个空列表flag，用于记录容器的状态。
flag_temp=[]            #初始化一个空列表flag_temp，用于临时记录容器的状态。
for o in range( ContainerNumber * NodeNumber):        #初始化flag和flag_temp列表，长度为ContainerNumber * NodeNumber，初始值为0。
    flag.append(0)
    flag_temp.append(0)

#定义了 Agent 类，负责与环境交互，选择动作，并更新模型，继承自PARL的Agent类
class Agent(parl.Agent):
    #构造函数，初始化算法、观测维度、动作维度、随机探索概率及其衰减。
    def __init__(self,
                 algorithm,
                 obs_dim,
                 act_dim,
                 e_greed,   # random exploration probability
                 e_greed_decrement=0):  
        assert isinstance(obs_dim, int) # 断言观察维度必须是整数。
        assert isinstance(act_dim, int) # 断言动作维度必须是整数。
        self.obs_dim = obs_dim #将观察维度赋值给实例变量。
        self.act_dim = act_dim #将动作维度赋值给实例变量。
        super(Agent, self).__init__(algorithm)    #调用父类的初始化方法。父类的初始化方法中包含了self.alg = algorithm

        self.global_step = 0      #初始化全局步数为0。
        self.update_target_steps = 200   #设置每200步更新一次目标网络。
        self.e_greed = e_greed   #初始化贪婪探索概率。
        self.e_greed_decrement = e_greed_decrement   #初始化贪婪探索概率的递减率。

    #构建预测程序和学习程序。预测程序用于选择最优动作，学习程序用于更新Q网络。
    #在静态图模式下，计算图需要提前定义好，然后在运行时执行。build_program 方法用于定义这些计算图。
    #即这个方法的主要作用是在静态图模式下（即使用 paddle.static 模式）定义模型的前向计算和训练流程。这对于提高模型的性能和效率非常有用，特别是在大规模训练和部署场景中。
    #静态图模式下，PaddlePaddle 可以对计算图进行优化，减少运行时的开销，提高训练和推理的速度。静态图模式下，PaddlePaddle 可以更有效地管理内存和其他资源，避免动态图模式下的内存碎片问题。
    #build_program 方法的作用是构建智能体进行预测和学习所需的 PaddlePaddle 程序。
    #PaddlePaddle 程序是描述计算流程的框架，包括数据输入、网络前向传播、损失函数计算、优化器更新等步骤。通过构建 PaddlePaddle 程序，我们可以定义智能体如何与环境交互并进行学习。
    #在预测和学习程序中，使用layers.data定义了输入数据的结构，包括状态（obs）、行动（action）、奖励（reward）、下一个状态（next_obs）和终止标志（terminal）。
    #build_program 方法在 Agent 类中起到了关键作用，它为预测动作和更新 Q 网络分别创建了独立的程序对象，并通过数据变量和相应的计算逻辑实现了这两个核心功能。这种方法的设计确保了模型训练和预测过程的稳定性和可靠性。
    def build_program(self):
        self.pred_program = fluid.Program()  # 创建一个预测程序/预测动作的 Q 值的计算图。这个程序用于生成给定状态（observation）的行动值（action values）.
        self.learn_program = fluid.Program() # 创建一个学习程序/学习和更新模型参数的计算图。这个程序用于更新神经网络的参数，即在训练过程中调整网络权重.

        with fluid.program_guard(self.pred_program):  # 使用上下文管理器保护预测程序，定义 PaddlePaddle 程序的范围，这样可以避免外部代码修改或删除该程序对象。
            obs = layers.data(                      # 定义观测数据。
                name='obs', shape=[self.obs_dim], dtype='float32')
            # 调用算法的预测方法，获取Q值。
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):  # 使用上下文管理器保护学习程序，定义 PaddlePaddle 程序的范围，这样可以避免外部代码修改或删除该程序对象。
            obs = layers.data(                   #定义观测数据
                name='obs', shape=[self.obs_dim], dtype='float32')
            action = layers.data(name='act', shape=[2], dtype='int32')    # 定义动作数据
            reward = layers.data(name='reward', shape=[], dtype='float32') #定义奖励数据。
            next_obs = layers.data(           # 定义下一个观测数据。
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool') #定义终止标志数据。
            self.cost = self.alg.learn(obs, action, reward, next_obs, terminal)   #调用算法的学习方法，获取损失值。

    #根据epsilon-greedy策略选择动作。
    def sample(self, obs):
        sample = random.random()  # 生成一个0到1之间的随机数。
        limit = ContainerNumber * NodeNumber - 1  #计算动作空间的最大值。
        if sample < self.e_greed:   #如果随机数小于贪婪探索概率，则随机选择一个动作。
            temp = random.randint(0,limit)    #随机生成一个动作索引。
            while flag[temp] == 1 or flag_temp[temp % ContainerNumber] == 1:  # 如果该动作已经被选过或对应的容器已经被选过，则重新生成动作索引。
                temp = random.randint(0,limit)
            act = temp         #将选中的动作索引赋值给act。
            flag[act] = 1      #标记该动作已被选中。
            flag_temp[act] = 1 #标记该动作对应的容器已被选中。
            flag_temp[temp % ContainerNumber] = 1  #标记该动作对应的容器已被选中。
        else: #否则，选择最优动作。
            act = self.predict(obs)  # 调用预测方法，获取最优动作。
        self.e_greed = max(          # 减少贪婪探索概率，但不低于0.01。
            0.01, self.e_greed - self.e_greed_decrement)  
        return act        #返回选择的动作

    #选择最优动作，通过排序索引选择最佳行动。
    def predict(self, obs):  # 定义预测最优动作的方法。
        # convert numbers to 1D vectors
        obs = np.expand_dims(obs, axis=0) #axis=0 表示在数组的第一个维度（即行维度）上增加一个维度，该函数将原始数组 obs 的形状从 (n,) 转换为 (1, n)
        obs = [obs]            #将观测数据转换为列表。
        obs = np.array(obs)     #将列表转换为NumPy数组。

        #pred_Q的形状为(batch_size, act_dim)，在 predict 方法中，batch_size 为 1，因为我们在处理单个样本，即pred_Q 是一个形状为 (1, act_dim) 的二维数组
        pred_Q = self.fluid_executor.run(     #运行预测程序，获取Q值。
            self.pred_program,     
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
        #fetch_list：这是一个列表，包含了你想要从执行的程序中获取的变量（或输出结果）
        #run方法返回的是一个列表，其中包含了fetch_list中指定的所有输出。因为这里fetch_list只包含了一个元素self.value，所以返回的列表只有一个元素，使用[0]来获取这个单一的元素。

        #pred_Q是一个形状为[1, act_dim]的数组，其中包含了对于每个可能行动的预测Q值。由于pred_Q是一个二维数组，np.argsort默认对最后一个维度进行排序，返回的是一个形状为[1, act_dim]的数组，其中包含了每个行动的索引。
        #例如，如果 pred_Q 是 [[0.1, 0.5, 0.3, 0.8]]，那么 np.argsort(pred_Q) 将返回 [[0, 2, 1, 3]]
        preq_Q_sorted = np.argsort(pred_Q) #获取Q值的排序索引。
        preq_Q_sorted = preq_Q_sorted[0]  #取出第一个元素。
        act = preq_Q_sorted[self.act_dim - 1]   #选择Q值最大的动作。
        i = 1         #初始化计数器。
        while flag[act] == 1 or flag_temp[act % ContainerNumber] == 1:   # 如果该动作已被选中或对应的容器已被选中，则选择次优动作。
            i += 1   #增加计数器
            act = preq_Q_sorted[self.act_dim - i]  #选择次优动作
        flag[act] = 1  #标记该动作已被选中。
        flag_temp[act] = 1 # 标记该动作对应的容器已被选中。
        flag_temp[act % ContainerNumber ] = 1 #标记该动作对应的容器已被选中。
        return act #返回选择的动作。


    #学习函数，使用DQN算法更新Q网络，同步模型参数。
    def learn(self, obs, act, reward, next_obs, terminal):
        #  Synchronize the parameters of model and target_model every 200 training steps
        if self.global_step % self.update_target_steps == 0: #如果全局步数达到更新目标网络的步数，则同步目标网络。
            self.alg.sync_target() #调用算法的同步目标网络方法。
        self.global_step += 1 #增加全局步数
        act = np.expand_dims(act, -1) #act 是一个标量，形状为 ()，-1表示在act最后一个维度上插入新的轴。如果 act 是标量 3，则 np.expand_dims(3, -1) 的结果是一个形状为 (1,) 的一维数组，即 [3]。
        feed = {              #构建输入数据字典。
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        cost = self.fluid_executor.run(   #运行学习程序，获取损失值。
            self.learn_program, feed=feed, fetch_list=[self.cost])[0] 
        return cost        #返回损失值。
