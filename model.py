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

import parl
from parl import layers
#定义了定义了Q网络模型，用于预测 Q 值，使用PaddlePaddle框架实现。
#继承自PARL的模型类，定义了三层全连接神经网络。
# 定义模型类，parl.Model 类是 PaddlePaddle 强化学习库中定义神经网络模型的基类。它提供了神经网络模型构建的框架和接口，方便开发者定义和训练强化学习算法中的模型。
#parl.Model类提供了一个完整的神经网络结构，包括输入层、隐藏层和输出层。通过继承parl.Model类，可以方便地定义和初始化DQN算法所需的模型参数。parl.Model 类允许用户定义神经网络的结构，包括层的类型、层数、激活函数等。
#在强化学习算法中，Model 类通常用于定义价值网络（如 DQN 中的 Q 网络）或策略网络（如 Actor-Critic 方法中的策略网络）。
class Model(parl.Model):
    def __init__(self, act_dim):
        # 初始化模型
        hid1_size = 128  # 第一个隐藏层的神经元数
        hid2_size = 128  # 第二个隐藏层的神经元数
        self.fc1 = layers.fc(size=hid1_size, act='relu')  # 第一个全连接层，激活函数为ReLU
        self.fc2 = layers.fc(size=hid2_size, act='relu')  # 第二个全连接层，激活函数为ReLU
        self.fc3 = layers.fc(size=act_dim, act=None)  # 输出层，没有激活函数，输出层的尺寸由act_dim参数指定，表示动作空间的大小

#parl.Model类中包含一个方法value，用于计算给定状态的值函数。该方法接收一个状态向量作为输入，经过一系列的前向传播操作后，返回对应的Q值。这个Q值表示在给定状态下采取不同行动的期望回报。
    def value(self, obs):
        # 顺序来调用init方法中定义的神经网络层，计算给定状态下的动作价值
        h1 = self.fc1(obs)  # 第一个全连接层的输出
        h2 = self.fc2(h1)  # 第二个全连接层的输出
        Q = self.fc3(h2)  # 输出层的输出，即每个动作的价值
        return Q  # 返回每个动作的价值
