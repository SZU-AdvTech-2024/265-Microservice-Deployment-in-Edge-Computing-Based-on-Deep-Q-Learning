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

# Modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py

import random
import collections
import numpy as np

#定义了ReplayMemory类，用于存储和采样经验回放
# 定义经验回放缓冲区类
class ReplayMemory(object):
    # 初始化经验回放缓冲区
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)  # 使用双端队列存储经验，最大长度为 max_size

    # 添加一条经验到缓冲区
    def append(self, exp):
        self.buffer.append(exp)  # 将经验 exp 添加到双端队列中

    # 从缓冲区中随机抽取一批经验
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)  # 随机抽取 batch_size 条经验
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []
        for experience in mini_batch:
            s, a, r, s_p, done = experience  # 解包经验 (obs, action, reward, next_obs, done)
            # 将解包的经验分别添加到对应的批次列表中
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)
        # 将批次列表转换为 NumPy 数组并返回
        return np.array(obs_batch).astype('float32'), \
               np.array(action_batch).astype('float32'), \
               np.array(reward_batch).astype('float32'), \
               np.array(next_obs_batch).astype('float32'), \
               np.array(done_batch).astype('float32')

    def __len__(self):
        # 返回缓冲区的当前长度
        return len(self.buffer)
