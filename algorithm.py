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

import copy
import paddle.fluid as fluid
import parl
from parl import layers

#parl.Algorithm 类是 PaddlePaddle 强化学习库中定义强化学习算法的重要工具，它提供了强化学习算法框架和接口，方便开发者实现和训练各种强化学习算法。
#它通过与parl.Model类和parl.Agent类的协作，共同构成了强化学习训练过程的核心部分。
#Model类负责定义网络的前向计算部分，而Algorithm类则定义了如何使用这些数据来更新模型。Agent类则负责与环境进行交互，收集数据，并将这些数据提供给Algorithm类进行模型更新。
#定义了 DQN 算法，包含预测和学习的逻辑，用于训练Q网络。
class DQN(parl.Algorithm):
    #构造函数，初始化模型、动作维度、折扣因子和学习率。
    def __init__(self, model, act_dim=None, gamma=None, lr=None):
        """ DQN algorithm
        
        Args:
            model (parl.Model): the network forwarding structure of the Q function
            act_dim (int): dimensions of action
            gamma (float): attenuation factor of reward
            lr (float): learning_rate
        """
        self.model = model #将模型赋值给实例变量。
        self.target_model = copy.deepcopy(model)  #深拷贝模型，创建目标模型

        assert isinstance(act_dim, int) #断言动作维度必须是整数。
        assert isinstance(gamma, float) #断言奖励衰减因子必须是浮点数。
        assert isinstance(lr, float)  #断言学习率必须是浮点数。
        self.act_dim = act_dim #将动作维度赋值给实例变量。
        self.gamma = gamma #将奖励衰减因子赋值给实例变量
        self.lr = lr #将学习率赋值给实例变量。

    #预测函数，接受观测数据,返回当前状态下所有动作的Q值。
    def predict(self, obs):
        """ use value network of self.model to get [Q(s,a1),Q(s,a2),...]
        """
        return self.model.value(obs)      #调用模型的value方法，返回所有动作的Q值。

    #学习函数，更新Q网络参数。接受观测数据、动作、奖励、下一个观测数据和终止标志。
    # obs 的形状为 [batch_size, obs_dim]，其中：batch_size 是批量大小，表示一次输入的数据样本数量。obs_dim 是观测数据的维度，表示每个观测数据的特征数量。action 的形状为 [batch_size]，其它输入参数以此类推。
    def learn(self, obs, action, reward, next_obs, terminal):
        """ use DQN algorithm to update value network of self.model
        """
        next_pred_value = self.target_model.value(next_obs) #使用目标模型预测下一个状态的所有动作的Q值。
        best_v = layers.reduce_max(next_pred_value, dim=1)  #获取下一个状态的最大Q值。
        # 阻止梯度传播，如果设置为 True，则在反向传播过程中不会计算该变量的梯度。确保目标网络的参数在计算目标Q值时不会被更新。
        #在 DQN 算法中，我们通常使用贝尔曼方程来计算目标 Q 值，即 target_q = reward + gamma * best_v。
        # 在这个过程中，best_v 是从目标网络预测的下一个状态的最大 Q 值中得到的，我们不希望在更新主网络的 Q 值时，best_v 的梯度影响主网络的 Q 值的更新。
        best_v.stop_gradient = True

        terminal = layers.cast(terminal, dtype='float32')#  将终止标志转换为浮点数，True为1，False为0。
        target = reward + (1.0 - terminal) * self.gamma * best_v  # 计算目标Q值。

        # forward propagation
        pred_value = self.model.value(obs)   # 使用当前模型预测当前状态的所有动作的Q值。
        # Convert action to onehot vector
        action_onehot = layers.one_hot(action, self.act_dim) # 将动作转换为one-hot向量。self.act_dim表示动作空间的维度，即可能的动作数量。
        action_onehot = layers.cast(action_onehot, dtype='float32')  # 将one-hot向量转换为浮点数。

        pred_action_value = layers.reduce_sum(            #计算选定动作的Q值。
            layers.elementwise_mul(action_onehot, pred_value), dim=1)

        # get loss
        cost = layers.square_error_cost(pred_action_value, target)     #计算平方误差损失。
        cost = layers.reduce_mean(cost)    #计算损失的平均值。
        optimizer = fluid.optimizer.Adam(learning_rate=self.lr)  # 使用Adam优化器。
        optimizer.minimize(cost) #最小化损失。
        return cost #返回损失值

    #同步模型参数到目标模型。
    def sync_target(self):
        """ Synchronize the model parameter values of self.model to self.target_model
        """
        self.model.sync_weights_to(self.target_model) #将当前模型的参数同步到目标模型，sync_weights_to方法是parl.Algorithm类中的一个方法，用于同步模型参数到目标模型
