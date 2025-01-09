#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in temppliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-


from model import Model
from algorithm import DQN
from agent import Agent
from env import Env
from replay_memory import ReplayMemory
from env import ContainerNumber
from env import NodeNumber
from CreateDeployment import  CreateDeployment

# 定义训练参数
MEMORY_SIZE = 10000  # 回放缓存的最大大小
LEARNING_RATE = 0.001  # 学习率
GAMMA = 0.9  # 奖励折扣因子

# 定义运行一次episode的函数
def run_episode(env, agent, rpm):
    obs, action = env.reset()  # 重置环境，获取初始观察值和动作
    actionlist=[]
    while True:
        action = agent.sample(obs)  # 根据当前观察值选择动作
        actionlist.append(env.index_to_act(action))
        next_obs, cost, done, _, _ = env.step(action)  # 执行动作，获取下一步观察值、成本、是否结束等信息
        obs = next_obs  # 更新当前观察值为下一步观察值
        if done:  # 如果当前episode结束
            break
    comm=env.sumCost()
    var=env.CalcuVar()
    cost,_,_=env.cost()
    return  actionlist,comm,var,cost# 返回总奖励和总成本

def main():

    env = Env()  # 创建环境实例
    action_dim = ContainerNumber * NodeNumber  # 动作维度
    obs_shape = ContainerNumber * 3 + NodeNumber * (ContainerNumber + 2)  # 观察值维度
    rpm = ReplayMemory(MEMORY_SIZE)  # 创建回放缓存实例
    model = Model(act_dim=action_dim)  # 创建模型实例
    algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)  # 创建DQN算法实例
    agent = Agent(
        algorithm,
        obs_dim=obs_shape,
        act_dim=action_dim,
        e_greed=0.2,
        e_greed_decrement=1e-6)  # 创建代理实例

    # load model# 加载模型（如果需要）
    save_path = './dqn_model.ckpt'
    agent.restore(save_path)
    actionlist,comm,var,cost=run_episode(env, agent, rpm)
    print(actionlist)
    print("通信开销：",comm)
    print("负载方差：", var)
    print("总成本：", cost)
    createdeployment = CreateDeployment()
    for a in actionlist:
        NodeIndex = a[0] + 1
        if a[1]==0:
            createdeployment.createproductpage(NodeIndex)
            print(a,"将服务Product部署到node",NodeIndex,sep='')
        elif a[1]==1:
            createdeployment.createdetails(NodeIndex)
            print(a, "将服务Details部署到结node", NodeIndex,sep='')
        elif a[1]==5:
            createdeployment.createrating(NodeIndex)
            print(a, "将服务Ratings部署到node", NodeIndex,sep='')
        else:
            version=a[1]-1
            createdeployment.createreviews(version,NodeIndex)
            print(a, "将服务Reviews-v",version,"部署到node", NodeIndex,sep='')
    return comm, var  # 返回通信开销和负载方差


if __name__ == '__main__':
    main()

