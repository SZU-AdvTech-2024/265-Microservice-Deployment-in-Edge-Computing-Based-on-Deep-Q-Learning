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

import os
import numpy as np
import parl
from parl.utils import logger  # 引入PARL的日志模块
import logging

from model import Model
from algorithm import DQN
from agent import Agent
from env import Env

from replay_memory import ReplayMemory

import env
from agent import flag
from agent import flag_temp
from env import ContainerNumber
from env import NodeNumber

# 定义训练参数
LEARN_FREQ = 6  # 学习频率
MEMORY_SIZE = 10000  # 回放缓存的最大大小
MEMORY_WARMUP_SIZE = 2000  # 预热阶段需要的最小样本数
BATCH_SIZE = 30  # 每次采样的批次大小
LEARNING_RATE = 0.001  # 学习率
GAMMA = 0.9  # 奖励折扣因子
# 初始化全局变量
sc_comm = 0  # 通信开销计数器
sc_var = 0  # 负载方差计数器
flag1 = 1  # 标志变量
ep = 0  # 当前轮次数
allCost = [[], [], [], [], [], []]  # 记录每步的成本历史记录
test_reward = 0  # 测试奖励
test_evareward = 0  # 测试平均奖励


# 定义运行一次episode的函数
def run_episode(env, agent, rpm):
    global flag1  # 全局标志变量
    global allCost  # 全局成本记录
    global ep  # 全局轮次数
    global test_reward  # 全局测试奖励
    obs_list = []  # 观察值列表
    next_obslist = []  # 下一步观察值列表
    action_list = []  # 动作列表
    done_list = []  # 是否结束列表
    total_reward = 0  # 总奖励
    total_cost = 0  # 总成本
    ep += 1  # 轮次数加1
    obs, action = env.reset()  # 重置环境，获取初始观察值和动作
    step = 0  # 步数初始化
    mini = -1  # 最小成本初始化
    co = 0  # 成本计数器初始化
    for o in range(ContainerNumber * NodeNumber):  # 初始化标志变量,记录已经部署的容器以及其所在结点
        flag_temp[o] = 0
        flag[o] = 0
    flag1 -= 1  # 标志变量减1

    while True:
        reward = 0  # 奖励初始化
        step += 1  # 步数加1
        obs_list.append(obs)  # 记录当前观察值

        action = agent.sample(obs)  # 根据当前观察值选择动作
        action_list.append(action)  # 记录当前动作
        next_obs, cost, done, _, _ = env.step(action)  # 执行动作，获取下一步观察值、成本、是否结束等信息
        next_obslist.append(next_obs)  # 记录下一步观察值
        done_list.append(done)  # 记录是否结束

        if allCost[step - 1]:  # 如果当前步骤有成本记录
            mini = min(allCost[step - 1])  # 更新最小成本
        if flag1 == 0:  # 如果是第一个episode
            # if it's the first episode, save the cost directly
            if cost > 0:  # 如果成本大于0
                allCost[step - 1].append(cost)  # 记录成本
                reward = 0  # 奖励设为0
                co += 1  # 成本计数器加1
            else:   #说明操作失败或有特殊情况。清除之前所有记录的成本。这是因为当前步骤的成本不大于0，可能意味着操作失败，之前的成本记录不再有效。终止当前episode，不再继续执行后续步骤。
                flag1 += 1  # 标志变量加1
                for i in range(co):  # 清除之前的成本记录
                    allCost[step - 1 - (i + 1)].clear()
                break
        else:  # 不是第一个episode
            if cost > 0:  # 如果成本大于0
                if step == 6:  # 如果是第6步
                    if abs(min(allCost[step - 1]) - cost) < 0.0001:  # 如果当前成本接近最小成本
                        reward = test_reward  # 奖励设为测试奖励
                    elif (min(allCost[step - 1]) - cost) > 0:  # 如果当前成本小于最小成本
                        test_reward = test_reward + 100  # 测试奖励增加100
                        reward = test_reward  # 奖励设为测试奖励
                    else:
                        reward = 10 * (min(allCost[step - 1]) - cost)  # 奖励设为成本差的10倍
                    for i in range(6):  # 将当前步骤的数据添加到回放缓存,在确定了最终的奖励后，将这个奖励分配给该episode中的所有步骤,这里体现了文中的奖励共享思想
                        rpm.append((obs_list[i], action_list[i], reward, next_obslist[i], done_list[i]))
                    allCost[step - 1].append(cost)  # 记录当前成本
            else:
                reward = -100  # 成本不大于0，当前行为违反约束，将受到处罚，奖励设为-100
                rpm.append((obs, action, reward, next_obs, done))  # 将当前步骤的数据添加到回放缓存

        root_logger = logging.getLogger()  # 获取根日志记录器，根日志记录器是所有日志记录器的顶级记录器，用于捕获所有未被捕获的日志消息。
        for h in root_logger.handlers[:]:  # 移除所有处理程序，这样做是为了确保没有旧的处理程序干扰新的日志记录。
            root_logger.removeHandler(h)
        logging.basicConfig(level=logging.INFO, filename='details.log')  # 设置日志级别和文件名，设置日志记录的最低级别为 INFO，这意味着只有 INFO 及以上级别的日志消息会被记录。
        logging.info('episode:{}  step:{} Cost:{} min Cost:{} Reward:{} global reward:{} Action:{}'.format(
            ep, step, cost, mini, reward, test_reward, env.index_to_act(action)))  # 记录日志信息

        # # 训练模型
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):  # 如果回放缓存足够大且达到学习频率
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)  # 从回放缓存中随机采样一批数据

            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_done)  # 使用采样数据训练模型
            with open("trainloss.txt", "a") as f:  # 记录训练损失
                f.write("%d,%.3f \n" % (ep, train_loss))
        total_reward += reward  # 累加总奖励
        total_cost += cost  # 累加总成本
        obs = next_obs  # 更新当前观察值为下一步观察值
        if done:  # 如果当前episode结束
            break
    return total_reward, total_cost  # 返回总奖励和总成本


# # 定义评估代理的函数
def evaluate(env, agent):
    global sc_comm, sc_var  # 全局通信开销和负载方差计数器
    eval_totalCost = []  # 记录评估过程中的总成本
    eval_totalReward = []  # 记录评估过程中的总奖励
    reward = 0  # 奖励初始化
    test_evareward = 0  # 测试平均奖励初始化
    for i in range(1):  # 进行一次评估
        env.prepare()  # 准备环境
        obs = env.update()  # 更新环境状态
        for o in range(ContainerNumber * NodeNumber):  # 初始化标志变量
            flag_temp[o] = 0
            flag[o] = 0

        episode_cost = 0  # 当前episode的成本初始化
        episode_reward = 0  # 当前episode的奖励初始化
        step = 0  # 步数初始化
        while True:
            step += 1  # 步数加1
            action = agent.predict(obs)  # 根据当前观察值选择最优动作
            obs, cost, done, comm, var = env.step(action)  # 执行动作，获取下一步观察值、成本、是否结束等信息
            if cost > 0:  # 如果成本大于0
                if step == 6:  # 如果是第6步
                    if abs(min(allCost[step - 1]) - cost) < 0.0001:  # 如果当前成本接近最小成本
                        reward = test_evareward  # 奖励设为测试平均奖励
                    elif min(allCost[step - 1]) - cost > 0:  # 如果当前成本小于最小成本
                        test_evareward += 100  # 测试平均奖励增加100
                        reward = test_evareward  # 奖励设为测试平均奖励
                    else:
                        reward = 10 * (min(allCost[step - 1]) - cost)  # 奖励设为成本差的10倍
            else:
                reward = -100  # 成本不大于0，奖励设为-100
            episode_cost = cost  # 记录当前episode的成本
            episode_reward = reward  # 记录当前episode的奖励
            sc_comm = comm  # 记录通信开销
            sc_var = var  # 记录负载方差
            if done:  # 如果当前episode结束
                break
        eval_totalCost.append(episode_cost)  # 记录当前episode的成本
        eval_totalReward.append(episode_reward)  # 记录当前episode的奖励
    return eval_totalCost, eval_totalReward, sc_comm, sc_var  # 返回评估结果


def main():
    global sc_comm, sc_var  # 全局通信开销和负载方差计数器
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
    # save_path = './dqn_model.ckpt'
    # agent.restore(save_path)

    # 预热阶段，填充回放缓存
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(env, agent, rpm)

    max_episode = 2000  # 最大训练轮次数

    # # 开始训练
    episode = 0  # 当前轮次数
    while episode < max_episode:
        # 训练部分
        for i in range(0, 50):
            total_reward, _ = run_episode(env, agent, rpm)  # 运行一次episode
            episode += 1  # 轮次数加1
            with open("reward.txt", "a") as q:  # 记录奖励
                q.write("%05d,%.3f \n" % (episode, total_reward))

        # 测试部分
        eval_totalCost, eval_totalReward, sc_comm, sc_var = evaluate(env, agent)  # 评估代理
        with open("cost.txt", "a") as f:  # 记录成本
            f.write("%d,%.6f \n" % (episode, np.mean(eval_totalCost)))
        root_logger = logging.getLogger()  # 获取根日志记录器
        for h in root_logger.handlers[:]:  # 移除所有处理程序
            root_logger.removeHandler(h)

        logging.basicConfig(level=logging.INFO, filename='a.log')  # 设置日志级别和文件名
        logging.info('episode:{} e_greed:{} Cost: {} Reward:{} Action:{}'.format(
            episode, agent.e_greed, np.mean(eval_totalCost), np.mean(eval_totalReward), env.action_queue))  # 记录日志信息

    # 训练结束后保存模型
    save_path = './dqn_model.ckpt'
    agent.save(save_path)
    return sc_comm, sc_var  # 返回通信开销和负载方差


if __name__ == '__main__':
    main()
