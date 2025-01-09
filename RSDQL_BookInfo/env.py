#定义了环境类，模拟微服务部署环境，并提供状态更新和成本计算等功能。包括节点状态、容器状态，通信权重、距离等信息等，用于模拟微服务部署的环境

#ContainerNumber, NodeNumber, ServiceNumber, ResourceType: 定义了容器数量、节点数量、服务数量和资源类型。
#service_containernum, service_container, service_container_relationship: 描述了每个服务包含的容器数、每个服务的容器列表及容器之间的关系。
#alpha, beta: 奖励权重因子。
#CPUnum, Mem: 节点的CPU核心数和内存大小。
ContainerNumber = 6  #容器总数
NodeNumber = 5 #节点总数
ServiceNumber = 4 #服务总数
ResourceType = 2 # 资源类型数
service_containernum = [1,1,3,1] #每个服务的容器数
service_container = [[0],[1],[2,3,4],[5]]  #每个服务对应的容器索引列表
service_container_relationship = [0,1,2,2,2,3] #每个容器所属的服务索引
alpha = 0.5 # 奖励权重因子，用于计算最终奖励
beta =[0.5,0.5] #用于计算节点负载方差的权重向量
count = 0 #计数器，用于记录某些操作的次数
CPUnum = 4 #每个节点的CPU核数
Mem = 4*1024 #每个节点的内存大小（以MB为单位）

import collections
import random
import numpy as np
import agent

class Env():

    def __init__(self):
        # 初始化状态
        self.State = []  # 完整状态
        self.node_state_queue = []  # 节点状态队列
        self.container_state_queue = []  # 容器状态队列
        self.action_queue = []  # 动作队列
        self.prepare()  # 准备初始状态

    ##初始化环境状态，设置初始容器和节点状态。
    def prepare(self):
        # 初始化容器状态队列
        self.container_state_queue = [
            -1, 0.5 / CPUnum, 128 / Mem,  # 容器0的状态：节点索引，CPU使用率，内存使用率
            -1, 0.5 / CPUnum, 256 / Mem,  # 容器1的状态
            -1, 0.5 / CPUnum, 256 / Mem,  # 容器2的状态
            -1, 0.5 / CPUnum, 256 / Mem,  # 容器3的状态
            -1, 0.5 / CPUnum, 256 / Mem,  # 容器4的状态
            -1, 0.5 / CPUnum, 128 / Mem   # 容器5的状态
        ]

        # 初始化节点状态队列
        for i in range(NodeNumber):
            # 添加每个节点的状态到列表，状态包括CPU使用率，内存使用率，以及其他辅助信息
            #node_state_queue中第几个元素就代表第几个结点的状态，每个元素是长度为8的队列
            #如果节点1的状态为[1, 0, 1, 0, 0, 0, 0.5, 0.3]，表示节点1上部署了容器1和容器3，节点1的CPU使用率为0.5，内存使用率为0.3
            self.node_state_queue.extend([0, 0, 0, 0, 0, 0, 0, 0])
        # 合并容器状态队列和节点状态队列，形成完整状态
        self.State = self.container_state_queue + self.node_state_queue
        self.action = [-1, -1]  # 当前动作
        self.action_queue = [-1, -1]  # 动作队列
        # 定义微服务之间的通信权重矩阵
        self.service_weight = [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 2],
            [0, 0, 2, 0]
        ]
        # 定义节点之间的通信距离矩阵
        self.Dist = [
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0]
        ]

   #计算两个容器之间的通信距离。
    def ContainerCost(self, i, j):
        # 计算容器i和j之间的通信距离
        m = -1
        n = -1
        m = self.container_state_queue[i * 3]  # 获取容器i所在的节点索引
        n = self.container_state_queue[j * 3]  # 获取容器j所在的节点索引
        p = service_container_relationship[i]  # 获取容器i所属的服务索引
        q = service_container_relationship[j]  # 获取容器j所属的服务索引
        if self.Dist[m][n] != 0 and (p != q):
            container_dist = self.Dist[m][n]  # 如果节点距离不为0且容器属于不同服务，则返回通信距离
        else:
            container_dist = 0  # 否则返回0
        return container_dist

    #计算两个容器之间的通信成本。
    def CalcuCost(self, i, j):
        # 计算容器i和j之间的通信成本
        cost = 0
        interaction = self.service_weight[i][j] / (service_containernum[i] * service_containernum[j])  # 计算服务i和j之间的交互频率
        for k in range(len(service_container[i])):
            for l in range(len(service_container[j])):
                cost += self.ContainerCost(service_container[i][k], service_container[j][l]) * interaction  # 累加每对容器之间的通信距离乘以交互频率
        return cost

    #计算总的通信开销。
    def sumCost(self):
        # 计算总通信成本
        Cost = 0
        for i in range(ServiceNumber):
            for j in range(ServiceNumber):
                Cost += self.CalcuCost(i, j)  # 累加每对服务之间的通信成本
        return 0.5 * Cost  # 返回总通信成本的一半

   #计算节点负载的方差。
    def CalcuVar(self):
        # 计算节点负载方差
        NodeCPU = []
        NodeMemory = []
        Var = 0
        for i in range(NodeNumber):
            U = self.node_state_queue[i * (ContainerNumber + 2) + ContainerNumber]  # 获取节点i的CPU使用率
            M = self.node_state_queue[i * (ContainerNumber + 2) + (ContainerNumber + 1)]  # 获取节点i的内存使用率
            NodeCPU.append(U)
            NodeMemory.append(M)
        Var += beta[0] * np.var(NodeCPU) + beta[1] * np.var(NodeMemory)  # 计算节点CPU和内存使用率的方差，并根据权重向量加权求和
        return Var

    #计算总成本，包括通信开销和负载平衡成本。
    def cost(self):
        # 计算总成本
        re = 0
        g1 = self.sumCost()  # 计算总通信成本
        g1 = g1 / 4  # 归一化总通信成本
        g2 = self.CalcuVar()  # 计算节点负载方差
        g2 = g2 / 0.052812500000000005  # 归一化节点负载方差
        re += alpha * g1 + (1 - alpha) * g2  # 根据权重因子计算总成本
        return 100 * re, g1, g2  # 返回总成本及其组成部分

    #更新环境状态操作。
    def state_update(self, container_state_queue, node_state_queue):
        # 更新状态
        self.State = container_state_queue + node_state_queue  # 更新容器状态队列和节点状态队列，重新组合成完整状态

    # 更新环境状态。
    def update(self):
        # 更新环境状态
        if self.action[0] >= 0 and self.action[1] >= 0:
            # 更新容器状态
            self.container_state_queue[self.action[1] * 3] = self.action[0]  # 将容器分配到指定节点
            # 更新节点状态
            self.node_state_queue[self.action[0] * (ContainerNumber + 2) + self.action[1]] = 1  # 更新节点的容器分配状态
            self.node_state_queue[self.action[0] * (ContainerNumber + 2) + ContainerNumber] += \
            self.container_state_queue[self.action[1] * 3 + 1]  # 增加节点的CPU使用率
            self.node_state_queue[self.action[0] * (ContainerNumber + 2) + (ContainerNumber + 1)] += \
            self.container_state_queue[self.action[1] * 3 + 2]  # 增加节点的内存使用率
            self.action_queue.append(self.action)  # 将当前动作添加到动作队列
        else:
            print("invalid action")  # 如果动作无效，打印错误信息
            self.node_state_queue = []  # 清空节点状态队列
            self.container_state_queue = []  # 清空容器状态队列
            self.action_queue = []  # 清空动作队列
            self.prepare()  # 重新准备初始状态
        self.state_update(self.container_state_queue, self.node_state_queue)  # 更新状态
        return self.State  # 返回新的状态

    #执行一步操作，返回新的状态、成本、是否完成标志和通信开销。
    def step(self, action):
        # 输入：action(Targetnode，ContainerIndex)
        # 输出：next state, cost, done
        global count
        self.action = self.index_to_act(action)  # 将输入的动作转换为实际的动作格式
        self.update()  # 更新环境状态
        cost, comm, var = self.cost()  # 计算当前状态的成本、通信成本和负载方差
        done = False  # 是否完成标志
        count = 0  # 初始化计数器
        for i in range(ContainerNumber):
            if self.container_state_queue[3 * i] != -1:
                count += 1  # 统计已分配的容器数
        if count == ContainerNumber:
            done = True  # 如果所有容器都已分配完毕，设置完成标志为True
        return self.State, cost, done, comm, var  # 返回新的状态、成本、是否完成标志、通信成本和负载方差

    #重置环境状态，准备新的状态。
    def reset(self):
        # 重置环境
        self.node_state_queue = []  # 清空节点状态队列
        self.container_state_queue = []  # 清空容器状态队列
        self.prepare()  # 重新准备初始状态
        return self.State, self.action  # 返回初始状态和初始动作

    #将索引转换为实际的动作。假设 ContainerNumber = 6，NodeNumber = 5，那么 index 的取值范围是从0到29（即 ContainerNumber * NodeNumber - 1）
    def index_to_act(self, index):
        # 将索引转换为动作，动作表示将某个容器部署到某个节点上。
        act = [-1, -1]
        act[0] = int(index / ContainerNumber)  # 目标节点索引
        act[1] = index % ContainerNumber  # 容器索引
        return act  # 返回转换后的动作
