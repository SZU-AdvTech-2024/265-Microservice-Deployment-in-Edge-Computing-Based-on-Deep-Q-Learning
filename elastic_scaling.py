# import pandas as pd
# import numpy as np
# from env import Env
# from env import ContainerNumber, NodeNumber, ServiceNumber, service_containernum, service_container  #从环境模块中导入环境类和其他相关参数。
# import train
#
# #定义了弹性伸缩的逻辑，根据资源利用率进行容器扩缩容。
#
#
# sigma = 0.9 #扩容阈值，当资源利用率超过此值时触发扩容。
# delta = 0.3 #缩容阈值，当资源利用率低于此值时触发缩容。
# alpha = 0.5 #计算资源利用率时的CPU权重。
# beta = 0.5 #计算资源利用率时的内存权重。
#
# df = pd.read_excel('D:\\RL code\\BookInfo\\monitor.xlsx',header=0)  #读取监控数据文件。
# df2 = pd.read_excel('D:\\RL code\\BookInfo\\cost.xlsx',header=0)    #读取成本数据文件。
# data = df.values[1]   #获取监控数据的第一行,即当前的CPU和内存使用率，通过Prometheus生成。
# data2 = df2.values[0] #获取成本数据的第一行,即当前的成本,这个是如何生成的呢？我觉得可以直接使用env.cost()直接计算,我分析了一下这个文件的结构，第一行，内容是当前成本，当前方差，后面是执行过的所有动作序列
# usage = [0] * ServiceNumber #初始化一个长度为服务数量的列表，用于存储每个服务的资源使用情况。
# Avg = [0] * ServiceNumber   #初始化一个长度为服务数量的列表，用于存储每个服务的平均资源使用率.
#
#
#
# #计算服务的平均资源使用率，并决定是否需要伸缩。计算每个服务的平均资源使用率。如果某个服务的平均资源使用率超过阈值 sigma，则需要扩展；如果低于阈值 delta，则需要收缩。
# def SCM():
#     k = -1      #初始化服务索引为-1。
#     decision = -1    #初始化决策为-1。
#     for i in range(ServiceNumber): # 遍历所有服务
#         for j in range(service_containernum[i]): #遍历当前服务的所有容器。
#             usage[i] += alpha * data[j+i+1] + beta * data[j+i+1+ContainerNumber] #计算当前服务的资源使用情况。
#         #print("usage[",i,"]",usage[i])
#         Avg[i] = usage[i] / service_containernum[i]    #计算当前服务的平均资源使用率。
#         print("Avg[",i,"]",Avg[i]) #打印当前服务的平均资源使用率。
#         #print("       ")
#         if Avg[i] > sigma: #如果平均资源使用率超过扩容阈值。
#             k = i      #记录当前服务索引。
#             decision = 1  #设置决策为扩容。
#         elif Avg[i] < delta:  #如果平均资源使用率低于缩容阈值。
#             k = i   #记录当前服务索引。
#             decision = 0 # 设置决策为缩容。
#     return k,decision    #返回服务索引和决策。
#
#
# #定义弹性伸缩函数，接受服务索引和决策。执行伸缩操作，选择在哪个节点上增加或减少容器实例。如果决策为扩展，则选择联合成本增加最小的节点进行扩展。如果决策为收缩，则选择联合成本减少最大的节点进行收缩。
# def elastic(k,decision):
#     container = service_container[k][0]  #获取当前服务的第一个容器。
#     Cost = data2[0] #获取当前的成本
#     Var = data2[1] #获取当前的方差。
#     if decision == 1: #如果决策为扩容。
#         score = float('inf')  #初始化得分无穷大。
#         for i in range(NodeNumber): # 遍历所有节点。
#             env = Env() #创建环境对象
#             for j in range(ContainerNumber): #遍历所有容器
#                 action = int(data2[j+2])  #获取当前动作
#                 env.step(action) #执行动作
#             env.step(i*ContainerNumber+container)  #在当前节点上部署容器
#             _,tCost, tVar = env.cost() #计算新的成本和方差
#             score_cost = beta * (tCost - Cost)  #计算成本变化得分
#             score_var = alpha * (tVar - Var)      #计算方差变化得分
#             print("on node",i,",score is ",score_var + score_cost ) #打印当前节点的得分
#             if score > (score_var + score_cost): #如果当前得分更优
#                 index = i #记录当前节点索引
#                 score = score_var + score_cost #更新得分
#     elif decision == 0: #如果决策为缩容
#         score = float('-inf') #初始化得分负无穷大
#         for i in range( NodeNumber-1, -1, -1): # 逆序遍历所有节点
#             env = Env() #创建环境对象
#             for j in range(ContainerNumber): #遍历所有容器
#                 for k in range(ContainerNumber): #再次遍历所有容器
#                     if j != k: #如果不是同一个容器
#                         action = int(data2[k+2]) #获取当前动作
#                         env.step(action) #执行动作
#             _,tCost, tVar = env.cost() #计算新的成本和方差
#             score_cost = beta * (tCost - Cost) #计算成本变化得分
#             score_var = alpha * (tVar - Var) #计算方差变化得分
#             if score < (score_var + score_cost): #如果当前得分更优
#                 index = i #记录当前节点索引
#                 score = score_var + score_cost #更新得分
#     return index,container #返回最优节点索引和容器
#
# k,decision = SCM() #调用资源使用率计算函数，获取服务索引和决策。
# print("scaling container is",k,", and decision is ",decision) #打印需要进行弹性伸缩的服务索引和决策。
# index,container = elastic(k,decision) #调用弹性伸缩函数，获取最优节点索引和容器。
# print("elastice scaling is to deploy container",container,"on node", index) #打印弹性伸缩的结果。
