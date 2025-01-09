import matplotlib.pyplot as plt

# 假设有以下数据
concurrent_requests = [100, 200, 300, 400]
default_response_time = [115, 987, 1649, 2505]
ia_response_time = [99, 865, 1637, 2121]
rsdol_response_time = [72, 817, 1685, 2278]

# 创建图表和轴对象
fig, ax = plt.subplots()

# 绘制折线图
ax.plot(concurrent_requests, default_response_time, 'ks-', label='DEFAULT')
ax.plot(concurrent_requests, ia_response_time, 'ro-', label='IA')
ax.plot(concurrent_requests, rsdol_response_time, 'b^-', label='RSDQL')

# 设置x轴和y轴标签
ax.set_xlabel('Number of concurrent requests')
ax.set_ylabel('Response Time(ms)')

# 设置图表标题
ax.set_title('(a) TP99')

# 设置y轴的范围
ax.set_ylim(0, 20000)

# 添加图例
ax.legend()

# 显示图表
plt.show()

#TP90的数据
#default_response_time = [83, 872, 1564, 2396]
#ia_response_time = [78, 731, 1551, 1950]
#rsdol_response_time = [66, 761, 1540, 2119]

#TP95的数据
#default_response_time = [91, 916, 1614, 2457]
#ia_response_time = [91, 768, 1590, 2012]
#rsdol_response_time = [67, 790, 1575, 2207]

#TP99的数据
#default_response_time = [115, 987, 1649, 2505]
#ia_response_time = [99, 865, 1637, 2121]
#rsdol_response_time = [72, 817, 1685, 2278]