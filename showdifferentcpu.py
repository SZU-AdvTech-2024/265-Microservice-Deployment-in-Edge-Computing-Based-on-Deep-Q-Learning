import matplotlib.pyplot as plt
import numpy as np

# 数据
strategies = ['DEFAULT', 'IA', 'RSDQL']
average_values = [1913, 2322, 1901]
error_values = [1000, 1800, 989]

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制柱状图
bars = ax.bar(strategies, average_values, color='orange')

# 计算误差线的上半部分位置
upper_errors = [err for err in error_values]
lower_errors = [0] * len(error_values)  # 下半部分误差线设置为0

# 绘制误差线的上半部分
ax.errorbar(np.arange(len(strategies)), average_values, yerr=[lower_errors, upper_errors],
            fmt='none', ecolor='black', capsize=10, capthick=2, elinewidth=2)

# 添加标题和标签
# ax.set_title('Average Memory Usage (MB)')
ax.set_xlabel('Strategy')
ax.set_ylabel('Average (m)')

# 设置网格
ax.grid(False)
# ax.set_ylim(0, 1200)
# 显示图形
plt.show()