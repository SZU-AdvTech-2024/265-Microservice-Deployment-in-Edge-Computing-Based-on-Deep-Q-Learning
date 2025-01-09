import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 示例数据
np.random.seed(0)
x = np.random.normal(size=100)
y1 = np.random.normal(size=100)
y2 = np.random.normal(size=100)
y3 = np.random.normal(size=100)

# 数据集
data = np.column_stack((x + y1, x + y2, x + y3))
columns = ['DEFAULT', 'IA', 'RSDQL']

# 创建DataFrame
df = pd.DataFrame(data, columns=columns)

# 设置绘图样式
sns.set(style="whitegrid")

# 创建小提琴图
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, inner=None, palette="Set3")

# 添加标签和标题
plt.xlabel('Number of concurrent requests')
plt.ylabel('Response Time(ms)')
plt.title('Violin Plot of Response Time')
plt.grid(False)
# 显示图形
plt.show()
