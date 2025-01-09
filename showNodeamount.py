import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.lines import Line2D

# 示例数据
services = ['Details', 'Productpage', 'Reviews', 'Ratings']

# 指定要生成的点的坐标
points = {
    'DEFAULT': [
        ('Details', 3, 1),
        ('Productpage', 4, 1),
        ('Reviews', 4, 2),
        ('Reviews', 3, 1),
        ('Ratings', 3, 1)
    ],
    'IA': [
        ('Details', 0, 1),
        ('Productpage', 0, 1),
        ('Reviews', 0, 3),
        ('Ratings', 0, 1)
    ],
    'RSDQL': [
        ('Details', 4, 1),
        ('Productpage', 3, 1),
        ('Reviews', 1, 1),
        ('Ratings', 1, 1),
        ('Reviews', 2, 1),
        ('Reviews', 4, 1)
    ]
}

# 创建3D向量数组来表示所有的点
all_points = []
colors = []
labels = []

for method, coords in points.items():
    for coord in coords:
        service, node, container_number = coord
        x = services.index(service)  # Service index
        y = node - 1  # Node index (assuming nodes start from 1, but array indices start from 0)
        z = container_number
        all_points.append([x, y, z])
        colors.append('blue' if method == 'DEFAULT' else 'orange' if method == 'IA' else 'green')
        labels.append(method)

all_points = np.array(all_points)
colors = np.array(colors)

# 创建3D图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图
scatter = ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], c=colors, s=150, alpha=0.8)

# 绘制垂直线
for point in all_points:
    ax.plot([point[0], point[0]], [point[1], point[1]], [0, point[2]], color='gray', linestyle='--', linewidth=0.5)

# 设置轴标签
ax.set_xlabel('Service')
ax.set_ylabel('Node')
ax.set_zlabel('Container Number', labelpad=10)

# 设置轴刻度
ax.set_xticks(np.arange(len(services)))
ax.set_xticklabels(services)
ax.set_yticks(np.arange(5))  # Adjust y ticks to include 0 to 4

# 设置z轴的位置
ax.set_zticks([0, 1, 2, 3, 4])
ax.set_zticklabels(['0', '1', '2', '3', '4'], fontsize=8)

# 添加图例
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='DEFAULT', markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='IA', markerfacecolor='orange', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='RSDQL', markerfacecolor='green', markersize=10)
]
ax.legend(handles=legend_elements)

# 显示图形
plt.show()