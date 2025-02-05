import matplotlib.pyplot as plt
import numpy as np

# 生成一些数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 创建一个独立的图例
figlegend = plt.figure(figsize=(10, 0.3)) # 指定图例的大小
ax_leg = figlegend.add_subplot(111)
# 绘制一个无可见内容的图（只为了创建图例）
ax_leg.plot([], [], label='RL-P')
ax_leg.plot([], [], label='RL-P(no mask)')
ax_leg.plot([], [], label='HAC')
ax_leg.plot([], [], label='A2C')
ax_leg.axis('off') # 关闭坐标轴
# 添加图例
figlegend.legend(loc='center',ncol=4, fontsize=8)
plt.savefig(fname="legend.png")
figlegend.show()