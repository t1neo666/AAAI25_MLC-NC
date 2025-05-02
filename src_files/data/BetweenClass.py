import numpy as np
import matplotlib.pyplot as plt

# 加载数据
longtail = np.load('C:\\Users\\Y\\Desktop\\Bcal\\BCaL\\appendix\\VOCdevkit\\longtail2012\\class_freq.pkl', allow_pickle=True)
longtail_freq = longtail['class_freq']

# 获取按值排序的索引
sorted_indices = np.argsort(longtail_freq)[::-1]
sorted_indices = sorted_indices.tolist()
sorted_indices_new = [str(x) for x in sorted_indices]

# 按排序后的索引获取对应的值
sorted_values = longtail_freq[sorted_indices]
sorted_values = sorted_values.tolist()

# 绘制索引与值的图表
plt.figure(figsize=(15, 7))

# 创建柱状图
plt.bar(sorted_indices_new, sorted_values)

# 去掉x轴的刻度标签
plt.xticks([])  # 这里设置为空列表，去掉x轴的数字
plt.yticks([])

# 去掉外边框
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)


# 保存图像为SVG格式
plt.savefig("voclongtail.svg", dpi=300, format="svg")

# 显示图像
plt.show()