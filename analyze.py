import matplotlib.pyplot as plt
import numpy as np

# 两个列表的数据
list1 = [1.3578892, 1.4228259, 1.3371569, 1.433261, 1.3381737, 1.3701596, 1.4387664,
         1.3940419, 1.376736, 1.3927791, 1.3790576, 1.3762774, 1.4065428, 1.3706331,
         1.3776891, 1.4200755, 1.4037144, 1.3544915, 1.4094682, 1.4193304]
list2 = [1.3556426, 1.4190977, 1.3352901, 1.4300745, 1.3347245, 1.3666098, 1.4357129,
         1.3906316, 1.3774798, 1.3917049, 1.3772248, 1.3725144, 1.4025202, 1.3676552,
         1.3829322, 1.4197714, 1.399909, 1.3513951, 1.4082041, 1.415456]

# 设置x轴的标签
labels = [f'Feature {i+1}' for i in range(len(list1))]

x = np.arange(len(labels))  # 标签位置
width = 0.35  # 柱状图的宽度

fig, ax = plt.subplots(figsize=(14, 8))
rects1 = ax.bar(x - width/2, list1, width, label='List 1')
rects2 = ax.bar(x + width/2, list2, width, label='List 2')

# 添加一些文本标签
ax.set_xlabel('Features')
ax.set_ylabel('Values')
ax.set_title('Comparison of Two Lists')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend()

# 自动标记柱状图上的数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()
