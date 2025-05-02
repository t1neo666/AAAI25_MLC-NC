import matplotlib.pyplot as plt

# 数据
dimensions = [20, 64, 128, 256, 512, 768]
total_map = [84.23, 83.84, 83.25, 82.88, 81.34, 82.08]
head_map = [72.48, 72.1, 70.98, 69.89, 65.74, 69.59]
medium_map = [88.16, 87.5, 87.82, 87.56, 86.23, 87.06]
tail_map = [90.1, 89.91, 89.03, 89.12, 89.38, 87.70]

# 创建图表
fig, ax1 = plt.subplots()

# 绘制柱状图
ax1.bar(range(len(dimensions)), total_map, color='skyblue', label='Total mAP')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('Total mAP', color='black')
ax1.set_ylim(80, 85)
ax1.set_title('mAP vs Dimension')
ax1.set_xticks(range(len(dimensions)))
ax1.set_xticklabels(dimensions)

# 统一右边的 y 轴范围和刻度
right_ylim = (60, 100)
right_yticks = [60, 70, 80, 90, 100]

# 创建第二个 y 轴 (Head mAP)
ax2 = ax1.twinx()
ax2.plot(range(len(dimensions)), head_map, color='orange', marker='o', label='Head mAP')
ax2.set_ylabel('Head / Medium / Tail mAP', color='black')
ax2.set_ylim(right_ylim)
ax2.set_yticks(right_yticks)
ax2.tick_params(axis='y', labelcolor='orange')

# 创建第三个 y 轴 (Medium mAP)
ax3 = ax1.twinx()
ax3.plot(range(len(dimensions)), medium_map, color='green', marker='s', label='Medium mAP')
ax3.set_ylim(right_ylim)
ax3.set_yticks(right_yticks)


# 创建第四个 y 轴 (Tail mAP)
ax4 = ax1.twinx()
ax4.plot(range(len(dimensions)), tail_map, color='red', marker='^', label='Tail mAP')
ax4.set_ylim(right_ylim)
ax4.set_yticks(right_yticks)

# # 添加图例
# fig.tight_layout()  # 调整布局以避免重叠
# ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')
# ax3.legend(loc='lower left')
# ax4.legend(loc='lower right')
#
# # 显示图表
# plt.show()
#
# # 保存图表为PDF文件
# fig.savefig('Total_mAP_vs_Dimension_with_Head_Medium_Tail_mAP_Unified.pdf', format='pdf')
# plt.close()


# 创建自定义图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()

# Total mAP的图例在左上角
ax1.legend(loc='upper left')

# 其他图例在右上角
fig.legend(lines2 + lines3 + lines4, labels2 + labels3 + labels4, loc='upper right', bbox_to_anchor=(0.89, 0.88))

# 调整布局以避免重叠，并为图例留出空间
fig.tight_layout(rect=[0, 0, 1, 0.95])

# 显示图表
plt.show()

# 保存图表为PDF文件
fig.savefig('Total_mAP_vs_Dimension_with_Head_Medium_Tail_mAP_Unified.pdf', format='pdf')
plt.close()