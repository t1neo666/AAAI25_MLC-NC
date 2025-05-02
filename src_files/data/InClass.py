import os

import pandas as pd
import torch
import random
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import SVG, display
longtail = np.load('C:\\Users\\Y\\Desktop\\Bcal\\BCaL\\appendix\\VOCdevkit\\longtail2012\\class_freq.pkl',allow_pickle=True)
longtail_labels = longtail['gt_labels']
longtail_labels = np.array(longtail_labels)
pos_label = longtail_labels.sum(axis=0)
neg_label = longtail_labels.shape[0] - pos_label
indices = np.arange(20)
indices_str = np.array([str(x) for x in indices])


plt.figure(figsize=(15, 7))
total_width, n = 0.8, 2
width = total_width/2
x1 = indices - width/2
x2 = x1 + width
plt.title('VOC2012 longtail')
plt.bar(x1, pos_label, width=width, label='positive')
plt.bar(x2, neg_label, width=width, label='negative')
plt.xticks(indices, indices_str)
plt.xlabel('class')
plt.ylabel('num')
plt.legend()
plt.savefig("vocinclass.svg", dpi=300, format="svg")
plt.show()