#  Sklearn 实现 GBDT 可视化

import numpy as np
import pydotplus
from sklearn.ensemble import GradientBoostingRegressor

X = np.arange(1, 11).reshape(-1, 1)
y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])

gbdt = GradientBoostingRegressor(max_depth=4, criterion='mse').fit(X, y)
import os
os.environ["PATH"] += os.pathsep + 'D:/graphviz/bin'

from IPython.display import Image
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

# 拟合训练6棵树
sub_tree = gbdt.estimators_[5, 0]
dot_data = export_graphviz(sub_tree, out_file=None, filled=True, rounded=True, special_characters=True, precision=2)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_pdf("img.pdf")