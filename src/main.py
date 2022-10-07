from utils import *
from plot import *
import numpy as np
import pandas as pd


normal_data = pd.read_csv('D:/ml/DBP391/archive/ptbdb_abnormal.csv', header=None)
abnormal_data = pd.read_csv('D:/ml/DBP391/archive/ptbdb_normal.csv', header=None)

X = 10
Y = 400

abnormal = abnormal_data.iloc[:X,:]
normal = normal_data.iloc[:Y,:]

data = pd.concat([normal, abnormal], ignore_index=True)

P = int(Y/X) * 2
res = LOF_algorithm(data, 3, p=P)

#Evaluation LOF model
y_true = np.concatenate((np.full((1, Y), True), np.full((1,X), False)), axis=1)
y_pre = np.copy(y_true)

for i in res:
    y_pre[0][i[0]] = not y_true[0][i[0]]

confusionMatrix(y_true[0], y_pre[0])

