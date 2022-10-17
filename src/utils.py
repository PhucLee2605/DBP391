from math import sqrt, ceil, floor
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from ProximityBased import *


def LOF(pathnor, pathab):
    normal_data = pd.read_csv(pathnor, header=None)
    abnormal_data = pd.read_csv(pathab, header=None)

    # Choose randomly 10 samples from abnormal and 4000 samples from normal
    X = 10
    Y = 4000

    abnormal = abnormal_data.sample(n=X).reset_index(drop=True)
    normal = normal_data.sample(n=Y).reset_index(drop=True)

    data = pd.concat([normal, abnormal], ignore_index=True)

    #LOF_algorithm
    P = int(Y / X)
    res = LOF_algorithm(data, 4, p=P)

    # y_true = np.concatenate((np.full((1, Y), True), np.full((1,X), False)), axis=1)
    # y_pre = np.copy(y_true)
    # #
    # print(len([i for i in res if i[0] >= Y]))
    # for i in res:
    #     y_pre[0][i[0]] = not y_true[0][i[0]]
    #
    #
    # confusionMatrix(y_true[0], y_pre[0])

    # Evaluation LOF model
    P = int(Y / X)
    true_pos = len([i for i in LOF_algorithm(data, 4, p=P) if i[0] >= Y])
    false_pos = P - true_pos

    false_neg = X - true_pos
    true_neg = X + Y - false_neg - true_pos - false_pos

    print(np.array([[true_pos, false_pos], [false_neg, true_neg]]))

    recall = true_pos / (true_pos + false_neg) * 100
    precision = true_pos / (true_pos + false_pos) * 100

    f1_score = 2 * (recall * precision) / (recall + precision)

    print(f"recall : {recall}\nprecision : {precision}\nf1 score : {f1_score}")
    return res




