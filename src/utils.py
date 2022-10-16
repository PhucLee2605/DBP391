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


def pca_reduce(data):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    return reduced_data


def normalize(data):
    min_max_scaler = MinMaxScaler()
    data_normalized = min_max_scaler.fit_transform(data)
    return data_normalized


class GridBased:
    def __init__(self, data, r, pi):
        self.data = pca_reduce(data)
        self.data = normalize(self.data)
        self.r = r
        self.pi = pi
        self.edge_length = self._get_edge_length()
        self.grid = [[[] for _ in range(ceil(1/self.edge_length))]
                     for _ in range(ceil(1/self.edge_length))]
        self.outliers = []


    def assign_points(self):
        for data_point in self.data:
            row_idx = floor(data_point[0]/self.edge_length)
            col_idx = floor(data_point[1]/self.edge_length)
            self.grid[row_idx][col_idx].append(data_point)


    def assign_single_point(self, point):
        point = np.array(point)
        assert np.array(point).shape[-1] == 2, "Yet, only implement grid-based for 2D"
        row_idx = floor(point[0]/self.edge_length)
        col_idx = floor(point[1]/self.edge_length)
        self.grid[row_idx][col_idx].append(point)


    def cells_prune(self):
        count_total_lv1 = 0
        for i in range(len(self.grid)):
            for j in range(len(self.grid)):
                current_idx = [i,j]
                neighbors = self.find_neighbor(current_idx)
                for neighbor in neighbors:
                    count_total_lv1 += self.count_lv1_objects(current_idx,neighbor)
                count_total_lv1 += len(self.grid[i][j]) #! add number of objects in current cell

                if count_total_lv1 <= self.pi*self.data.shape[0]:
                    self.outliers.extend(self.grid[i][j])


    def _get_edge_length(self):
        return self.r/2*sqrt(self.data.shape[-1])


    def count_lv1_objects(self, current_cell_idx, cell_to_compare):
        count_no_data_lv1 = 0
        # print("current",current_cell_idx[0],current_cell_idx[1])
        # print("compare",cell_to_compare[0],cell_to_compare[1], end = "\n\n")
        for data_ in self.grid[cell_to_compare[0]][cell_to_compare[1]]:
            for data in self.grid[current_cell_idx[0]][current_cell_idx[1]]:
                if np.linalg.norm(data - data_) > self.r:
                    return 0
            count_no_data_lv1 += 1
        
        return count_no_data_lv1


    def find_neighbor(self, cell_idx):
        neighbors = []
        row_temp = -2
        col_temp = -2
        for _ in range(5):
            for _ in range(5):
                if not (row_temp == 0 and col_temp == 0):
                    idx = idx = [min(max(0, cell_idx[0] + row_temp), len(self.grid)-1),
                                 min(max(0, cell_idx[1] + col_temp), len(self.grid)-1)]
                    if idx not in neighbors:
                        neighbors.append(idx)
                row_temp += 1
            col_temp += 1
            row_temp = -2
        return neighbors