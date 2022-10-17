import pandas as pd
import numpy as np
from math import sqrt, ceil, floor
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform



'''
    Density-based Method
'''
# Reachdist function
def reachdist(distance_df, observation, index):
    return distance_df[observation][index]


# LOF algorithm implementation from scratch
def LOF_algorithm(data_input, k, distance_metric="cityblock", p=5):
    distances = pdist(data_input.values, metric=distance_metric)
    dist_matrix = squareform(distances)
    distance_df = pd.DataFrame(dist_matrix)

    observations = distance_df.columns
    lrd_dict = {}
    n_dist_index = {}
    reach_array_dict = {}

    for observation in observations:
        dist = distance_df[observation].nsmallest(k + 1).iloc[k]
        indexes = distance_df[distance_df[observation] <= dist].drop(observation).index
        n_dist_index[observation] = indexes

        reach_dist_array = []
        for index in indexes:
            # make a function reachdist(observation, index)
            dist_between_observation_and_index = reachdist(distance_df, observation, index)
            dist_index = distance_df[index].nsmallest(k + 1).iloc[k]
            reach_dist = max(dist_index, dist_between_observation_and_index)
            reach_dist_array.append(reach_dist)
        lrd_observation = len(indexes) / sum(reach_dist_array)
        reach_array_dict[observation] = reach_dist_array
        lrd_dict[observation] = lrd_observation

    # Calculate LOF
    LOF_dict = {}
    for observation in observations:
        lrd_array = []
        for index in n_dist_index[observation]:
            lrd_array.append(lrd_dict[index])
        LOF = sum(lrd_array) * sum(reach_array_dict[observation]) / np.square(len(n_dist_index[observation]))
        LOF_dict[observation] = LOF

    return sorted(LOF_dict.items(), key=lambda x: x[1], reverse=True)[:p]



'''
    
    
    
    Grid-based Method
    
    
    
    
'''


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
        self.grid = [[[] for _ in range(ceil(1 / self.edge_length))]
                     for _ in range(ceil(1 / self.edge_length))]
        self.outliers = []

    def assign_points(self):
        for data_point in self.data:
            row_idx = floor(data_point[0] / self.edge_length)
            col_idx = floor(data_point[1] / self.edge_length)
            self.grid[row_idx][col_idx].append(data_point)

    def assign_single_point(self, point):
        point = np.array(point)
        assert np.array(point).shape[-1] == 2, "Yet, only implement grid-based for 2D"
        row_idx = floor(point[0] / self.edge_length)
        col_idx = floor(point[1] / self.edge_length)
        self.grid[row_idx][col_idx].append(point)

    def cells_prune(self):
        count_total_lv1 = 0
        for i in range(len(self.grid)):
            for j in range(len(self.grid)):
                current_idx = [i, j]
                neighbors = self.find_neighbor(current_idx)
                for neighbor in neighbors:
                    count_total_lv1 += self.count_lv1_objects(current_idx, neighbor)
                count_total_lv1 += len(self.grid[i][j])  # ! add number of objects in current cell

                if count_total_lv1 <= self.pi * self.data.shape[0]:
                    self.outliers.extend(self.grid[i][j])

    def _get_edge_length(self):
        return self.r / 2 * sqrt(self.data.shape[-1])

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
                    idx = idx = [min(max(0, cell_idx[0] + row_temp), len(self.grid) - 1),
                                 min(max(0, cell_idx[1] + col_temp), len(self.grid) - 1)]
                    if idx not in neighbors:
                        neighbors.append(idx)
                row_temp += 1
            col_temp += 1
            row_temp = -2
        return neighbors
