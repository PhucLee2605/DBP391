import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform


# Reachdist function
def reachdist(distance_df, observation, index):
    return distance_df[observation][index]


# LOF algorithm implementation from scratch
def LOF_algorithm(data_input, k, distance_metric="cityblock", p=5):
    distances = pdist(data_input.values, metric=distance_metric)
    dist_matrix = squareform(distances)
    distance_df = pd.DataFrame(dist_matrix)

    k = k if distance_metric == "cityblock" else k
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