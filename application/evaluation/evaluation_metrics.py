import argparse
import numpy as np
import pandas as pd
from fastdtw import fastdtw
import matplotlib.pyplot as plt

def main():
    reference_trajectory_path = "./"
    predicted_trajectory_path = "./"
    calc_metrics(reference_trajectory_path, predicted_trajectory_path)    

def calc_metrics(reference_trajectory_path, predicted_trajectory_path):
    reference_path_df = pd.read_csv(reference_trajectory_path, sep='	', header=None)
    reference_path_df.columns = ['POS_X', 'POS_Y', 'POS_Z', 'Q_W', 'Q_X', 'Q_Y', 'Q_Z']
    reference_path_df = reference_path_df.iloc[1: , :]

    predicted_path_df = pd.read_csv(predicted_trajectory_path, sep='    ', header=None)
    predicted_path_df.columns = ['POS_X', 'POS_Y', 'POS_Z', 'Q_W', 'Q_X', 'Q_Y', 'Q_Z']
    predicted_path_df = predicted_path_df.iloc[1: , :]

    reference_path_df[['POS_X', 'POS_Y', 'POS_Z']] = reference_path_df[['POS_X', 'POS_Y', 'POS_Z']].apply(pd.to_numeric)
    predicted_path_df[['POS_X', 'POS_Y', 'POS_Z']] = predicted_path_df[['POS_X', 'POS_Y', 'POS_Z']].apply(pd.to_numeric)

    reference_path_df['-POS_X'] = reference_path_df['POS_X'].apply(lambda x: x*-1)
    reference_path_df['-POS_Y'] = reference_path_df['POS_Y'].apply(lambda x: x*-1)
    reference_path_df['-POS_Z'] = reference_path_df['POS_Z'].apply(lambda x: x*-1)

    predicted_path_df['-POS_X'] = predicted_path_df['POS_X'].apply(lambda x: x*-1)
    predicted_path_df['-POS_Y'] = predicted_path_df['POS_Y'].apply(lambda x: x*-1)
    predicted_path_df['-POS_Z'] = predicted_path_df['POS_Z'].apply(lambda x: x*-1)

    reference_path = get_locs(reference_path_df)
    predicted_path = get_locs(predicted_path_df)
    
    predicted_path_sr = 1
    success_distance_threshold = 20

    print(f"PL (Predicted path): {calculate_path_length(predicted_path).round(1)}")
    print(f"PL (Reference path): {calculate_path_length(reference_path).round(1)}")
    print(f"SR: {calculate_sr(predicted_path, reference_path, success_distance_threshold)}")
    print(f"OSR: {calculate_oracle_sr(predicted_path, reference_path, success_distance_threshold)}")
    print(f"NE: {calculate_ne(predicted_path, reference_path).round(2)}")
    print(f"nDTW: {calculate_ndtw(predicted_path, reference_path, success_distance_threshold).round(2)}")
    print(f"SDTW: {calculate_sdtw(predicted_path, reference_path, success_distance_threshold).round(2)}")
    print(f"CLS: {calculate_cls(predicted_path, reference_path, success_distance_threshold).round(2)}")
    print(f"SPL: {calculate_spl(predicted_path, predicted_path_sr, reference_path).round(2)}\n") 
    
    plot_paths(predicted_path, reference_path)

# Returns locations in np array
def get_locs(df):
    y_coordinate = df['POS_Y'].iloc[0]
    res = []
    
    for row in df.iterrows():
        res.append(np.array([row[1]['POS_X'], y_coordinate, row[1]['POS_Z']]))
    return np.array(res)

# Returns euclidian distance between point_a and point_b
def euclidian_distance(position_a, position_b) -> float:
    return np.linalg.norm(np.array(position_b) - np.array(position_a), ord=2)

# Returns Success Rate (SR) that indicates the navigation is considered successful 
# if the agent stops within success_distance_threshold of the destination
def calculate_sr(p_path, r_path, success_distance_threshold):
    distance_to_target = euclidian_distance(np.array(p_path[-1]), np.array(r_path[-1]))
    #print(distance_to_target)
    if distance_to_target <= success_distance_threshold:
        return 1.0
    else:
        return 0.0

# Returns Oracle Success Rate (OSR) where one navigation is considered oracle success
# if the distance between the destination and any point on the trajectory is less than success_distance_threshold
def calculate_oracle_sr(p_path, r_path, success_distance_threshold):
    for p in p_path:
        if euclidian_distance(p, r_path[-1]) <= success_distance_threshold:
            #print(p, r_path[-1])
            return 1.0
    return 0.0

# Returns normalized dynamic time warping metric score
def calculate_ndtw(p_path, r_path, success_distance_threshold):
    dtw_distance = fastdtw(p_path, r_path, dist=euclidian_distance)[0]
    nDTW = np.exp(-dtw_distance / (len(r_path) * success_distance_threshold))
    return nDTW

# Returns Success weighted by normalized Dynamic Time Warping 
# where one navigation is considered successful if one is returned
def calculate_sdtw(p_path, r_path, success_distance_threshold):
    return calculate_sr(p_path, r_path, success_distance_threshold) * calculate_ndtw(p_path, r_path, success_distance_threshold)

# Returns path length
def calculate_path_length(path):
    # Compute differences between consecutive points
    diffs = np.diff(path, axis=0)
    
    # Calculate Euclidean distances
    distances = np.sqrt((diffs**2).sum(axis=1))
    
    # Sum up distances to get total path length
    total_length = distances.sum()
    return total_length

# Returns path coverage score that indicates how well the reference path
# is covered by the predicted path
def path_coverage(p_path, r_path, success_distance_threshold):
    coverage = 0.0
    for r_loc in r_path:
        min_distance = float('inf')
        for p_loc in p_path:
            distance = euclidian_distance(p_loc, r_loc)
            if distance < min_distance:
                min_distance = distance
        coverage += np.exp(-min_distance / success_distance_threshold)
    return coverage / len(r_path)

# Returns the expected optimal length score given reference path’s coverage of predicted path
def calculate_epl(p_path, r_path, success_distance_threshold):
    return path_coverage(p_path, r_path, success_distance_threshold) * calculate_path_length(r_path)

# Returns length score of predicted path respect to reference path
def calculate_ls(p_path, r_path, success_distance_threshold):
    return calculate_epl(p_path, r_path, success_distance_threshold) / (calculate_epl(p_path, r_path, success_distance_threshold) + np.abs(calculate_epl(p_path, r_path, success_distance_threshold) - calculate_path_length(p_path)))

# Returns Coverage weighted by Length Score (CLS) indicates
# how closely predicted path conforms with the entire reference path 
def calculate_cls(p_path, r_path, success_distance_threshold):
    return path_coverage(p_path, r_path, success_distance_threshold) * calculate_ls(p_path, r_path, success_distance_threshold)

def calculate_ne(predicted_path, reference_path):
    return euclidian_distance(np.array(predicted_path[-1]), np.array(reference_path[-1])).round(2)

def calculate_spl(p_path, p_sr, r_path):
    p_path_length = calculate_path_length(p_path)
    r_path_length = calculate_path_length(r_path)

    return p_sr * (r_path_length / (max(p_path_length, r_path_length)))

# Plots the reference and predicted paths
def plot_paths(p_path, r_path):
    plt.plot(r_path[:, 0], r_path[:, 2], 'red')
    plt.plot(p_path[:, 0], p_path[:, 2], 'blue')

    plt.xlabel('X')
    plt.ylabel('Z')
    plt.legend (["Reference Path", "Predicted Path"])
    plt.show()

# Plots the reference and predicted paths
def plot_paths_temp(p_path, r_path):
    ax = plt.axes(projection='3d')
    ax.plot3D(r_path[:, 0], r_path[:, 1], r_path[:, 2], 'red')
    ax.plot3D(p_path[:, 0], p_path[:, 1], p_path[:, 2], 'blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend (["Reference Path", "Predicted Path"])
    plt.show()


if __name__ == '__main__':
    main()
