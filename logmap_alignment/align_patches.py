import os
import numpy as np
import torch
from sklearn.cluster import DBSCAN
import time
import functools
import multiprocessing
from multiprocessing import RawArray

# Define BASE_DIR and ROOT_DIR
if '_file_' in globals():
    BASE_DIR = os.path.dirname(os.path.abspath(_file_))
else:
    BASE_DIR = os.getcwd()
ROOT_DIR = os.path.dirname(BASE_DIR)

def ICP(X, Y):
    n_pc_points = X.size(1)
    mu_x = torch.mean(X, dim=1)
    mu_y = torch.mean(Y, dim=1)

    centered_x = (X - mu_x.unsqueeze(1)).unsqueeze(2)
    centered_y = (Y - mu_y.unsqueeze(1)).unsqueeze(2).transpose(1, 2)

    C = torch.matmul(centered_y, centered_x).sum(dim=0)
    u, s, v = torch.linalg.svd(C)
    R_opt = torch.matmul(u, v.transpose(0, 1))
    t_opt = mu_y - torch.matmul(R_opt, mu_x)

    opt_labels = torch.matmul(R_opt.unsqueeze(1), X.unsqueeze(-1)).squeeze(-1) + t_opt.unsqueeze(1)
    return opt_labels

def align_current_neighborhood(center_map_indices, neighbor_map_indices, center_map, neighbor_map):
    intersection = list(set(center_map_indices).intersection(set(neighbor_map_indices)))
    aligned_neighborhood = None
    target_idx = None

    if len(intersection) > 10:
        target_idx = torch.tensor([torch.where(center_map_indices == i)[0].item() for i in intersection])
        source_idx = torch.tensor([torch.where(neighbor_map_indices == i)[0].item() for i in intersection])

        source = neighbor_map[source_idx]
        target = center_map[target_idx]

        aligned_neighborhood = ICP(source.unsqueeze(0), target.unsqueeze(0)).squeeze(0)

    return aligned_neighborhood, target_idx

def correct_point(points, appearance, point_weights):
    if appearance > 0:
        corrected_point = torch.zeros(3)
        epsilons = [0.6 * 0.05, 0.6 * 0.12, 0.6 * 0.15, 0.6 * 0.2]
        cleaned_points = []
        attempt_number = 0

        while len(cleaned_points) <= 0 and attempt_number < 3:
            clustering = DBSCAN(eps=epsilons[attempt_number], min_samples=5).fit(points.numpy(), sample_weight=point_weights.numpy())
            cleaned_points = points[clustering.labels_ == 0]
            tmp_point_weights = point_weights[clustering.labels_ == 0]
            attempt_number += 1

        if len(cleaned_points) > 0:
            corrected_point[:2] = torch.tensor(np.average(cleaned_points[:, :2].numpy(), axis=0, weights=tmp_point_weights.numpy()))
        else:
            corrected_point = torch.tensor([0.5, 0.5, 0.0])
    else:
        corrected_point = torch.tensor([0.5, 0.5, 0.0])

    return corrected_point

def align_patch(predicted_map, predicted_neighborhood_indices, center_point):
    center_map = predicted_map[center_point]
    center_map_indices = predicted_neighborhood_indices[center_point]
    center_neighbors = predicted_neighborhood_indices[center_point][1:]

    neighbors_maps = predicted_map[center_neighbors]
    neighbors_maps_indices = predicted_neighborhood_indices[center_neighbors]

    aligned_neighborhoods = torch.zeros(len(center_neighbors), len(center_map), 3)
    aligned_neighborhood_weight = torch.zeros(len(center_neighbors), len(center_map))

    for neighbor in range(n_nearest_neighbors):
        aligned_neighborhood, target_idx = align_current_neighborhood(
            center_map_indices, neighbors_maps_indices[neighbor], center_map, neighbors_maps[neighbor]
        )
        if aligned_neighborhood is not None:
            aligned_neighborhood_weight[neighbor, target_idx] = 1
            aligned_neighborhoods[neighbor, target_idx] = aligned_neighborhood

    aligned_neighborhoods = torch.cat([center_map.unsqueeze(0), aligned_neighborhoods], dim=0)
    aligned_neighborhood_weight = torch.cat([torch.ones(1, len(center_map)), aligned_neighborhood_weight], dim=0)

    distance_to_center_weight = torch.maximum(torch.tensor(0.2), torch.pow(1 - torch.norm(center_map, dim=1), 2))
    distance_to_center_weight = distance_to_center_weight.unsqueeze(1).repeat(1, n_nearest_neighbors + 1)
    aligned_neighborhood_weight *= distance_to_center_weight

    distance_to_neighbor_center = torch.maximum(torch.tensor(0.2), torch.pow(1 - torch.norm(neighbors_maps, dim=2), 2))
    aligned_neighborhood_weight[1:] *= distance_to_neighbor_center

    n_appearance = torch.sum(aligned_neighborhood_weight, dim=0)
    corrected_patch = torch.zeros(len(center_map), 3)

    for point in range(len(center_map)):
        points = aligned_neighborhoods[:, point][aligned_neighborhood_weight[:, point] > 0.1]
        point_weights = aligned_neighborhood_weight[:, point][aligned_neighborhood_weight[:, point] > 0.1]
        corrected_point = correct_point(points, n_appearance[point], point_weights)
        corrected_patch[point] = corrected_point

    return corrected_patch

def main(file):
    predicted_map = torch.from_numpy(np.load(os.path.join(raw_predictions, f"predicted_map_{file}.npy"))).double()
    points = torch.from_numpy(np.loadtxt(os.path.join(in_path, f"{file}.xyz"))).double()

    corrected_maps = torch.zeros_like(predicted_map)
    n_points = len(predicted_map)
    predicted_neighborhood_indices = torch.from_numpy(np.load(os.path.join(raw_predictions, f"predicted_neighborhood_indices_{file}.npy"))).long()

    def align_patch_func(shared_predicted_map, shared_predicted_neighborhood_indices, i):
        return align_patch(shared_predicted_map, shared_predicted_neighborhood_indices, i)

    print('Start:', file)
    with multiprocessing.Pool(64) as pool:
        corrected_maps = pool.map(
            functools.partial(align_patch_func, predicted_map, predicted_neighborhood_indices), range(n_points)
        )
        corrected_maps = torch.stack(corrected_maps)
        np.save(os.path.join(res_path, f'corrected_maps_{file}.npy'), corrected_maps.numpy())

if _name_ == '_main_':
    print('Patches alignment: this step took around 20 sec per shape of 10k points on our machine')
    in_path = os.path.join(ROOT_DIR, 'data/test_data')
    raw_predictions = os.path.join(ROOT_DIR, 'data/test_data/raw_prediction')
    res_path = os.path.join(ROOT_DIR, 'data/test_data/aligned_prediction')

    os.makedirs(res_path, exist_ok=True)

    files = os.listdir(in_path)
    files = [x.replace('.xyz', '') for x in files if x.endswith('.xyz')]
    n_nearest_neighbors = 30

    for file in files:
        main(file)