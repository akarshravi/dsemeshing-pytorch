import os
import sys
import trimesh
import torch
import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial import Delaunay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BATCH_SIZE = 128

def init_config():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "allow_growth": True
    }
    return config

def init_graph(X3D, X3D_normals, n_neighbors):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    n_points = X3D.shape[0]

    config = init_config()
    device = config["device"]

    normals = torch.tensor(X3D_normals, dtype=torch.float32, device=device, requires_grad=True)

    corrected_map = torch.zeros((BATCH_SIZE, n_neighbors + 1, 3), dtype=torch.float32, device=device)
    corrected_points_neighbors = torch.zeros((BATCH_SIZE, n_neighbors + 1), dtype=torch.int32, device=device)

    # Placeholder-like structures
    first_index_pl = torch.zeros(BATCH_SIZE, dtype=torch.int32, device=device)

    # Example function call for triangulation (you need to replace this with actual implementation in PyTorch)
    corrected_approx_triangles, corrected_indices = delaunay_tf.get_triangles_geo_batches(
        n_neighbors=n_neighbors,
        gdist=corrected_map,
        gdist_neighbors=corrected_points_neighbors[:, 1:],
        first_index=first_index_pl
    )

    # PyTorch optimizer and parameter initialization
    optimizer = optim.Adam([normals], lr=0.001)

    ops = {
        "corrected_indices": corrected_indices,
        "corrected_approx_triangles": corrected_approx_triangles,
        "first_index_pl": first_index_pl,
        "corrected_map": corrected_map,
        "corrected_points_neighbors": corrected_points_neighbors,
    }
    
    return normals, ops, optimizer
def reconstruct(name):
    logmap_points = np.loadtxt(os.path.join(in_path, name))
    name = name.replace('.xyz', "")
    X3D = logmap_points
    tree = KDTree(logmap_points)
    X3D_normals = np.zeros([X3D.shape[0], 3])
    X3D_normals[:, 2] = 1

    n_points = len(logmap_points)
    normals, ops, optimizer = init_graph(X3D, X3D_normals, n_neighbors)
    points_indices = list(range(n_points))

    predicted_neighborhood_indices = np.load(os.path.join(raw_prediction_path, "predicted_neighborhood_indices_{}.npy".format(name)))
    corrected_predicted_map = np.load(os.path.join(res_path, "corrected_maps_{}.npy".format(name)))

    triangles = []
    indices = []
    for step in range(1 + len(X3D) // BATCH_SIZE):
        if step % 50 == 0:
            print("step: {}/{}".format(step, n_points // BATCH_SIZE))
        if (step + 1) * BATCH_SIZE > n_points:
            current_points = points_indices[-BATCH_SIZE:]
        else:
            current_points = points_indices[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]

        center_points = np.array(X3D[current_points])
        points_neighbors = tree.query(center_points, n_neighbors + 1)[1]

        first_index_pl = torch.tensor(current_points, dtype=torch.int32, device=normals.device)
        corrected_map = torch.tensor(corrected_predicted_map[current_points], dtype=torch.float32, device=normals.device)
        corrected_points_neighbors = torch.tensor(predicted_neighborhood_indices[current_points], dtype=torch.int32, device=normals.device)

        with torch.no_grad():
            corrected_approx_triangles = ops["corrected_approx_triangles"]
            corrected_indices = ops["corrected_indices"]

            # Example of PyTorch computation for corrected triangles and indices
            res_triangles = corrected_approx_triangles  # Replace with actual logic
            res_indices = corrected_indices  # Replace with actual logic

        if (step + 1) * BATCH_SIZE > n_points:
            res_triangles = res_triangles[-n_points % BATCH_SIZE:]
            res_indices = res_indices[-n_points % BATCH_SIZE:]

        triangles.append(res_triangles.cpu().numpy())
        indices.append(res_indices.cpu().numpy())

    indices = np.concatenate(indices)
    triangles = np.concatenate(triangles)

    trigs = np.sort(np.reshape(indices[triangles > 0.5], [-1, 3]), axis=1)
    uni, inverse, count = np.unique(trigs, return_counts=True, axis=0, return_inverse=True)
    triangle_occurence = count[inverse]
    np.save(os.path.join(res_path, "patch_frequency_count_{}.npy".format(name)), np.concatenate([uni, count[:, np.newaxis]], axis=1))


if __name__ == '__main__':
    # Evaluate the new meshes and count frequency of triangles
    in_path = os.path.join(ROOT_DIR, 'data/test_data')
    raw_prediction_path = os.path.join(ROOT_DIR, 'data/test_data/raw_prediction')
    res_path = os.path.join(ROOT_DIR, 'data/test_data/aligned_prediction')
    n_neighbors = 120
    n_nearest_neighbors = 30

    # Evaluate all .xyz files in the in_path directory
    files = os.listdir(in_path)
    files = [x for x in files if x.endswith('.xyz')]

    for name in files:
        reconstruct(name)

