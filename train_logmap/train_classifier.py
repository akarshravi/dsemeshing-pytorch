import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pointnet_seg import classification_net
import sys
from torch_dataset import CustomDataset
from scipy.spatial.transform import Rotation as R
gpu=0


BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
def safe_norm(x, epsilon=1e-8, axis=-1):
    return torch.sqrt(torch.clamp(torch.sum(x**2, dim=axis), min=epsilon))

BATCH_SIZE = 64
RESIZE=True
N_ORIG_NEIGHBORS = 200
N_NEIGHBORS_DATASET = 120
N_NEAREST_NEIGHBORS = 30
N_NEIGHBORS = 120
TESTING_SHAPES = [21, 11, 26]
#TESTING_SHAPES = [1,2]
TRAINING_SHAPES = list(set(list(range(56))) - set(TESTING_SHAPES))
#TRAINING_SHAPES = list(set(list(range(6))) - set(TESTING_SHAPES))
N_TRAINING_SHAPES = len(TRAINING_SHAPES)
N_TESTING_SHAPES = len(TESTING_SHAPES)
LOG_DIR = "../log/log_famousthingi_classifier"
n_patches = 10000
path_records = "../data/numpy_training_data/famousthingi_logmap_patches_{}.npy"


TRAINSET_SIZE = n_patches*N_TRAINING_SHAPES
VALSET_SIZE = n_patches*N_TESTING_SHAPES


def collate_fn_classifier(batch):
    batch=torch.stack(batch)
    rotations = torch.tensor(R.random(BATCH_SIZE).as_matrix(), dtype=torch.float32)
    batch = batch[:, :N_ORIG_NEIGHBORS]
    batch = batch[:, :N_NEIGHBORS]
    neighbor_points = batch[:, :N_NEIGHBORS, :3]
    gt_map = batch[:, :N_NEIGHBORS, 3:]
    # Subtract the first neighbor point (broadcasting is automatic in PyTorch)
    neighbor_points = neighbor_points - neighbor_points[:, 0:1, :].expand(-1, N_NEIGHBORS, -1)
    # Perform the matrix multiplication with rotations
    neighbor_points = torch.bmm(rotations, neighbor_points.transpose(1, 2)).transpose(1, 2)

    if RESIZE:
        # Compute the diagonal (safe_norm equivalent)
        diag = torch.norm(torch.max(neighbor_points, dim=1).values - torch.min(neighbor_points, dim=1).values, dim=-1)

        # Expand the diagonal to match neighbor_points shape
        diag = diag.view(-1, 1, 1).expand(-1, neighbor_points.size(1), neighbor_points.size(2))

        # Divide neighbor_points by the diagonal
        neighbor_points = neighbor_points / diag

        # Divide gt_map by the diagonal (only consider the first 2 dimensions)
        gt_map = gt_map / diag[:, :, :2]
        return neighbor_points,gt_map    

def get_model_predictions(model, batch, device):
    neighbor_points,gt_map=batch
    neighbor_points,gt_map=neighbor_points.to(device),gt_map.to(device)
    map = model(neighbor_points)  # Predicted output
    map = map.squeeze()  # Equivalent to tf.squeeze

    # Ground truth distances
    gt_dists = torch.norm(gt_map, dim=-1)

    # Compute nearest neighbors for ground truth
    _, geo_neighbors = torch.topk(-gt_dists, k=N_NEAREST_NEIGHBORS, dim=1)
    geo_neighbors_indices = torch.stack(
        [
            torch.arange(BATCH_SIZE, device=device).unsqueeze(1).repeat(1, N_NEAREST_NEIGHBORS),
            geo_neighbors,
        ],
        dim=2,
    )
    geo_neighbors_indices = geo_neighbors_indices.view(-1, 2)

    # Create label tensor
    labels = torch.zeros_like(map, dtype=torch.float32)
    labels[geo_neighbors_indices[:, 0], geo_neighbors_indices[:, 1]] = 1.0

    # Compute classification loss
    class_loss = torch.mean((labels - torch.sigmoid(map)) ** 2) * 30.0

    # Predicted nearest neighbors
    _, predicted_geo_neighbors = torch.topk(torch.sigmoid(map), k=N_NEAREST_NEIGHBORS, dim=1)
    predicted_geo_neighbors_indices = torch.stack(
        [
            torch.arange(BATCH_SIZE, device=device).unsqueeze(1).repeat(1, N_NEAREST_NEIGHBORS),
            predicted_geo_neighbors,
        ],
        dim=2,
    )
    predicted_geo_neighbors_indices = predicted_geo_neighbors_indices.view(-1, 2)

    # Create predicted labels tensor
    predicted_labels = torch.zeros_like(map, dtype=torch.float32)
    predicted_labels[
        predicted_geo_neighbors_indices[:, 0], predicted_geo_neighbors_indices[:, 1]
    ] = 1.0

    # Compute accuracy
    accuracy = 1 - torch.mean(torch.abs(labels - predicted_labels))

    # Distance-based loss
    loss_dist = torch.mean((map - gt_dists) ** 2) * 5.0

    # Total loss
    loss = class_loss  # You can add loss_dist if needed
    return loss,accuracy

def train_one_epoch(test_loader,model,optimizer,device,epoch):
    model.train()
    l_loss= []
    l_acc= []
    n_steps=len(test_loader)
    for step,batch in enumerate(test_loader):
        loss,acc=get_model_predictions(model, batch, device)
        
        # Backpropagation and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        l_loss.append(loss.item())
        l_acc.append(acc.item())
        # Print metrics
        #print(f"Loss: {loss.item()}, Accuracy: {accuracy.item()}")
        if step%200 == 0:
            print("epoch {:4d} \t step {:5d}/{:5d} \t loss: {:3.4f} accuracy: {:3.2f}%".format(epoch, step,n_steps, np.mean(l_loss), np.mean(l_acc)*100))
            l_loss = []
            l_acc = []
        



def eval_one_epoch(model, dataloader, device, epoch):
    model.eval()  # Set model to evaluation mode
    l_loss = []
    l_acc = []

    print('Validation:')
    
    n_steps = len(dataloader)
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            #batch=batch.to(device)
            #rotations = torch.eye(3).view(1, 3, 3).repeat(BATCH_SIZE, 1, 1).to(device)
            loss,acc=get_model_predictions(model, batch, device)           
            l_loss.append(loss.item())
            l_acc.append(acc.item())
            
            if step == n_steps - 1:
                print(f"Validation epoch {epoch:4d} step {step:5d}/{n_steps:5d} "
                      f"loss: {np.mean(l_loss):.4f} accuracy: {np.mean(l_acc) * 100:.2f}%")
    
    avg_loss = np.mean(l_loss)
    return avg_loss



if __name__ == '__main__':
    

    # Example usage
    filenames = [path_records.format(i) for i in range(0,56)]  
    #filenames = [path_records.format(i) for i in range(0,6)]
    training_data = CustomDataset([filenames[i] for i in TRAINING_SHAPES],N_NEIGHBORS,N_ORIG_NEIGHBORS)
    testing_data = CustomDataset([filenames[i] for i in TESTING_SHAPES],N_NEIGHBORS,N_ORIG_NEIGHBORS)
    train_loader = DataLoader(training_data,drop_last=True, batch_size=BATCH_SIZE ,num_workers=8,shuffle=True,collate_fn=collate_fn_classifier)
    test_loader = DataLoader(testing_data, drop_last=True,batch_size=BATCH_SIZE,num_workers=8, shuffle=True,collate_fn=collate_fn_classifier)



    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    lr = 0.0005
    
    best_loss = float("inf")

    device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
    model=classification_net(batch_size=BATCH_SIZE)
    model = nn.DataParallel(model)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_epochs=1000


    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)



    for epoch in range(num_epochs):
        if epoch > 20:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001
        # Train and evaluate


        start_event.record()
        # Code to measure
        train_one_epoch(train_loader,model,optimizer,device,epoch)
        end_event.record()

        torch.cuda.synchronize()  # Wait for GPU to finish
        elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds
        print(f"Epoch {epoch} completed in {elapsed_time / 1000:.2f} seconds.")


        loss_val = eval_one_epoch(model, test_loader, device, epoch)

        # Save the best model
        if loss_val < best_loss:
            torch.save(model.state_dict(), os.path.join(LOG_DIR, "best_model.pth"))
            best_loss = loss_val

        # Save the model at each epoch
        torch.save(model.state_dict(), os.path.join(LOG_DIR, "model.pth"))

