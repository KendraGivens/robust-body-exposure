#%%
import os
from bm_dataset import BMDataset
from gnn_manager import GNN_Manager
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_seeds', type=int, default=100)
args = parser.parse_args()

dataset_dir =  f'./DATASETS/Recover_Data/TL_2, 4, 5, 8, 10, 11, 12, 13, 14, 15_Recover_Data_100_seeds_10000_states_lowerbody'
datasets = [0]
recover = True

not_subsampled = BMDataset(
        recover=recover,
        root=dataset_dir,
        description='Recovering_Standard_Dataset_Not_Subsampled',
        voxel_size=np.nan, edge_threshold=0.06,
        rot_draping=True)
#%%
model_names = [
        f'tl_2, 4, 5, 8, 10, 11, 12, 13, 14, 15_{args.num_seeds}_states_1k_lowerbody'
        ]

dataset_sizes = [16000]
datasets = [not_subsampled]*len(dataset_sizes)

for i in range(len(datasets)):
        initial_dataset = datasets[i]
        initial_dataset = initial_dataset[:dataset_sizes[i]+100]

        print(datasets[i])
        torch.cuda.empty_cache()
        # device = 'cuda:0'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gnn_manager = GNN_Manager(device)
        gnn_manager.set_initial_dataset(initial_dataset, (dataset_sizes[i], 100))
        save_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'trained_models/FINAL_MODELS'))
        model_description = f'{model_names[i]}_{dataset_sizes[i]}'
        train_test = (dataset_sizes[i], .1*dataset_sizes[i])
        num_images = 100
        epochs = 250
        proc_layers = 4
        learning_rate = 1e-4
        seed = 1001
        # batch_size = 100
        batch_size = 50
        num_workers = 4
        use_displacement = True

        gnn_manager.initialize_new_model(save_dir, train_test,
                proc_layers, num_images, epochs, learning_rate, seed, batch_size, num_workers,
                model_description, use_displacement)
        gnn_manager.set_dataloaders()
        gnn_manager.train(epochs)
# %%
