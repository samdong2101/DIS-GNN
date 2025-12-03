import yaml
from dis_gnn.model.gnn import GNN 
from dis_gnn.model.downstream.classifier import Classifier 
from dis_gnn.data.featurization import GraphFeaturizer
from dis_gnn.data.dataloader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import math
import time
from torch_scatter import scatter_mean
import argparse
import pickle
from tqdm import tqdm
import numpy as np 
from dis_gnn.utils.plotter import Plotter
from dis_gnn.utils.logger import logger 
from datetime import datetime


def load_data(structures_path = None, df_path = None, batch_size = 32, property_name = 'band_gap', composition = None, cutoff = 4.0, save_path = './dis_gnn/data/data'):
    if df_path is None:
        with open(structures_path,'rb') as f:
            structures = pickle.load(f)
        gf = GraphFeaturizer(structures, cutoff, property_name, composition = composition, save_path = save_path)
        df = gf.featurize()
        dl = DataLoader(df, batch_size = batch_size)
        batched_node_features, batched_edge_indices, batched_edge_features, batched_labels, batched_node_indices, batched_group_sizes = dl.get_data()
            
    else:
        with open(df_path, 'rb') as f:
            df = pickle.load(f)
        dl = DataLoader(df, batch_size = batch_size)
        batched_node_features, batched_edge_indices, batched_edge_features, batched_labels, batched_node_indices, batched_group_sizes = dl.get_data()
    
    return batched_node_features, batched_edge_indices, batched_edge_features, batched_labels, batched_node_indices, batched_group_sizes
            


def train(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
      # load config file
    model_cfg = cfg["model"]
    checkpoint_path = model_cfg["checkpoint_path"]
    batch_size = model_cfg["batch_size"]
    num_epochs = model_cfg["num_epochs"]
    criterion = model_cfg["criterion"]
    optimizer = model_cfg["optimizer"]
    n_hid = model_cfg["n_hid"]
    lr = model_cfg["lr"]
    test_val_split = model_cfg["test_val_split"]
    num = model_cfg['test_val_split']
    gnn_cfg = model_cfg["gnn"]
    gnn_n_layers = gnn_cfg["n_layers"]
    gnn_dropout = gnn_cfg["dropout"]
    gnn_n_resid_layers = gnn_cfg["n_resid_layers"]
    gnn_n_mlp_layers = gnn_cfg["n_mlp_layers"]
    clf_cfg = model_cfg["classifier"]
    clf_hidden_dim = clf_cfg["hidden_dim"]
    dm_cfg = cfg["data_module"]
    df_path = dm_cfg["df_path"]
    structures_path = dm_cfg["structures_path"]
    composition = dm_cfg["composition"]
    data = load_data(df_path = df_path, structures_path = structures_path, batch_size = batch_size, 
        composition = composition, property_name = dm_cfg['property_name'], save_path = dm_cfg["save_path"], cutoff = dm_cfg["cutoff"])
    batched_node_features, batched_edge_indices, batched_edge_features, batched_target_values, batched_node_indices, batched_group_sizes = data[0], data[1], data[2], data[3], data[4], data[5]
    in_dim = batched_node_features[0].shape[1]
    num = int(num*len(batched_node_features))
    num_val = len(batched_node_features) - num
    stamp = str(datetime.now()).replace(' ','_')

    if criterion == 'bce':
        criterion = nn.BCELoss()
      


    device = "cuda" 
    gnn = GNN(in_dim, n_hid, gnn_n_layers, batch_size, dropout=gnn_dropout, n_resid_layers=gnn_n_resid_layers, n_mlp_layers = gnn_n_mlp_layers).to(device)
    classifier = Classifier(n_hid, hidden_dim=clf_hidden_dim)
    model = nn.Sequential(gnn, classifier).to(device)
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
      # stats = 'NEW LOG'
      # description = f'changing {variable_name} FROM {old_variable} TO {variable} --> 0 epoch'
      # knob_logger('/blue/hennig/sam.dong/disordered_classifier/logger_4_b.txt',stats, description, add = True)
      #for name, param in model.named_parameters():
      #      print(f"{name}: requires_grad={param.requires_grad}") 
    training_accuracy = []
    training_losses = []
    validation_accuracy = []
    validation_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_predictions_inner = []
        train_labels_inner = []
        train_losses_inner = []
        #description = f'changing {variable_name} FROM {old_variable} TO {variable} --> {epoch} epoch'
        for i in tqdm(range(num), desc = f'Epoch {epoch+1}/{num_epochs}'):
            optimizer.zero_grad(set_to_none=True)
            
                  # --- Move only current batch to GPU ---
            flattened_node_features, flattened_edge_indices, flattened_edge_features = batched_node_features[i].to(device), batched_edge_indices[i].to(device), torch.tensor(batched_edge_features[i], dtype=torch.float32, device=device)
            y_labels = batched_target_values[i].to(dtype=torch.float32, device=device).reshape(batch_size, 1)
                  
            #state = torch.repeat_interleave(batched_target_values[i], batched_group_sizes[i]).to(dtype=torch.float32, device = device) #batched_target_values[i].to(dtype=torch.float32, device=device)
            #print(state.shape)
            group_sizes = batched_group_sizes[i].to(device = device)
            state = batched_target_values[i].to(dtype=torch.float32, device=device)
            output = gnn(flattened_node_features, flattened_edge_indices, flattened_edge_features, state, group_sizes)
            pooled_output = gnn.pool(output, batched_node_indices[i])
            pred = classifier(pooled_output)
            loss = criterion(pred, y_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

                  
            train_losses_inner.append(loss.item())
            train_predictions_inner.append(pred.detach().cpu())
            train_labels_inner.append(y_labels.detach().cpu())
            #scheduler.step()
            # --- Detach and move to CPU to save GPU memory ---
            # Free memory for this batch
            del output, pooled_output, pred, y_labels
        torch.cuda.empty_cache()
        train_all_preds = torch.cat(train_predictions_inner, dim=0)
        train_all_labels = torch.cat(train_labels_inner, dim=0)
        train_preds_binary = (train_all_preds > 0.5).float()
        train_acc = (train_preds_binary == train_all_labels).float().mean().item()
        print('[Training loss]:', np.mean(train_losses_inner))
        print('[Training accuracy]:', train_acc)
        logger('./logs/',np.mean(train_losses_inner), stamp, desc = 'training_loss')
        logger('./logs/',train_acc, stamp, desc = 'training_accuracy')
        training_losses.append(np.mean(train_losses_inner))
        training_accuracy.append(train_acc)
        # --- Validation ---
            
        model.eval()
        validation_losses_inner = []
        validation_predictions_inner = []
        validation_labels_inner = []
        with torch.no_grad():
            for i in range(num_val):
                flattened_node_features_val, flattened_edge_indices_val, flattened_edge_features_val = batched_node_features[num + i].to(device), batched_edge_indices[num + i].to(device), torch.tensor(batched_edge_features[num + i], dtype=torch.float32, device=device)
                y_labels_val = batched_target_values[num + i].to(dtype=torch.float32, device=device).reshape(batch_size, 1)
                        #state_val = torch.repeat_interleave(batched_target_values[num + i], batched_group_sizes[num + i]).to(dtype=torch.float32, device = device)
                state_val = batched_target_values[num+i].to(dtype=torch.float32, device=device)
                group_sizes_val = batched_group_sizes[num + i].to(device = device)
                output_val = gnn(flattened_node_features_val, flattened_edge_indices_val, flattened_edge_features_val, state_val, group_sizes_val)
                output_val = gnn.pool(output_val, batched_node_indices[num + i])
                pred_val = classifier(output_val)
                loss_val = criterion(pred_val, y_labels_val)
                validation_losses_inner.append(loss_val.item())
                validation_predictions_inner.append(pred_val.detach().cpu())
                validation_labels_inner.append(y_labels_val.detach().cpu())
      
                del output_val, pred_val, y_labels_val
            torch.cuda.empty_cache()
            val_all_preds = torch.cat(validation_predictions_inner, dim=0)
            val_all_labels = torch.cat(validation_labels_inner, dim=0)
            val_preds_binary = (val_all_preds > 0.5).float()
            #validation_losses_outer.append(validation_losses_inner)
            val_acc = (val_preds_binary == val_all_labels).float().mean().item()
            #val_accuracies.append(val_acc)
            print('[Validation loss]:', np.mean(validation_losses_inner))
            print('[Validation accuracy]:', val_acc)
            logger('./logs/',np.mean(validation_losses_inner), stamp, desc = 'validation_loss')
            logger('./logs/',val_acc, stamp, desc = 'validation_accuracy')
            validation_losses.append(np.mean(validation_losses_inner))
            validation_accuracy.append(np.mean(val_acc))
    
    return training_losses, training_accuracy, validation_losses, validation_accuracy      

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="Path to YAML config file"
    )
    parser.add_argument("--save_plot",
              type=str,
              required=False,
              help="directory to save parity plots"
              )
    args = parser.parse_args()
    #data = load_data(df_path = '/blue/hennig/sam.dong/disordered_classifier/data/graph_df_4_cutoff_12_atoms_100_mev.pkl') 
    training_losses, training_accuracy, validation_losses, validation_accuracy = train(args.config)
    plotter = Plotter(args.save_plot)
    plotter.plot_losses(training_losses, validation_losses) 
    plotter.plot_accuracies(training_accuracy, validation_accuracy)

if __name__ == "__main__":
    main()
