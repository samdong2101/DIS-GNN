import yaml
from dis_gnn.model.gnn import GNN
from dis_gnn.model.downstream.downstream import Classifier, Regressor
from dis_gnn.data.featurization import GraphFeaturizer
from dis_gnn.data.featurization import LineGraphFeaturizer
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
import os
from dis_gnn.data.dataloader import DataLoader, load_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import argparse

class Inference:
    def __init__(self, stamp, df_path = None, ldf_path = None):
        self.stamp = stamp
        self.cfg_path = f'./logs/{stamp}/config.yaml'
        with open(self.cfg_path,"r") as f:
            self.cfg = yaml.safe_load(f)
        self.df_path = df_path
        self.ldf_path = ldf_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def load_config(self):
        cfg = self.cfg
        model_cfg = cfg['model']
        task = model_cfg["task"]
        include_coords = model_cfg['include_coords']
        include_cells = model_cfg['include_cells']
        include_states = model_cfg['include_states']
        checkpoint_path = model_cfg["checkpoint_path"]
        batch_size = model_cfg["batch_size"]
        num_epochs = model_cfg["num_epochs"]
        criterion_name = model_cfg["criterion"]
        optimizer = model_cfg["optimizer"]
        n_hid = model_cfg["n_hid"]
        lr = model_cfg["lr"]
        test_val_split = model_cfg["test_val_split"]
        split = model_cfg['test_val_split']
        gnn_cfg = model_cfg["gnn"]
        gnn_n_layers = gnn_cfg["n_layers"]
        gnn_dropout = gnn_cfg["dropout"]
        gnn_n_resid_layers = gnn_cfg["n_resid_layers"]
        gnn_n_mlp_layers = gnn_cfg["n_mlp_layers"]
        clf_cfg = model_cfg["downstream"]
        clf_hidden_dim = clf_cfg["hidden_dim"]
        dm_cfg = cfg["data_module"]
        df_path = dm_cfg["df_path"]
        ldf_path = dm_cfg["ldf_path"]
        structures_path = dm_cfg["structures_path"]
        composition = dm_cfg["composition"]
        num_atoms = dm_cfg["num_atoms"]
        scope = model_cfg["scope"]
        return model_cfg, task, checkpoint_path, batch_size, num_epochs, criterion_name, optimizer,n_hid, lr, test_val_split, split, gnn_cfg, gnn_n_layers, gnn_dropout, gnn_n_resid_layers, gnn_n_mlp_layers, clf_cfg,clf_hidden_dim, dm_cfg, df_path, ldf_path, structures_path, composition, num_atoms, scope, include_coords, include_cells, include_states

    def load_model(self, node_features, line_node_features, criterion_name, task, optimizer, split, n_hid,
                   gnn_n_layers, batch_size, gnn_dropout, gnn_n_resid_layers, gnn_n_mlp_layers, clf_hidden_dim, lr, include_coords,
                   include_cells, include_states):
        in_dim = node_features[0].shape[1]
        line_in_dim = line_node_features[0].shape[1]
        print('SPLIT:',split)
        num = int(split*len(node_features))
        print('TOTAL DATASIZE:',len(node_features))
        print('TRAIN DATASIZE:',num)
        
        num_val = len(node_features) - num
        print('VAL DATASIZE:',num_val)
        if criterion_name == 'bce':
            criterion = nn.BCELoss()

        else:
            criterion = nn.L1Loss()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gnn = GNN(in_dim, line_in_dim, n_hid, gnn_n_layers, batch_size, dropout=gnn_dropout, n_resid_layers=gnn_n_resid_layers, n_mlp_layers = gnn_n_mlp_layers, include_coords = include_coords, include_cells = include_cells, include_states = include_states).to(device)
        if task == 'classification':
            downstream = Classifier(n_hid, hidden_dim=clf_hidden_dim)
        else:
            downstream = Regressor(n_hid, hidden_dim = clf_hidden_dim)
        model = nn.Sequential(gnn, downstream).to(device)
        if optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr = lr)

        return model, gnn, downstream, criterion, num, num_val

    def get_checkpoint(self):
        path = f'./logs/{self.stamp}/validation_loss.txt'

        with open(path, "r") as f:
            # read numeric values, skip empty lines
            vals = [float(line.strip()) for line in f if line.strip()]

        if not vals:
            return None  # or raise error

        # line numbers are 1-indexed
        min_idx = vals.index(min(vals)) + 1
        #min_idx_small = vals.index(min(vals))
        # nearest multiple of 25 at or above this line
        checkpoint_high = int(math.ceil(min_idx / 25) * 25)
        checkpoint_low = int(math.ceil(min_idx/25)*25)-25
        print('CHECKPOINT HIGH AT:',checkpoint_high)
        print('CHECKPOINT LOW AT:',checkpoint_low) 
        return checkpoint_high, checkpoint_low

    def classification_plotter(self, true, preds):
        true_list = [float(i) for i in true]
        preds_list = [float(i) for i in preds]
        # Ensure data is numpy array for indexing
        y_true = np.array(true_list).astype(float)
        y_score = np.array(preds_list).astype(float)

        # ROC computation
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # 1. Find the Optimal Point (Youden's J-Statistic)
        # J = TPR - FPR. We want to maximize this.
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]

        # 2. Setup for Color Bar (Thresholds)
        # Create segments for the color mapping
        points = np.array([fpr, tpr]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        fig, ax = plt.subplots(figsize=(4, 3))

        # Create the multicolored line
        norm = plt.Normalize(0, 1)
        lc = LineCollection(segments, cmap='Greens', norm=norm)
        lc.set_array(thresholds)
        lc.set_linewidth(7)

        # Plotting
        line = ax.add_collection(lc)
        ax.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.7, label='Random Guess')
        # Turn ticks inward on both axes
        ax.tick_params(axis='both', which='both', direction='in')

        # Format tick labels to two decimal places
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # 3. Annotate the Optimal Point
        #ax.scatter(fpr[best_idx], tpr[best_idx], color='red', s=100, zorder=5,
        #           label=f'Optimal (T={best_threshold:.2f})')

        # --- CHANGES START HERE ---

        # Add AUC as a text box inside the plot
        stats_text = f'AUC = {roc_auc:.3f}'
        ax.text(0.6, 0.2, stats_text, fontsize=8)

        # Formatting
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        #ax.set_title('Receiver Operating Characteristic') # Simplified title

        # --- CHANGES END HERE ---

        # Add Color Bar
        cbar = fig.colorbar(line, ax=ax)
        cbar.set_label('Threshold Value', rotation=90, labelpad=15)
        auc_handle = Line2D([], [], color='none', label=f'AUC = {roc_auc:.3f}')
        ax.legend(handles=[auc_handle], loc='lower right')
        ax.legend(loc='lower right')
        #plt.grid(alpha=0.3)
        plt.tight_layout()
        os.makedirs(f'./figures/{self.stamp}', exist_ok=True)
        plt.savefig(f'./figures/{self.stamp}/ROC_{self.stamp}.png')
        plt.show()

        # inputs
        y_true = np.asarray(true_list, dtype=float)
        y_score = np.asarray(preds_list, dtype=float)

        # compute PR + thresholds
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        # plot
        fig, ax = plt.subplots(figsize=(4, 3))

        sc = ax.scatter(
            recall[:-1],
            precision[:-1],
            c=thresholds,
            cmap="Greens",
            label=f"AP = {ap:.3f}"
        )

        plt.colorbar(sc, ax=ax, label="Decision threshold")

        # axis labels
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")

        # ticks inward
        ax.tick_params(direction="in", which="both")

        # two decimal places on axes
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        ax.legend()
        plt.tight_layout()
        os.makedirs(f'./figures/{self.stamp}', exist_ok=True)
        plt.savefig(f'./figures/{self.stamp}/PR_{self.stamp}.png')
        #plt.savefig(f'./figures/PR_{self.stamp}.pdf')
        plt.show()


        # Inputs
        y_true = np.array([float(i) for i in true])
        y_score = np.array([float(i) for i in preds])

        threshold = 0.5
        y_pred = (y_score >= threshold).astype(int)

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["+", "-"]
        )
        # You can change the color scheme using 'cmap'
        # Some options to try:
        # "Blues"       -> default light-to-dark blue
        # "Greens"      -> light-to-dark green
        # "Reds"        -> light-to-dark red
        # "Purples"     -> light-to-dark purple
        # "coolwarm"    -> blue-to-red gradient
        # "viridis"     -> perceptually uniform
        # "magma"       -> dark-to-light warm tones
        # "cividis"     -> colorblind-friendly
        plt.rcParams['figure.figsize'] = [7, 6]  # width, height

        # Plot with color map
        disp.plot(
            cmap="Greens",          # color scheme
            values_format="d",      # integer format
            colorbar=False          # optional
        )

        # Customize axis labels and ticks
        disp.ax_.set_xlabel("Predicted", fontsize=16)
        disp.ax_.set_ylabel("Target", fontsize=16)
        disp.ax_.tick_params(axis='x', labelsize=20)
        disp.ax_.tick_params(axis='y', labelsize=20)

        # Increase font size of numbers inside squares
        for row in disp.text_:       # disp.text_ is now a 2D array of Text objects
            for text in row:
                text.set_fontsize(40)
                #text.set_weight('bold')

        # Optional title
        #plt.title(f"Confusion Matrix (threshold = {threshold})", fontsize=18)
        plt.tight_layout()
        plt.savefig(f'./figures/confusion_matrix_{self.stamp}.pdf')
        plt.show()

    def regression_plotter(self, true, preds, scope):
        if scope == 'node':
            preds = torch.cat(preds)
            true = torch.cat(true)
        else:
            pass

        # 1. Prepare your data
        # (Assuming 'preds' and 'true' are your torch/list tensors)
        x = np.array([i.detach().cpu().numpy() for i in true]).squeeze()
        y = np.array([i.detach().cpu().numpy() for i in preds]).squeeze()

        # 2. Apply combined boolean mask
        mask = (y <= 0.99) & (x <= 0.99)
        #mask = (y >= 0) & (x >=0)
        #mask = (y > 0.0001) & (y <= 0.05) & (x >= 0.0001) & (x <= 0.05)
        x_filt = x[mask]
        y_filt = y[mask]

        # 3. Calculate Point Density (KDE)
        # We stack the data, calculate density, then sort so densest points are on top
        xy = np.vstack([x_filt, y_filt])
        z = gaussian_kde(xy)(xy)
        z = z/np.sum(z)
        # Sort the points by density
        idx = z.argsort()
        x_plot, y_plot, z_plot = x_filt[idx], y_filt[idx], z[idx]

        # 4. Calculate Line of Best Fit & R^2
        m, b = np.polyfit(x_filt, y_filt, 1)
        y_fit = m * x_filt + b

        ss_res = np.sum((y_filt - (m * x_filt + b)) ** 2)
        ss_tot = np.sum((y_filt - np.mean(y_filt)) ** 2)
        r2 = 1 - ss_res / ss_tot

        # 5. Plotting
        plt.figure(figsize=(5, 4))

        # The scatter density plot
        sc = plt.scatter(x_plot, y_plot, c=z_plot, s=15, cmap='Blues', edgecolor='none')
        plt.colorbar(sc, label='Relative Density')

        # The best-fit line
        plt.plot(x_filt, y_fit,
                 color='black',
                 linewidth=1.5,   # Slightly thicker for visibility
                 linestyle='--',  # Dashed is usually clearer than dotted for density plots
                 label=f"y = {m:.3f}x + {b:.3f}\n$R^2$ = {r2:.3f}",
                 zorder=10)       # Forces the line to stay ON TOP of the dense dots

        # Formatting
        plt.xlabel("Experimental Site Occupancy")
        plt.ylabel("Predicted Site Occupancy")
        #plt.title("Scatter Density Plot with Best Fit")
        plt.legend()
        #plt.grid(alpha=0.3)

        # Optional: Set y-limits if needed
        # plt.ylim(-2, 8)
        plt.tight_layout()
        os.makedirs(f'./figures/{self.stamp}', exist_ok=True)
        plt.savefig(f'./figures/{self.stamp}/parity_plot_{self.stamp}.png')
        #plt.savefig(f'./figures/parity_plot_{self.stamp}.png')
        plt.show()
        plt.close()
        mae = np.mean(np.abs(x_filt - y_filt))
        print('MEAN ABSOLUTE ERROR:',mae) 
        print('R SQUARED:', r2)
        return r2

    def infer(self,custom_checkpoint = None):
        device = self.device
        (
        model_cfg,
        task,
        checkpoint_path,
        batch_size,
        num_epochs,
        criterion_name,
        optimizer,
        n_hid,
        lr,
        test_val_split,
        split,
        gnn_cfg,
        gnn_n_layers,
        gnn_dropout,
        gnn_n_resid_layers,
        gnn_n_mlp_layers,
        clf_cfg,
        clf_hidden_dim,
        dm_cfg,
        df_path,
        ldf_path,
        structures_path,
        composition,
        num_atoms,
        scope,
        include_coords,
        include_cells,
        include_states,
        ) = self.load_config()
        print('LOADED CONFIG!')
        print(df_path)
        data = load_data(df_path = df_path,
                         ldf_path = ldf_path, 
                         structures_path = structures_path, 
                         batch_size = batch_size, 
                         composition = composition,
                         num_atoms = num_atoms,  
                         property_name = dm_cfg['property_name'], 
                         df_save_path = dm_cfg["df_save_path"], 
                         ldf_save_path = dm_cfg["ldf_save_path"],
                         cutoff = dm_cfg["cutoff"], 
                         task = task,
                         scope = scope)
        print('LOADED DATA!')
        globals().update(data)

        model,gnn, downstream, criterion, num, num_val = self.load_model(batched_node_features, 
                                                                         batched_line_node_features, 
                                                                         criterion_name, 
                                                                         task, 
                                                                         optimizer, 
                                                                         split, 
                                                                         n_hid,
                                                                         gnn_n_layers, 
                                                                         batch_size, 
                                                                         gnn_dropout, 
                                                                         gnn_n_resid_layers, 
                                                                         gnn_n_mlp_layers, 
                                                                         clf_hidden_dim, 
                                                                         lr,
                                                                         include_coords, 
                                                                         include_cells, 
                                                                         include_states)
        print('LOADED MODEL!')
        checkpoint_high,checkpoint_low = self.get_checkpoint()

        try:
            state_dict = torch.load(f'./checkpoints/{self.stamp}/{self.stamp}_{checkpoint_high}_epochs.pt', map_location="cpu")
            print(f'loaded state dict {self.stamp}_{checkpoint_high}_epochs.pt')
        except:
            state_dict = torch.load(f'./checkpoints/{self.stamp}/{self.stamp}_{checkpoint_low}_epochs.pt', map_location="cpu")
            print(f'loaded state dict {self.stamp}_{checkpoint_low}_epochs.pt')
        model.load_state_dict(state_dict)

        preds = []
        true = []
        num_epochs = 1
        error_metrics = [[],[],[],[],[]]

        for epoch in range(num_epochs):
            with torch.no_grad():
                for i in range(num_val): # used to be num_val
                    flattened_node_features_val, flattened_edge_indices_val, flattened_edge_features_val = batched_node_features[num + i].to(self.device), batched_edge_indices[num + i].to(self.device), torch.tensor(batched_edge_features[num + i], dtype=torch.float32, device=self.device)

                    flattened_line_node_features_val = batched_line_node_features[num + i].to(dtype=torch.float32, device=self.device)
                    flattened_line_edge_indices_val = batched_line_edge_indices[num + i].to(self.device)
                    flattened_line_edge_features_val = batched_line_edge_features[num + i].to(dtype=torch.float32, device=self.device)

                    if include_states:
                        state_val = torch.zeros(batch_size).to(dtype = torch.float32, device = device)
                    else:
                        state_val = torch.zeros(batch_size).to(dtype = torch.float32, device = device)
                    if include_coords:
                        coords_val = batched_coords[num+i].to(dtype = torch.float32, device = device)
                    else:
                        coords_val = None
                    if include_cells:
                        cells_val = batched_cells[num+i].to(dtype = torch.float32,device = device)
                    else:
                        cells_val = None


                    group_sizes_val = batched_group_sizes[num + i].to(device = device)
                    output_val = gnn(flattened_node_features_val, 
                                     flattened_edge_indices_val, 
                                     flattened_edge_features_val,
                                     global_state = state_val,
                                     group_size = group_sizes_val, 
                                     cell = cells_val, 
                                     coords = coords_val,
                                     line_node_feature = flattened_line_node_features_val, 
                                     line_edge_index = flattened_line_edge_indices_val,
                                     line_edge_feature = flattened_line_edge_features_val, )
                    if scope == 'graph':
                        output_val = gnn.pool(output_val, batched_node_indices[num + i])
                        y_labels_val = batched_labels[num + i].to(dtype=torch.float32, device=device).reshape(batch_size, 1)
                    else:
                        output_val = output_val
                        y_labels_val = batched_labels[num + i].to(dtype=torch.float32, device=device).reshape(batched_labels[num + i].shape[1], 1)
                    pred_val = downstream(output_val)
                    loss_val = criterion(pred_val, y_labels_val)
                    error_metrics[0].append(flattened_node_features_val)
                    error_metrics[1].append(coords_val)
                    error_metrics[2].append(cells_val)
                    error_metrics[3].append(group_sizes_val)
                    error_metrics[4].append([pred_val,y_labels_val])
                    preds.append(pred_val)
                    true.append(y_labels_val)
        with open(f'./error_metrics/{self.stamp}_error_metrics.pkl','wb') as f:
            pickle.dump(error_metrics,f) 
        print('DONE INFERENCING!')
        if task == 'classification':
            self.classification_plotter(true, preds)
        else:
            self.regression_plotter(true, preds, scope)
        print('DONE PLOTTING!')


    def check_r_2(self, custom_checkpoint=None, checkpoint_spacing=100):

        device = self.device

        (
            model_cfg,
            task,
            checkpoint_path,
            batch_size,
            num_epochs,
            criterion_name,
            optimizer,
            n_hid,
            lr,
            test_val_split,
            split,
            gnn_cfg,
            gnn_n_layers,
            gnn_dropout,
            gnn_n_resid_layers,
            gnn_n_mlp_layers,
            clf_cfg,
            clf_hidden_dim,
            dm_cfg,
            df_path,
            ldf_path,
            structures_path,
            composition,
            num_atoms,
            scope,
            include_coords,
            include_cells,
            include_states
        ) = self.load_config()

        print('LOADED CONFIG!')
        print(df_path)

        data = load_data(
            df_path=df_path,
            ldf_path=ldf_path,
            structures_path=structures_path,
            batch_size=batch_size,
            composition=composition,
            num_atoms=num_atoms,
            property_name=dm_cfg['property_name'],
            df_save_path=dm_cfg["df_save_path"],
            ldf_save_path=dm_cfg["ldf_save_path"],
            cutoff=dm_cfg["cutoff"],
            task=task,
            scope=scope
        )

        print('LOADED DATA!')

        globals().update(data)

        model, gnn, downstream, criterion, num, num_val = self.load_model(
            batched_node_features,
            batched_line_node_features,
            criterion_name,
            task,
            optimizer,
            split,
            n_hid,
            gnn_n_layers,
            batch_size,
            gnn_dropout,
            gnn_n_resid_layers,
            gnn_n_mlp_layers,
            clf_hidden_dim,
            lr,
            include_coords,
            include_cells,
            include_states
        )

        print('LOADED MODEL!')
        print('MODEL NAME:',model)
        checkpoint_high, checkpoint_low = self.get_checkpoint()
        with open(f"./logs/{self.stamp}/validation_loss.txt") as f:
            num_lines = sum(1 for _ in f)
        checkpoint_high = ((num_lines // 25)-1) * 25
        #checkpoint_high = 125
        checkpoint_low = 25
        checkpoint_range = range(
            checkpoint_low,
            checkpoint_high + 1,
            checkpoint_spacing
        )
        r2s = []
        for checkpoint in checkpoint_range:
            #checkpoint = 2000
            print(f'\n==============================')
            print(f'CHECKPOINT: {checkpoint}')
            print(f'==============================')

            try:

                state_dict = torch.load(
                    f'./checkpoints/{self.stamp}/{self.stamp}_{checkpoint}_epochs.pt',
                    map_location="cpu"
                )

                print(f'loaded state dict {self.stamp}_{checkpoint}_epochs.pt')

            except Exception as e:

                print(f'FAILED TO LOAD CHECKPOINT {checkpoint}')
                print(e)

                continue

            model.load_state_dict(state_dict)
            num_params = sum(p.numel() for p in model.parameters())
            print(f'Model has {num_params} parameters')
            preds = []
            true = []

            infer_epochs = 1

            error_metrics = [[], [], [], [], []]
            for epoch in range(infer_epochs):

                with torch.no_grad():

                    for i in range(num_val):

                        flattened_node_features_val = batched_node_features[num + i].to(self.device)

                        flattened_edge_indices_val = batched_edge_indices[num + i].to(self.device)

                        flattened_edge_features_val = torch.tensor(
                            batched_edge_features[num + i],
                            dtype=torch.float32,
                            device=self.device
                        )

                        flattened_line_node_features_val = batched_line_node_features[num + i].to(
                            dtype=torch.float32,
                            device=self.device
                        )

                        flattened_line_edge_indices_val = batched_line_edge_indices[num + i].to(
                            self.device
                        )

                        flattened_line_edge_features_val = batched_line_edge_features[num + i].to(
                            dtype=torch.float32,
                            device=self.device
                        )
                        if include_states:
                            state_val = torch.zeros(batch_size).to(dtype = torch.float32, device = device)
                        else:
                            state_val = torch.zeros(batch_size).to(dtype = torch.float32, device = device)
                        if include_coords:
                            coords_val = batched_coords[num+i].to(dtype = torch.float32, device = device)
                        else:
                            coords_val = None
                        if include_cells:
                            cells_val = batched_cells[num+i].to(dtype = torch.float32,device = device)
                        else:
                            cells_val = None
                        """
                        state_val = torch.zeros(batch_size).to(
                            dtype=torch.float32,
                            device=device
                        )

                        cells_val = batched_cells[num + i].to(
                            dtype=torch.float32,
                            device=self.device
                        )

                        coords_val = batched_coords[num + i].to(
                            dtype=torch.float32,
                            device=self.device
                        )
                        """
                        #cells_val,coords_val = None,None
                        group_sizes_val = batched_group_sizes[num + i].to(
                            device=device
                        )

                        output_val = gnn(
                            flattened_node_features_val,
                            flattened_edge_indices_val,
                            flattened_edge_features_val,
                            global_state=state_val,
                            group_size=group_sizes_val,
                            cell=cells_val,
                            coords=coords_val,
                            line_node_feature=flattened_line_node_features_val,
                            line_edge_index=flattened_line_edge_indices_val,
                            line_edge_feature=flattened_line_edge_features_val
                        )

                        if scope == 'graph':

                            output_val = gnn.pool(
                                output_val,
                                batched_node_indices[num + i]
                            )

                            y_labels_val = batched_labels[num + i].to(
                                dtype=torch.float32,
                                device=device
                            ).reshape(batch_size, 1)

                        else:

                            y_labels_val = batched_labels[num + i].to(
                                dtype=torch.float32,
                                device=device
                            ).reshape(
                                batched_labels[num + i].shape[1],
                                1
                            )

                        pred_val = downstream(output_val)

                        loss_val = criterion(pred_val, y_labels_val)

                        error_metrics[0].append(flattened_node_features_val)
                        error_metrics[1].append(coords_val)
                        error_metrics[2].append(cells_val)
                        error_metrics[3].append(group_sizes_val)
                        error_metrics[4].append([pred_val, y_labels_val])

                        preds.append(pred_val)
                        true.append(y_labels_val)
            """
            with open(
                f'./error_metrics/{self.stamp}_{checkpoint}_error_metrics.pkl',
                'wb'
            ) as f:

                pickle.dump(error_metrics, f)
            """
            print(f'DONE INFERENCING CHECKPOINT {checkpoint}!')
            
            if task == 'classification':

                self.classification_plotter(true, preds)

            else:

                r2=self.regression_plotter(true, preds, scope)
                r2s.append(r2)
            print(f'DONE PLOTTING CHECKPOINT {checkpoint}!')
        with open(
            f'./error_metrics/{self.stamp}_error_metrics.pkl',
            'wb'
        ) as f:

            pickle.dump(error_metrics, f)

        plt.scatter(checkpoint_range,r2s)
        os.makedirs(f'./figures/{self.stamp}', exist_ok=True)
        plt.savefig(f'./figures/{self.stamp}/r_2_analysis_{self.stamp}.png')
        with open(f'./error_metrics/{self.stamp}_r_2_values.pkl','wb') as f:
            pickle.dump([f'{self.stamp}',checkpoint_range,r2s],f)


def main():
    parser = argparse.ArgumentParser(description="Run GNN inference.")
    parser.add_argument(
        "--stamp",
        type=str,
        required=True,
        help="Timestamp string for the inference run (e.g. 2026-02-23_11:57:12.172453)",
    )
    """
    parser.add_argument(
        "--df_path",
        type=str,
        default="./dis_gnn/data/data/test_df.pkl",
        help="Path to df pickle",
    )
    parser.add_argument(
        "--ldf_path",
        type=str,
        default="./dis_gnn/data/data/test_ldf.pkl",
        help="Path to ldf pickle",
    )

    parser.add_argument("--config_path",
            type=str,
            required=True,
            help="Path to config file, config is loaded in ./logs/"
            )
    """
    args = parser.parse_args()

    test = Inference(args.stamp)#, args.df_path, args.ldf_path)
    #test.infer()
    test.check_r_2(checkpoint_spacing=25)

if __name__ == "__main__":
    main()
