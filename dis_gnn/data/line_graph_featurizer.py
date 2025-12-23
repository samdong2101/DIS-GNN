import os
import csv
import math
import io
import fnmatch
import random
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from pymatgen.core import Lattice, Structure, Molecule, Element
from pymatgen.io.vasp.outputs import Poscar
from pymatgen.transformations.standard_transformations import RotationTransformation
import pickle


def filter_by_elements(data, symbols=None):
    if symbols is None:
        return data

    allowed = set(symbols)
    filtered = []

    for entry in data:
        # e.g. "Nb-Sn-O" -> ["Nb", "Sn", "O"]
        elements = entry.get("chemsys", "").split("-")

        # keep only if *every* element is in allowed list
        if all(elem in allowed for elem in elements):
            filtered.append(entry)

    return filtered

class LineGraphFeaturizer:
    def __init__(self, df, save_path):
        self.df = df
        self.save_path = save_path

    def get_node_features(self,index):
        return torch.tensor(self.df['edge_feature'][index]), self.df['structure'][index] 

    def get_angles(self, struct, edge_index):
        struct_angles = []
        pairs_dict = {}
        source_coords = struct.cart_coords
        ids = edge_index[0]
        vals = edge_index[1]
        triplets, line_edge_index = self.get_triplets(edge_index)
        angles = []

        for src,vert,targ in triplets:
            v1 = struct.cart_coords[src] - struct.cart_coords[vert]
            v2 = struct.cart_coords[vert] - struct.cart_coords[targ]
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_theta = np.clip(cos_theta, -1, 1) 
            angle = np.arccos(cos_theta)            # radians
            angles.append(angle)
            angle_deg = np.degrees(angle)

        return angles, line_edge_index

    def get_triplets(self, edge_index, num_nodes=None):
        # edge_index: [2, E]
        src, dst = edge_index
        if num_nodes is None:
            num_nodes = int(torch.max(edge_index)) + 1

        # For each node, list incoming and outgoing neighbors
        # adjacency lists
        neighbors_out = [[] for _ in range(num_nodes)]
        neighbors_in  = [[] for _ in range(num_nodes)]

        for u, v in zip(src.tolist(), dst.tolist()):
            neighbors_out[u].append(v)
            neighbors_in[v].append(u)

        triplets = []

        line_edge_index = [[],[]]# find all i -> j -> k
        for j in range(num_nodes):
            for i in neighbors_in[j]: # i -> j
                for k in neighbors_out[j]:
                    if i!=j and j!=k and i!=k:# j -> k
                        triplets.append([i, j, k])
                        line_edge_index[0].append(j)
                        line_edge_index[1].append(k)
        line_edge_index = torch.tensor(line_edge_index)
        possible_nodes = []
        for trip in triplets:
            
            node1 = trip[0:2]
            node2 = trip[1:3]
            if node1 not in possible_nodes:
                possible_nodes.append(node1)
            if node2 not in possible_nodes:
                possible_nodes.append(node2)
            else:
                pass 
        nodes_dict = {}
        reverse_node_dict = {}
        for node_id,node in enumerate(possible_nodes):
            nodes_dict[node_id] = node
            reverse_node_dict[tuple(node)] = node_id
        line_edge_index = [[],[]]
        for trip in triplets:
            node1 = trip[0:2]
            node2 = trip[1:3]
            line_edge_index[0].append(reverse_node_dict[tuple(node1)])
            line_edge_index[1].append(reverse_node_dict[tuple(node2)])
        return triplets, torch.tensor(line_edge_index, dtype=torch.long)

    def angular_gaussian_basis(self, angles, centers=None, width=0.3, device=None):

        if centers is None:
            centers = torch.linspace(0, torch.pi, 40, device=device)
        else:
            centers = centers.to(device)

        out = []
        for a in angles:
            a = torch.as_tensor(a, device=device).float().unsqueeze(-1) # (N, 1)
            feat = torch.exp(-0.5 * ((a - centers) / width)**2)         # (N, F)
            out.append(feat)
        return out #edge_features

    def get_gaussian_basis(self, index):
        structure = self.df['structure'][index] 
        edge_index = self.df['edge_index'][index] 
        
        #for ind in tqdm(range(len(structures)), desc="extracting angles gaussian basis"):
            
        angle, line_edge_index = self.get_angles(structure, edge_index)
        angle_basis = self.angular_gaussian_basis(angle)
        return angle_basis, line_edge_index
    
    def featurize(self):
        df = []
        for ind in tqdm(range(len(self.df)),desc = 'featurizing line graph'):
            line_node_feature, structure = self.get_node_features(ind)
            line_edge_feature, line_edge_index = self.get_gaussian_basis(ind)
            df.append({
                        'id': ind,
                        'structure': structure,
                        'line_edge_index': line_edge_index,
                        'line_node_feature': line_node_feature,
                        'line_edge_feature':line_edge_feature})
        df = pd.DataFrame(df)
        print(f'df has {len(df)} datapoints')
        if self.save_path is not None: 
            with open(self.save_path, 'wb') as f:
                pickle.dump(df,f)
        
        return df


def main():
    with open('/blue/hennig/sam.dong/dis_gnn_github/DIS-GNN/dis_gnn/data/data/df_1hc_all_elems_angular_basis_12_atoms_4_cutoff_symmetries.pkl','rb') as f:
        df = pickle.load(f) 
    print(df['structure'][0].cart_coords) 
    lgf = LineGraphFeaturizer(df[0:2], '/blue/hennig/sam.dong/dis_gnn_github/DIS-GNN/dis_gnn/data/data/line_graph_df_test.pkl')
    ldf = lgf.featurize()

if __name__ == "__main__":
    main()



