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
import psutil
from pymatgen.core import Lattice, Structure, Molecule, Element
from pymatgen.io.vasp.outputs import Poscar
from pymatgen.transformations.standard_transformations import RotationTransformation
import pickle
import string
from pyxtal.symmetry import Group, index_from_letter
from ase.data import atomic_masses

def filter_by_elements(data, symbols=None):
    if symbols is None:
        return data

    allowed = set(symbols)
    filtered = []

    for entry in data:
        elements = entry.get("chemsys", "").split("-")
        if all(elem in allowed for elem in elements):
            filtered.append(entry)
    return filtered

class GraphFeaturizer:
    def __init__(self, 
                 structures, 
                 cutoff = 4.0, 
                 property_name = 'band_gap', 
                 composition = None, 
                 num_atoms = 12, 
                 e_above_hull = 0.1, 
                 save_path = None, 
                 scope = 'graph'):
        
        with open('/blue/hennig/sam.dong/dis_gnn_github/DIS-GNN/dis_gnn/data/data/structures.pkl','rb') as f:
            self.structures = pickle.load(f)
        
        """
        # UNCOMMENT THIS FOR GENERAL PREDICTIONS
        #self.structures = filter_by_elements(structures, composition)
        #self.ind = [i for i,structure in enumerate(self.structures) if len(structure['structure']['sites']) <= num_atoms and structure['energy_above_hull'] is not None and structure['energy_above_hull'] <= e_above_hull and structure['is_magnetic'] == True]
        #print('---------------------------------------------- LEN SELF.IND:',len(self.ind)) 
        #self.ind = [i for i,structure in enumerate(self.structures) if len(structure) <= num_atoms]# ['structure']['sites']) <= num_atoms]
        #self.structs = [Structure.from_dict(self.structures[i]['structure']) for i in self.ind] #[self.structures[i] for i in self.ind]
        #[Structure.from_dict(self.structures[i]['structure']) for i in self.ind]
        """
        self.ind = [i for i,structure in enumerate(self.structures) if len(structure['sites']) <= num_atoms]
        self.structs = [Structure.from_dict(self.structures[i]) for i in self.ind]

        if scope == 'graph':
            with open('/blue/hennig/sam.dong/dis_gnn_github/DIS-GNN/dis_gnn/data/data/labels.pkl','rb') as f:
                self.properties = pickle.load(f)
            self.properties = [self.properties[i] for i in self.ind]
            with open('/blue/hennig/sam.dong/dis_gnn_github/DIS-GNN/dis_gnn/data/data/classification_labels.pkl','rb') as f:
                self.classification_labels = pickle.load(f)
            self.classification_labels = [self.classification_labels[i] for i in self.ind]

        else:
            with open('/blue/hennig/sam.dong/dis_gnn_github/DIS-GNN/dis_gnn/data/data/labels.pkl','rb') as f:
                self.properties = pickle.load(f)
            with open('/blue/hennig/sam.dong/dis_gnn_github/DIS-GNN/dis_gnn/data/data/classification_labels.pkl','rb') as f:
                self.classification_labels = pickle.load(f)
            self.properties = [self.properties[i] for i in self.ind]
            self.classification_labels = [self.classification_labels[i] for i in self.ind]
        self.cutoff = cutoff
        self.property_name = property_name
        self.save_path = save_path


    def create_adjacency_matrices(self, 
                                  structure, 
                                  cutoff):
        begin = time.time()
        num_atoms = len(structure)
        init_adjacency = np.zeros((num_atoms,num_atoms))
        distance_mat = structure.distance_matrix
        bonded = np.where(distance_mat<=cutoff)
        same_atoms = np.where(distance_mat==0)
        init_adjacency[bonded[0],bonded[1]] = 1
        init_adjacency[same_atoms[0],same_atoms[1]] = 0
        row, col = np.where(init_adjacency == 1)
        edge_index = torch.tensor([row,col])
        edge_type = torch.ones_like(edge_index[0])
        end = time.time()
        return init_adjacency,edge_index,edge_type
    

    def create_node_features_old(self, structure):
        groups = {
        'transition_metals': ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc',
                              'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                              'Rf', 'Db', 'Sg', 'Bh', 'Hs'],
        'alkali_metals': ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'],
        'alkaline_earth_metals': ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'],
        'metalloids': ['B', 'Si', 'Ge', 'As', 'Sb', 'Te'],
        'post_transition_metals': ['Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'Po'],
        'reactive_non_metals': ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Se', 'Br', 'I'],
        'noble_gases': ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn'],
        'unknown': ['Mt', 'Ds', 'Rg', 'Cn'],
        'lanthanides': ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'],
        'actinides': ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
        }
        elems = [Element.from_Z(structure.atomic_numbers[i]) for i in range(len(structure))]
        wyckoff_dict = {letter:i for i,letter in enumerate(string.ascii_lowercase, start=1)}
        sg,wp = structure.get_symmetry_dataset()['number'], structure.get_symmetry_dataset()['wyckoffs']
        divisor = len(Group(sg))
        relative_wyckoffs = [wyckoff_dict[letter.lower()]/divisor for letter in wp]
        new_node = torch.tensor([
            [
                elems[i].number,
                float(elems[i].atomic_mass or 0),
                float(elems[i].atomic_radius or 0),
                float(elems[i].electron_affinity or 0),
                float(elems[i].group or 0),
                float(elems[i].ionization_energy or 0),
                0.0 if (elems[i].X is None or math.isnan(elems[i].X)) else float(elems[i].X),
                float(
                    list(getattr(elems[i], "atomic_orbitals_eV",{"s":0}).values())[-1]
                    if getattr(elems[i], "atoic_orbitals_eV",None)
                    else 0
                    ),
                relative_wyckoffs[i]]
            for i in range(len(structure))
            ])

        node_types = [
            next((k for k, v in groups.items() if e.symbol in v), None)
            for e in elems
        ]
        return new_node,node_types

    
    def create_node_features(self, structure):
        begin = time.time()
        node_features = []
        for i in range(len(structure)):
            node_init = np.zeros(118)
            node_init[Element(structure.as_dict()['sites'][i]['species'][0]['element']).Z] = 1
            node_features.append(node_init)

        node_types = None
        end = time.time()
        return torch.tensor(node_features).to(dtype=torch.float32), node_types

    
    def create_edge_features(self, 
                             structure, 
                             adjacency_matrix):
        begin = time.time()
        bond_distances = []
        edge_features = []
        start_time = time.time()
        def gaussian_expansion(dist, num_bins, width):
            centers = np.linspace(0, self.cutoff, num_bins)
            return np.exp(-((dist - centers)**2) / (width**2))
        bonded_atoms = np.where(adjacency_matrix == 1)
        dist_mat = structure.distance_matrix
        dist_mat = dist_mat*adjacency_matrix
        bond_lengths = dist_mat.flatten()
        gauss = [gaussian_expansion(i,40,0.5) for i in bond_lengths if i!=0]
        edge_feature = gauss 
        edge_features.append(edge_feature)
        bond_distances.append(bond_lengths)
        end = time.time()
        return torch.tensor(edge_feature)


    def get_angles(self, 
                   struct, 
                   edge_index):
        begin = time.time()
        struct_angles = []
        pairs_dict = {}
        source_coords = struct.cart_coords
        ids = edge_index[0]
        vals = edge_index[1]
        common_atoms = torch.nonzero(torch.bincount(ids) > 1).flatten()
        for num in common_atoms:
            pairs_dict[int(num)] =[]
            pairs_dict[int(num)].append(source_coords[vals[torch.where(ids==num)]] - source_coords[num])
        
        return pairs_dict

    def angular_gaussian_basis(self, 
                               angles, 
                               centers=None, 
                               width=0.3, 
                               device=None):
        if centers is None:
            centers = torch.linspace(0, torch.pi, 40, device=device)
        else:
            centers = centers.to(device)
    
        out = []
        for a in angles:
            a = torch.as_tensor(a, device=device).float().unsqueeze(-1) # (N, 1)
            feat = torch.exp(-0.5 * ((a - centers) / width)**2)         # (N, F)
            out.append(feat)
        return torch.tensor(np.array(out))


    def featurize(self):
        df = []
        for i, structure in tqdm(enumerate(self.structs), total=len(self.structs), desc="featurizing structures"):
          
            adjacency_matrix,edge_index, edge_type = self.create_adjacency_matrices(structure, self.cutoff)
            node_feature, node_type = self.create_node_features(structure)
            edge_feature = self.create_edge_features(structure, adjacency_matrix)
            angles = self.get_angles(structure, edge_index)

            if len(edge_feature) == 0: #or len(angles)==0:
                continue
            if not angles:
                continue
            edge_attr = torch.tensor(edge_feature)
            try:
                edge_index[0].max()+1
            except Exception as e:
                print(e)
            if edge_index[0].max()+1 != node_feature.shape[0]:
                continue

            df.append({
                'id': i,
                'structure': structure,
                'edge_index': edge_index,
                'edge_type': edge_type,
                'node_feature': node_feature,
                'node_type': node_type,
                'radial_basis': edge_feature,
                'edge_feature':edge_attr,
                f'{self.property_name}': self.properties[i] if self.properties[i] is not None else 0,
                'classification_labels': self.classification_labels[i] if self.classification_labels[i] is not None else 0
                })
        print(f'dataframe has {len(df)} points')
        df = pd.DataFrame(df)
        if self.save_path is not None:
            with open(self.save_path, 'wb') as f:
                pickle.dump(df, f)
        return df

class LineGraphFeaturizer:
    def __init__(self,
                 df, 
                 save_path, 
                 crystal_save_path):
        self.df = df
        self.save_path = save_path
        self.crystal_save_path = crystal_save_path
    def get_node_features(self,index):
        return torch.tensor(self.df['edge_feature'][index]), self.df['structure'][index]

    def get_angles(self, 
                   struct, 
                   edge_index):
        struct_angles = []
        pairs_dict = {}
        source_coords = struct.cart_coords
        ids = edge_index[0]
        vals = edge_index[1]
        triplets, line_edge_index = self.get_triplets(edge_index)
        start_time = time.time()
        angles = []
        
        for src,vert,targ in triplets:
            v1 = struct.cart_coords[src] - struct.cart_coords[vert]
            v2 = struct.cart_coords[vert] - struct.cart_coords[targ]
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_theta = np.clip(cos_theta, -1, 1)
            angle = np.arccos(cos_theta)            # radians
            angles.append(angle)
            angle_deg = np.degrees(angle)
        end_time = time.time()
        return angles, line_edge_index

    def get_angles_new(self, 
                       struct, 
                       edge_index):
        import numpy as np
        import time
        
        source_coords = struct.cart_coords
        start_1 = time.time()
        triplets, line_edge_index = self.get_triplets(edge_index)
        triplets = np.asarray(triplets, dtype=np.int64)
        end_1 = time.time()
    
        start_2 = time.time()
        # unpack indices
        try:
            src = triplets[:, 0]
            vert = triplets[:, 1]
            targ = triplets[:, 2]
        except Exception as e:
            print('error:', e) 

        
        v1 = source_coords[src] - source_coords[vert]
        v2 = source_coords[vert] - source_coords[targ]

        # norms
        norm_v1 = np.linalg.norm(v1, axis=1)
        norm_v2 = np.linalg.norm(v2, axis=1)

        # dot products
        dot = np.einsum('ij,ij->i', v1, v2)

        # cosine + clipping
        denom = norm_v1 * norm_v2
        denom[denom == 0] = 1e-12  # safety against division by zero

        cos_theta = dot / denom
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angles = np.arccos(cos_theta)


        return angles, line_edge_index


    def get_triplets(self, 
                     edge_index, 
                     num_nodes=None):
        start_time = time.time()
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
        start_1 = time.time()
        line_edge_index = [[],[]]# find all i -> j -> k
        for j in range(num_nodes):
            for i in neighbors_in[j]: # i -> j
                for k in neighbors_out[j]:
                    if i!=j and j!=k and i!=k:# j -> k
                        triplets.append([i, j, k])
                        line_edge_index[0].append(j)
                        line_edge_index[1].append(k)
        line_edge_index = torch.tensor(line_edge_index)
        end_1 = time.time()
        possible_nodes = []
        start_2 = time.time()
        
        for trip in triplets:

            node1 = trip[0:2]
            node2 = trip[1:3]
            if node1 not in possible_nodes:
                possible_nodes.append(node1)
            if node2 not in possible_nodes:
                possible_nodes.append(node2)
            else:
                pass
        end_5 = time.time()

        end_2 = time.time()
        nodes_dict = {}
        reverse_node_dict = {}
        start_3 = time.time()
        for node_id, node in enumerate(possible_nodes): # used to be enumerate(possible_nodes) 
            nodes_dict[node_id] = list(node)
            reverse_node_dict[tuple(node)] = node_id
        end_3 = time.time()
        line_edge_index = [[],[]]
        start_4 = time.time()
        for trip in triplets:
            node1 = trip[0:2]
            node2 = trip[1:3]
            line_edge_index[0].append(reverse_node_dict[tuple(node1)])
            line_edge_index[1].append(reverse_node_dict[tuple(node2)])
        end_4 = time.time()
        end_time = time.time()
        return triplets, torch.tensor(line_edge_index, dtype=torch.long)

    def get_triplets_new(self, 
                         edge_index, 
                         num_nodes=None):

        src, dst = edge_index
        if num_nodes is None:
            num_nodes = int(torch.max(edge_index)) + 1

        # For each node, list incoming and outgoing neighbors
        neighbors_out = [[] for _ in range(num_nodes)]
        neighbors_in = [[] for _ in range(num_nodes)]

        for u, v in zip(src.tolist(), dst.tolist()):
            neighbors_out[u].append(v)
            neighbors_in[v].append(u)

        triplets = []

        reverse_node_dict = {}
        
        # Find all i -> j -> k
        for j in range(num_nodes):
            for i in neighbors_in[j]: # i -> j
                for k in neighbors_out[j]:
                    # Ensure it's a valid path of 3 distinct atoms
                    if i != j and j != k and i != k:
                        triplets.append([i, j, k])
                        
                        # Track unique "line nodes" (which are edges in the original graph)
                        node1 = (i, j)
                        node2 = (j, k)
                        
                        if node1 not in reverse_node_dict:
                            reverse_node_dict[node1] = len(reverse_node_dict)
                        if node2 not in reverse_node_dict:
                            reverse_node_dict[node2] = len(reverse_node_dict)

        # Build the line_edge_index in a single pass over the triplets
        line_edge_index = [[], []]
        for trip in triplets:
            node1 = tuple(trip[0:2])
            node2 = tuple(trip[1:3])
            line_edge_index[0].append(reverse_node_dict[node1])
            line_edge_index[1].append(reverse_node_dict[node2])

        return triplets, torch.tensor(line_edge_index, dtype=torch.long)

    def get_triplets_vectorized(self, edge_index):

        src, dst = edge_index
        num_edges = edge_index.size(1)
        
        edge_id = torch.arange(num_edges, device=edge_index.device)
        
        adj = torch.zeros((num_edges, num_edges), dtype=torch.bool)
      
        matches = (edge_index[1].view(-1, 1) == edge_index[0].view(1, -1))
        
        not_self_loop = (edge_index[0].view(-1, 1) != edge_index[1].view(1, -1))
        
        valid_triplets_mask = matches & not_self_loop
        line_edge_index = valid_triplets_mask.nonzero(as_tuple=False).t()
        
        e1_idx = line_edge_index[0]
        e2_idx = line_edge_index[1]
        
        triplets = torch.stack([
            edge_index[0, e1_idx], # i
            edge_index[1, e1_idx], # j
            edge_index[1, e2_idx]  # k
        ], dim=1)
        
        return triplets.tolist(), line_edge_index



    def angular_gaussian_basis(self, 
                               angles, 
                               centers=None,
                               width=0.3, 
                               device=None):

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


    def angular_gaussian_basis_new(self, angles, centers=None, width=0.3, device=None):

        if centers is None:
            centers = torch.linspace(0, torch.pi, 40, device=device)
        else:
            centers = centers.to(device)

        # convert entire array at once
        angles = torch.as_tensor(angles, device=device).float().unsqueeze(-1)  # (N, 1)

        feats = torch.exp(-0.5 * ((angles - centers) / width) ** 2)  # (N, F)

        return feats




    def get_gaussian_basis(self, index):
        structure = self.df['structure'][index]
        edge_index = self.df['edge_index'][index]
        angle, line_edge_index = self.get_angles_new(structure,edge_index) 
        angle_basis = self.angular_gaussian_basis_new(angle)
        angle_basis = torch.tensor(np.array(angle_basis))
        return angle_basis, line_edge_index

    def featurize(self):
        df = []
        id = 0 
        for ind in tqdm(range(len(self.df)),desc = 'featurizing line graph'):
            line_node_feature,structure= self.get_node_features(ind) 
            featurize_end1 = time.time()

            featurize_start2 = time.time()
            line_edge_feature, line_edge_index = self.get_gaussian_basis(ind) 
            featurize_end2 = time.time()

            featurize_start3 = time.time()
            if line_edge_index[0].max()+1 != line_node_feature.shape[0]:
                self.df = self.df.drop(ind)
                continue

            df.append({
                        'id': id,
                        'line_edge_index': line_edge_index,
                        'line_node_feature': line_node_feature,
                        'line_edge_feature':line_edge_feature})
            del line_edge_index, line_node_feature, line_edge_feature
            id += 1
        crystal_df = pd.DataFrame(self.df)
        ldf = pd.DataFrame(df)
        print(f'crystal df has {len(self.df)} datapoints')
        print(f'angle df has {len(ldf)} datapoints')
        
        if self.save_path is not None:
            with open(self.save_path, 'wb') as f:
                pickle.dump(ldf,f)
        print(f'successfully saved ldf to {self.save_path}!')
        for i in range(len(ldf)):
            self.df.iloc[i,self.df.columns.get_loc('id')] = i
        if self.crystal_save_path is not None:
            with open(self.crystal_save_path,'wb') as f:
                pickle.dump(crystal_df,f) 
        print(f'successfully filtered and saved df to {self.crystal_save_path}!')
        return crystal_df,ldf


