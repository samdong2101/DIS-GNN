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
import string
from pyxtal.symmetry import Group, index_from_letter

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

class GraphFeaturizer:
    def __init__(self, structures, cutoff = 4.0, property_name = 'band_gap', composition = None, num_atoms = 12, e_above_hull = 0.1, save_path = None):
        self.structures = filter_by_elements(structures, composition)
        self.ind = [i for i,structure in enumerate(self.structures) if len(structure['structure']['sites']) <= num_atoms and structure['energy_above_hull'] is not None and
    structure['energy_above_hull'] <= e_above_hull]
        self.structs = [Structure.from_dict(self.structures[i]['structure']) for i in self.ind]
        self.structs = [s * [2, 2, 1] if len(s) <= 2 else s for s in self.structs]
        self.properties = [self.structures[i][property_name] for i in self.ind]
        self.cutoff = cutoff
        self.property_name = property_name
        self.save_path = save_path
    def create_adjacency_matrices(self, structure, cutoff):
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
        node_features = []
        for i in range(len(structure)):
            node_init = np.zeros(118)
            node_init[structure.atomic_numbers[i]] = 1
            node_features.append(node_init)

        node_types = None
        return torch.tensor(node_features).to(dtype=torch.float32), node_types

    def create_edge_features(self, structure,adjacency_matrix):
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
        return torch.tensor(edge_feature)

    def get_angles(self, struct, edge_index):
        struct_angles = []
        pairs_dict = {}
        source_coords = struct.cart_coords
        ids = edge_index[0]
        vals = edge_index[1]
        common_atoms = torch.nonzero(torch.bincount(ids) > 1).flatten()
        for num in common_atoms:
            pairs_dict[int(num)] =[]
            pairs_dict[int(num)].append(source_coords[vals[torch.where(ids==num)]] - source_coords[num])
        angles_per_atom = []
        for src,targ in enumerate(list(pairs_dict.values())):
            vectors = targ[0]
            angles = []
            num_vecs = len(vectors) 
            
            for i in range(num_vecs):
                for j in range(i+1, num_vecs):
                    v1 = vectors[i]
                    v2 = vectors[j]
                    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_theta = np.clip(cos_theta, -1, 1) 
                    angle = np.arccos(cos_theta)            # radians
                    angles.append(angle)
                angles_per_atom.append(torch.mean(torch.tensor(angles)))# degrees
        return angles_per_atom
    
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
        return torch.tensor(np.array(out))

    def featurize(self):
        df = []
        for i, structure in tqdm(enumerate(self.structs), total=len(self.structs), desc="featurizing structures"):
            adjacency_matrix,edge_index, edge_type = self.create_adjacency_matrices(structure, self.cutoff)
            node_feature, node_type = self.create_node_features(structure)
            edge_feature = self.create_edge_features(structure, adjacency_matrix)
            angles = self.get_angles(structure, edge_index)
            

            if len(edge_feature) == 0 or len(angles)==0:
                continue
             
            angular_basis = self.angular_gaussian_basis(angles)

            if torch.tensor(edge_feature).shape != angular_basis.shape:
                continue
            edge_attr = torch.concat([torch.tensor(edge_feature),angular_basis],dim = 1)
            
            try:
                edge_index[0].max()+1
            except:
                print('structure:', structure)
                print('adjacency_matrix:', adjacency_matrix)
                print('edge index:',edge_index)
                print(alksdjf)
            #assert edge_index[0].max()+1 == node_feature.shape[0], f"{edge_index[0].max()} != {node_feature.shape}, {adjacency_matrix}"
            if edge_index[0].max()+1 != node_feature.shape[0]:
                continue

            df.append({
                'id': i,
                'structure': structure,
                'adjacency_matrix': adjacency_matrix,
                'edge_index': edge_index,
                'edge_type': edge_type,
                'node_feature': node_feature,
                'node_type': node_type,
                'radial_basis': edge_feature,
                'angular_basis':angular_basis,
                'edge_feature':edge_attr,
                f'{self.property_name}': self.properties[i] if self.properties[i] is not None else 0
            })
        print(f'dataframe has {len(df)} points')
        df = pd.DataFrame(df)
        if self.save_path is not None:
            with open(self.save_path, 'wb') as f:
                pickle.dump(df, f)
        return df

