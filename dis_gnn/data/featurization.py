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

class GraphFeaturizer:
    def __init__(self, structures, cutoff = 4.0, property_name = 'band_gap', composition = None, num_atoms = 12, e_above_hull = 0.1, save_path = None):
        self.structures = filter_by_elements(structures, composition)
        self.ind = [i for i,structure in enumerate(self.structures) if len(structure['structure']['sites']) <= num_atoms and structure['energy_above_hull'] is not None and
    structure['energy_above_hull'] <= e_above_hull]
        self.structs = [Structure.from_dict(self.structures[i]['structure']) for i in self.ind]
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
    def create_node_features(self, structure):
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
        elems = [Element.from_Z(i) for i in structure.atomic_numbers]
        new_node = torch.tensor([
            [
                e.number,
                float(e.atomic_mass or 0),
                float(e.atomic_radius or 0),
                float(e.electron_affinity or 0),
                float(e.group or 0),
                float(e.ionization_energy or 0),
                0.0 if (e.X is None or math.isnan(e.X)) else float(e.X),
                float(
                    list(getattr(e, "atomic_orbitals_eV", {"s": 0}).values())[-1]
                    if getattr(e, "atomic_orbitals_eV", None)
                    else 0
                ),
            ]
            for e in elems
        ])
        node_types = [
            next((k for k, v in groups.items() if e.symbol in v), None)
            for e in elems
        ]
        return new_node,node_types

    def create_node_features_old(self,structure):
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
    
        elems = [Element.from_Z(i) for i in structure.atomic_numbers]
        node_features = []
    
        for e in elems:
            # Original 8 features
            features = torch.tensor([
                e.number,
                float(e.atomic_mass or 0),
                float(e.atomic_radius or 0),
                float(e.electron_affinity or 0),
                float(e.group or 0),
                float(e.ionization_energy or 0),
                0.0 if (e.X is None or math.isnan(e.X)) else float(e.X),
                float(
                    list(getattr(e, "atomic_orbitals_eV", {"s": 0}).values())[-1]
                    if getattr(e, "atomic_orbitals_eV", None)
                    else 0
                ),
            ], dtype=torch.float)
    
            # Create a flat vector of length 126
            embedded = torch.zeros(126, dtype=torch.float)
            # Insert the feature vector starting at atomic number index (subtract 1 for 0-indexing)
            start_idx = e.number - 1
            embedded[start_idx:start_idx + 8] = features
            node_features.append(embedded)
    
        node_features = torch.stack(node_features, dim=0)  # shape: (num_atoms, 126)
    
        node_types = [
            next((k for k, v in groups.items() if e.symbol in v), None)
            for e in elems
        ]
    
        return node_features, node_types
    def create_edge_features(self,structure,adjacency_matrix):
        bond_distances = []
        edge_features = []
        count = 0 
        t = 0
        start_time = time.time()
        def gaussian_expansion(dist, num_bins, width):
            centers = np.linspace(0, self.cutoff, num_bins)
            return np.exp(-((dist - centers)**2) / (width**2))
        bonded_atoms = np.where(adjacency_matrix == 1) 
        element_i = [structure.species[bonded_atoms[0][count_i]] for count_i in range(len(bonded_atoms[0]))]
        element_j = [structure.species[bonded_atoms[1][count_j]] for count_j in range(len(bonded_atoms[1]))]
        atomic_radii_i = [
            i.atomic_radius if (i.atomic_radius is not None and not math.isnan(i.atomic_radius))
            else 10
            for i in element_i
        ]
        atomic_radii_j = [
            i.atomic_radius if (i.atomic_radius is not None and not math.isnan(i.atomic_radius))
            else 10
            for i in element_j
        ]
        bond_lengths = np.array(atomic_radii_i) + np.array(atomic_radii_j)
        gauss = [gaussian_expansion(i,20,0.5) for i in bond_lengths]
        edge_feature = gauss #[gauss for count3 in range(len(bonded_atoms[0]))]
        edge_features.append(edge_feature)
        bond_distances.append(bond_lengths)
        count += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        return edge_features

    
    def featurize(self):
        df = []
        for i, structure in tqdm(enumerate(self.structs), total=len(self.structs), desc="featurizing structures"):
            adjacency_matrix,edge_index, edge_type = self.create_adjacency_matrices(structure, self.cutoff)
            node_feature, node_type = self.create_node_features(structure)
            edge_feature = self.create_edge_features(structure, adjacency_matrix)
            if len(edge_feature[0]) == 0:
                continue
            df.append({
                'id': i,
                'structure': structure,
                'adjacency_matrix': adjacency_matrix,
                'edge_index': edge_index,
                'edge_type': edge_type,
                'node_feature': node_feature,
                'node_type': node_type,
                'edge_feature': edge_feature,
                f'{self.property_name}': self.properties[i] if self.properties[i] is not None else 0
            })
        df = pd.DataFrame(df)
        if self.save_path is not None:
            with open(self.save_path, 'wb') as f:
                pickle.dump(df, f) 
        return df
