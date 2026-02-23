import numpy as np
import torch
import pandas as pd
import pickle
from .featurization import GraphFeaturizer, LineGraphFeaturizer 
def calculate_bce_baseline(proportion_positive: float) -> float:
    """
    Calculates the Binary Cross-Entropy (BCE) loss for the prior probability 
    baseline model, given the proportion of positive (label 1) samples.

    The baseline model always predicts the 'proportion_positive' for every sample.

    Args:
        proportion_positive: The prevalence of the positive class (label 1), 
                             i.e., (Count of 1s) / (Total Count). Must be 
                             between 0.0 and 1.0.

    Returns:
        The BCE baseline loss value.
    """
    p = proportion_positive
    

    if p == 0.0 or p == 1.0:
        return 0.0

    p = np.clip(p, 1e-15, 1 - 1e-15)
    
    baseline_loss = - (p * np.log(p) + (1 - p) * np.log(1 - p))
    
    return baseline_loss



def load_data(structures_path = None, df_path = None, ldf_path = None, batch_size = 32, property_name = 'band_gap', composition = None, num_atoms = 20, cutoff = 4.0, df_save_path = None, ldf_save_path = None, task = 'classification', scope = 'graph'): 
    if df_path is None:
        with open(structures_path,'rb') as f:
            structures = pickle.load(f)
        gf = GraphFeaturizer(structures, cutoff, property_name, composition = composition, num_atoms = num_atoms, save_path = df_save_path, scope = scope)
        df = gf.featurize()
        lgf = LineGraphFeaturizer(df, save_path=ldf_save_path)
        ldf = lgf.featurize()
        dl = DataLoader(df, batch_size = batch_size, graphtype='crystal', task = task, property_name = property_name, scope = scope)
        ldl = DataLoader(ldf, batch_size = batch_size, graphtype='line', task = task, property_name = property_name, scope = scope)
        batched_node_features, batched_edge_indices, batched_edge_features, batched_labels, batched_node_indices, batched_group_sizes, batched_cells, batched_coords = dl.get_data()
        batched_line_node_features, batched_line_edge_indices, batched_line_edge_features = ldl.get_data()
    else:
        with open(df_path, 'rb') as f:
            df = pickle.load(f)
        with open(ldf_path, 'rb') as f:
            ldf = pickle.load(f)
        dl = DataLoader(df, batch_size = batch_size, graphtype='crystal', task = task, property_name = property_name, scope = scope)
        ldl = DataLoader(ldf, batch_size = batch_size, graphtype='line', task = task, property_name = property_name, scope = scope)
        batched_node_features, batched_edge_indices, batched_edge_features, batched_labels, batched_node_indices, batched_group_sizes, batched_cells, batched_coords = dl.get_data()
        batched_line_node_features, batched_line_edge_indices, batched_line_edge_features = ldl.get_data()

    data = {
    "batched_node_features": batched_node_features,
    "batched_edge_indices": batched_edge_indices,
    "batched_edge_features": batched_edge_features,
    "batched_labels": batched_labels,
    "batched_node_indices": batched_node_indices,
    "batched_group_sizes": batched_group_sizes,
    "batched_cells": batched_cells,
    "batched_coords": batched_coords,
    "batched_line_node_features": batched_line_node_features,
    "batched_line_edge_indices": batched_line_edge_indices,
    "batched_line_edge_features": batched_line_edge_features
    }

    return data





class DataLoader:
  def __init__(self, df, batch_size = 32, graphtype = 'crystal', task = 'classification', property_name = 'band_gap', labels_list = None, scope = 'graph'):
    self.df = df
    self.batch_size = batch_size
    self.graphtype = graphtype   
    self.task = task
    self.property_name = property_name
    self.labels_list = labels_list
    self.scope = scope
  def load_data(self, df, labels_list = None):
    if self.graphtype == 'crystal':
        node_features = [i for i in df['node_feature']]
        edge_indices = [i for i in df['edge_index']]
        if labels_list is None:
            labels_list = [i for i in df[self.property_name]]
        edge_features = [i for i in df['edge_feature']]
        if self.scope == 'graph':
            if self.task == 'classification':
                labels = [0 if bg == 0 or np.isnan(bg) else 1 for bg in labels_list]
            else:
                labels = [0 if bg == 0 or np.isnan(bg) else bg for bg in labels_list]
        else:
            labels = labels_list
        structures = [i for i in df['structure']]
        cells = [torch.tensor(i.lattice.matrix).flatten() for i in structures]
        coords = [torch.tensor(i.frac_coords) for i in structures]
        #frac = sum(labels)/len(labels)
        #baseline = calculate_bce_baseline(frac)
        #print('BASELINE BCE LOSS:',baseline)
        return node_features,edge_indices, labels_list, edge_features, labels, structures, cells, coords
    else:
        node_features = [i for i in df['line_node_feature']]
        edge_indices = [i for i in df['line_edge_index']]
        edge_features = [i for i in df['line_edge_feature']]
        return node_features, edge_indices, edge_features

  def batch_node_features(self, feature_list, batch_size):
      num_batches = int(np.round(len(feature_list)/batch_size)) - 1
      batched_node_features = []
      batched_group_sizes = []
      batch_index = 0
      
      for batch in range(num_batches):
          try:  
              batched_group_size = torch.tensor([len(i) for i in feature_list[batch_index:batch_index + batch_size]])
              batched_node_feature = np.concatenate(feature_list[batch_index:batch_index + batch_size])
              batched_node_features.append(torch.tensor(batched_node_feature))
              batched_group_sizes.append(batched_group_size)
              batch_index = batch_index + batch_size
          except:
              batched_group_size = torch.tensor([len(i) for i in feature_list[batch_index:]])
              batched_node_feature = np.concatenate(feature_list[batch_index:])
              batched_node_features.append(torch.tensor(batched_node_feature))
              batched_group_sizes.append(batched_group_size)
      return batched_node_features, batched_group_sizes

  def batch_cells(self, feature_list, batch_size):
      num_batches = int(np.round(len(feature_list)/batch_size)) - 1
      batched_cells = []
      batch_index = 0
      for batch in range(num_batches):
          try:
              batched_cell = torch.cat(feature_list[batch_index:batch_index + batch_size])
              batched_cells.append(torch.tensor(batched_cell))
              batch_index = batch_index + batch_size
          except:
              batched_cell = torch.cat(feature_list[batch_index:])
              batched_cells.append(torch.tensor(batched_cell))
      return batched_cells


  def batch_coordinates(self, coordinate_list, batch_size):
    num_batches = int(np.round(len(coordinate_list) / batch_size)) - 1
    batched_coords = []
    batch_index = 0

    for batch in range(num_batches):
        try:
            batched_coord = torch.cat(coordinate_list[batch_index:batch_index + batch_size])
            batched_coords.append(torch.tensor(batched_coord))
            batch_index += batch_size
        except:
            batched_coord = torch.cat(coordinate_list[batch_index:])
            batched_coords.append(torch.tensor(batched_coord))

    return batched_coords


  def batch_site_properties(self, site_properties_list, batch_size):
    num_batches = int(np.round(len(site_properties_list) / batch_size)) - 1
    batched_site_properties = []
    batch_index = 0

    for batch in range(num_batches):
        try:
            batched_site_property = torch.cat(site_properties_list[batch_index:batch_index + batch_size])
            batched_site_properties.append(torch.tensor(batched_site_property))
            batch_index += batch_size
        except:
            batched_site_property = torch.cat(site_properties_list[batch_index:])
            batched_site_properties.append(torch.tensor(batched_site_property))
    return batched_site_properties




  def batch_edge_features(self, feature_list, batch_size):
      num_batches = int(np.round(len(feature_list)/batch_size)) - 1 
      batched_edge_features = []
      batch_index = 0
      for batch in range(num_batches):
          try:
              batched_edge_feature = feature_list[batch_index:batch_index + batch_size]
              batched_edge_feature = np.concatenate(feature_list[batch_index:batch_index + batch_size])
              batched_edge_features.append(torch.tensor(batched_edge_feature))
              batch_index = batch_index + batch_size
          except:
              batched_edge_feature = feature_list[batch_index:]
              batched_edge_feature = np.concatenate(feature_list[batch_index:])
              batched_edge_features.append(torch.tensor(batched_edge_feature))
      return batched_edge_features

  def batch_edge_indices(self, feature_list, batch_size, node_features):
      num_batches = int(np.round(len(feature_list)/batch_size)) - 1
      batched_edge_indices = []
      batch_index = 0
      shifted_tensors_list = []
      batch = 0 
      
      for b in range(num_batches):
          count = 0
          shift_factor = 0
          shifted_tensors = []
          for i in feature_list[batch:batch+batch_size]:
              try:
                  shifted_tensor = i + shift_factor
                  shifted_tensors.append(shifted_tensor)
                  shift_factor = shift_factor + len(node_features[batch:batch+batch_size][count])
                  count = count + 1
              except:
                  pass
          
          batch = batch + batch_size
          shifted_tensors_list.append(shifted_tensors)
  
      for batch in shifted_tensors_list:
          concat_batch = torch.cat(batch,dim = 1)
          batched_edge_indices.append(concat_batch)
      return batched_edge_indices

  def batch_node_indices(self, feature_list, batch_size):
      num_batches = int(np.round(len(feature_list)/batch_size)) - 1
      batched_node_indices = []
      i = 0
      ind = 0
      for batch in range(num_batches):
          b_zero = [np.zeros(i.shape[0]) for i in feature_list[ind:ind+batch_size]]
          b = [zero + i for i,zero in enumerate(b_zero)]
          batched_node_indices.append(b)
          ind = ind + batch_size
      return batched_node_indices 


  def batch_labels(self, feature_list, batch_size):
      num_batches = int(np.round(len(feature_list)/batch_size)) - 1
      batched_target_values = []
      i = 0
      for batch in range(num_batches):
          batched_target_value = feature_list[i:i + batch_size]
          i = i + batch_size
          batched_tensor = torch.tensor(batched_target_value)
          repeated_tensor = batched_tensor
          batched_target_values.append(repeated_tensor)
      return batched_target_values
  
  def batch_data(self, batch_size, node_features, edge_indices, edge_features, labels=None, cells=None, coords=None, site_properties=None):
        if self.graphtype == 'crystal':
            batched_node_features, batched_group_sizes = self.batch_node_features(node_features, self.batch_size)
            batched_edge_indices = self.batch_edge_indices(edge_indices, self.batch_size, node_features)
            batched_edge_features = self.batch_edge_features(edge_features, self.batch_size)
            batched_node_indices = self.batch_node_indices(node_features, self.batch_size)
            batched_labels = self.batch_labels(labels, self.batch_size)
            batched_cells = self.batch_cells(cells, self.batch_size)
            batched_coords = self.batch_coordinates(coords, self.batch_size)
            #batched_site_properties = self.batch_site_properties(site_properties, self.batch_size) 
            return batched_node_features, batched_edge_indices, batched_edge_features, batched_labels, batched_node_indices, batched_group_sizes, batched_cells, batched_coords
        else:
            batched_node_features, batched_group_sizes = self.batch_node_features(node_features, self.batch_size)
            batched_edge_indices = self.batch_edge_indices(edge_indices, self.batch_size, node_features)
            batched_edge_features = self.batch_edge_features(edge_features, self.batch_size)
            return batched_node_features, batched_edge_indices, batched_edge_features
  def get_data(self):
    if self.graphtype == 'crystal':
        node_features, edge_indices, labels_list, edge_features, labels, structures, cells, coords = self.load_data(self.df)
        batched_node_features, batched_edge_indices, batched_edge_features, batched_labels, batched_node_indices, batched_group_sizes, batched_cells, batched_coords = self.batch_data(self.batch_size, node_features, edge_indices, edge_features, labels, cells, coords)
        return batched_node_features, batched_edge_indices, batched_edge_features, batched_labels, batched_node_indices, batched_group_sizes, batched_cells, batched_coords
    else:
        node_features, edge_indices, edge_features = self.load_data(self.df)
        batched_node_features, batched_edge_indices, batched_edge_features = self.batch_data(self.batch_size, node_features, edge_indices, edge_features)
        return batched_node_features, batched_edge_indices, batched_edge_features 
