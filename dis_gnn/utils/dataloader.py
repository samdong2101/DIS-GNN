class DataLoader:
  def __init__(df, batch_size = 32):
    self.df = df
    self.batch_size = batch_size
    
  def load_data(self, df):
    node_features = [i for i in df['node_feature']]
    edge_indices = [i for i in df['edge_index']]
    band_gaps = [i for i in df['band_gap']]
    edge_features = [i[0] for i in df['edge_feature']]
    labels = [0 if bg == 0 else 1 for bg in band_gaps]
    structures = [i for i in df['structure']]
    return node_features,edge_indices, band_gaps, edge_features, labels, structures


  def batch_node_features(self, feature_list, batch_size):
      num_batches = int(np.round(len(feature_list)/batch_size)) - 1
      batched_node_features = []
      batch_index = 0
      
      for batch in range(num_batches):
          try:
  
              batched_node_feature = np.concatenate(feature_list[batch_index:batch_index + batch_size])
              batched_node_features.append(torch.tensor(batched_node_feature))
              batch_index = batch_index + batch_size
          except:
              batched_node_feature = np.concatenate(feature_list[batch_index:])
              batched_node_features.append(torch.tensor(batched_node_feature))
      return batched_node_features
  def batch_edge_features(self, feature_list, batch_size):
      num_batches = int(np.round(len(feature_list)/batch_size)) - 1 
      batched_edge_features = []
      batch_index = 0
      for batch in range(num_batches):
          try:
              batched_edge_feature = feature_list[batch_index:batch_index + batch_size]
              batched_edge_feature = [item for sublist in batched_edge_feature for item in sublist]
              batched_edge_features.append(batched_edge_feature)
              batch_index = batch_index + batch_size
          except:
              batched_edge_feature = feature_list[batch_index:]
              batched_edge_feature = [item for sublist in batched_edge_feature for item in sublist]
              batched_edge_features.append(batched_edge_feature)
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
  
  def batch_data(self, batch_size, node_features, edge_indices, edge_features, labels):
    batched_node_features = self.batch_node_features(node_features, self.batch_size)
    batched_edge_indices = self.batch_edge_indices(edge_indices, self.batch_size, node_features)
    batched_edge_features = self.batch_edge_features(edge_features, self.batch_size)
    batched_labels = self.batch_labels(labels, self.batch_size)
    return batched_node_features, batched_edge_indices, batched_edge_features, batched_labels

  def get_data():
    node_features, edge_indices, band_gaps, edge_features, labels, structures = self.load_data(self.df)
    batched_node_features, batched_edge_indices, batched_edge_features, batched_labels = self.batch_data(self, batch_size, node_features, edge_indices, edge_features, labels)
    return batched_node_features, batched_edge_indices, batched_edge_featuers, batched_labels

