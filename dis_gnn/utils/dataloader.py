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
    
  def batch_data(self, batch_size, node_features, edge_indices, edge_features, labels):
    batched_node_features = batch_node_features(node_features,batch_size)
    batched_edge_indices = batch_edge_indices(edge_indices, batch_size, node_features)
    batched_edge_features = batch_edge_features(edge_features,batch_size)
    batched_labels = batch_labels(labels, batch_size)
    return batched_node_features, batched_edge_indices, batched_edge_features, batched_labels

  def get_data():
    node_features, edge_indices, band_gaps, edge_features, labels, structures = self.load_data(self.df)
    batched_node_features, batched_edge_indices, batched_edge_features, batched_labels = self.batch_data(self, batch_size, node_features, edge_indices, edge_features, labels)
    return batched_node_features, batched_edge_indices, batched_edge_featuers, batched_labels

