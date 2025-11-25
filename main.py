print('Dis-GNN-v.0.0.0')
import yaml


with open("/blue/hennig/sam.dong/disordered_classifier/config/config.yaml", "r") as f:
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
gnn_cfg = model_cfg["gnn"]
gnn_n_hid = gnn_cfg["n_hid"]
gnn_n_layers = gnn_cfg["n_layers"]
gnn_dropout = gnn_cfg["dropout"]
gnn_n_resid_layers = gnn_cfg["n_resid_layers"]
gnn_n_mlp_layers = gnn_cfg["n_mlp_layers"]
clf_cfg = model_cfg["classifier"]
clf_hidden_dim = clf_cfg["hidden_dim"]
num = 
if optimizer == 'adam':
      optimizer = torch.optim.Adam(model.parameters(), lr = lr)
if criterion == 'bce':
      criterion = nn.BCELoss()
      


device = "cuda" 
gnn = GNN(in_dim, n_hid, gnn_n_layers, dropout=gnn_drop_out, n_resid_layers=gnn_n_resid_layers).to(device)
classifier = Classifier(n_hid, hidden_dim=clf_hidden_dim)
model = nn.Sequential(gnn, classifier).to(device)
# stats = 'NEW LOG'
# description = f'changing {variable_name} FROM {old_variable} TO {variable} --> 0 epoch'
# knob_logger('/blue/hennig/sam.dong/disordered_classifier/logger_4_b.txt',stats, description, add = True)
for epoch in range(num_epochs):
      model.train()
      train_predictions_inner = []
      train_labels_inner = []
      train_losses_inner = []
      #description = f'changing {variable_name} FROM {old_variable} TO {variable} --> {epoch} epoch'
      for i in tqdm(range(num), desc = f'Epoch {epoch}/{num_epochs}'):
            optimizer.zero_grad(set_to_none=True)
      
            # --- Move only current batch to GPU ---
            flattened_node_features = batched_node_features[i].to(device)
            flattened_edge_indices = batched_edge_indices[i].to(device)
            flattened_node_types_emb = node_types_emb[i].to(device)
            flattened_edge_types = batched_edge_types[i].to(device)
            flattened_edge_features = torch.tensor(batched_edge_features[i], dtype=torch.float32, device=device)
            
            y_labels = batched_target_values[i].to(dtype=torch.float32, device=device).reshape(batch_size, 1)
            
            # --- Forward pass ---
            output = gnn(flattened_node_features, flattened_edge_indices, flattened_node_types_emb,
                 flattened_edge_types, flattened_edge_features)
            
            pooled_output = gnn.pool(output, batched_node_idx[i])
            pred = classifier(pooled_output)
            # --- Loss and backward ---
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
      
      # --- Validation ---
      
      model.eval()
      validation_losses_inner = []
      validation_predictions_inner = []
      validation_labels_inner = []
      with torch.no_grad():
            for i in range(num_val):
                  flattened_node_features_val = batched_node_features[num + i].to(device)
                  flattened_edge_indices_val = batched_edge_indices[num + i].to(device)
                  flattened_node_types_emb_val = node_types_emb[num + i].to(device)
                  flattened_edge_types_val = batched_edge_types[num + i].to(device)
                  flattened_edge_features_val = torch.tensor(batched_edge_features[num + i], dtype=torch.float32, device=device)
                  y_labels_val = batched_target_values[num + i].to(dtype=torch.float32, device=device).reshape(batch_size, 1)
                  output_val = gnn(flattened_node_features_val, flattened_edge_indices_val,
                               flattened_node_types_emb_val, flattened_edge_types_val, flattened_edge_features_val)
                  output_val = gnn.pool(output_val, batched_node_idx[num + i])
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
      validation_losses_outer.append(validation_losses_inner)
      val_acc = (val_preds_binary == val_all_labels).float().mean().item()
      val_accuracies.append(val_acc)
