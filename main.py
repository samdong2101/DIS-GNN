print('Dis-GNN-v.0.0.0')


gnn = GNN(in_dim, n_hid, num_types, num_relations, n_heads, n_layers, dropout=drop_out,
      conv_name='hgt', prev_norm=False, last_norm=False, use_RTE=False, n_resid_layers=2).to(device)
classifier = Classifier(n_hid, hidden_dim=32)
model = nn.Sequential(gnn, classifier).to(device)
#model.load_state_dict(torch.load("/blue/hennig/sam.dong/disordered_classifier/model_best_checkpoint.pth"))
optimizer = torch.optim.Adam(model.parameters(), lr=6e-6)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.75)
criterion = nn.BCELoss()
num_epochs = 500
stats = 'NEW LOG'
description = f'changing {variable_name} FROM {old_variable} TO {variable} --> 0 epoch'
knob_logger('/blue/hennig/sam.dong/disordered_classifier/logger_4_b.txt',stats, description, add = True)
for epoch in range(num_epochs):
print('-----------------------------------------------------------------------------------------------------------')
print(f'Epoch: {epoch}')
print('-----------------------------------------------------------------------------------------------------------')
model.train()
description = f'changing {variable_name} FROM {old_variable} TO {variable} --> {epoch} epoch'
train_predictions_inner = []
train_labels_inner = []
train_losses_inner = []

for i in range(int(len(batched_node_features[0:num]))):
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
    #scheduler.step()
    # --- Detach and move to CPU to save GPU memory ---
    train_losses_inner.append(loss.item())
    train_predictions_inner.append(pred.detach().cpu())
    train_labels_inner.append(y_labels.detach().cpu())
    # Free memory for this batch
    if epoch + 1 != num_epochs:
        pass
        #if i == [j for j in range(int(len(batched_node_features[0:num])))][-1]:
            # ind1 = random.randint(0, len(output)-1)
            # ind2 = random.randint(0,len(output)-1)
            # similarity_logger(output[ind1],output[ind2],'output',unsqueeze = False)
            # plt.imshow(output[ind1].detach().cpu().numpy().reshape(1,n_hid))
            # plt.show()
            # plt.imshow(output[ind2].detach().cpu().numpy().reshape(1,n_hid))
            # plt.show()
            
            #print('training prediction:',pred)
        del output, pooled_output, pred, y_labels
    else:
        
        pass
    torch.cuda.empty_cache()



#grad_logger(classifier,gnn)
# --- Compute training accuracy ---
print('train_predictions_inner length:',len(train_predictions_inner))
train_all_preds = torch.cat(train_predictions_inner, dim=0)
train_all_labels = torch.cat(train_labels_inner, dim=0)
train_losses_outer.append(train_losses_inner)
train_preds_binary = (train_all_preds > 0.5).float()
train_acc = (train_preds_binary == train_all_labels).float().mean().item()
train_accuracies.append(train_acc)

print('[Training loss]:', np.mean(train_losses_inner))
print('[Training accuracy]:', train_acc)

# --- Safety check for available GPU memory ---
free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
if free_mem < 2 * 1024**3:  # less than 2GB
    print("⚠️ GPU memory low — breaking early to prevent OOM.")
    torch.cuda.empty_cache()
    break

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

        if epoch + 1 != num_epochs:
            #if i == [j for j in range(num_val)][-1]:
                #pass
                #ind1 = random.randint(0, len(output_val)-1)
                #ind2 = random.randint(0,len(output_val)-1)
                #similarity_logger(output_val[ind1],output_val[ind2],'output',unsqueeze = False)

                #print('validation prediction:',pred_val)
            

            del output_val, pred_val, y_labels_val
        torch.cuda.empty_cache()
val_all_preds = torch.cat(validation_predictions_inner, dim=0)
val_all_labels = torch.cat(validation_labels_inner, dim=0)
val_preds_binary = (val_all_preds > 0.5).float()
validation_losses_outer.append(validation_losses_inner)
val_acc = (val_preds_binary == val_all_labels).float().mean().item()
val_accuracies.append(val_acc)
