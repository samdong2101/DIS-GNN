import numpy as np 

class EarlyStopping:
  def __init__(file_path, model, val_loss, patience = 25):
    self.file_path = file_path
    self.model = model
    self.patience = patience 
    self.val_loss = val_loss
  def early_stopping(val_loss, patience = 10):
    
      train_chunk = train_loss[-patience:]
      val_chunk = val_loss[-patience:]
      x_train = np.arange(len(train_chunk))
      x_val = np.arange(len(val_chunk))
      m_train, b_train = np.polyfit(x_train, train_chunk, 1)
      m_val, b_val = np.polyfit(x_val, val_chunk, 1)
      min_val_loss = np.min(val_loss)
    
      if np.min(val_chunk) > min_val_loss:
          return True
  
  def save_checkpoint(file_path, model):
    stopping = self.early_stopping(self.val_loss, self.patience)
    if early_stopping:
      torch.save(model.state_dict(), file_path)
      return stopping





  
  
  
