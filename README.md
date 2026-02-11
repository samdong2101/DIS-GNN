
<img width="2500" height="1267" alt="DIS-GNN-WORKFLOW_vertical (4)" src="https://github.com/user-attachments/assets/1da4a5d4-2307-461f-8b8d-137c1972eaf5" />


# Getting DIS-GNN

## 1. Create Conda Environment
- While not mandatory, we recommend creating a clean conda environment before installing DIS-GNN to avoid potential package conflicts. You can create and activate a conda environment with the following commands:
```bash
conda create -n dis_gnn python=3.12
  ```
### Activate the environment
 ```bash
 conda activate dis_gnn
  ```
## 2. Install from source
 ```bash
 git clone https://github.com/samdong2101/DIS-GNN.git
  ```
## 3. Install required packages
 ```bash
 conda install pip
 ```
 ```bash 
pip install -r requirements.txt
  ```
Before installing the requirements file, we recommend installing torch first
 ```bash 
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  ```
followed by 
 ```bash 
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
  ```
## 4. Using dis-GNN
- via the usage file, to train a model run 
```bash
python -m main --config ./dis_gnn/config/config.yaml
```
