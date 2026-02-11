
<img width="2500" height="1267" alt="DIS-GNN-WORKFLOW_vertical (4)" src="https://github.com/user-attachments/assets/1da4a5d4-2307-461f-8b8d-137c1972eaf5" />


# Getting DIS-GNN

- Main packages needed
pytorch version

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
 pip install -r requirements.txt
  ```
## 4. Using dis-GNN
- via the usage file, to train a model run 
```bash
python -m main --config ./dis_gnn/config/config.yaml
```
