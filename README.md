
<img width="2500" height="1274" alt="DIS-GNN-WORKFLOW_vertical (3)" src="https://github.com/user-attachments/assets/8387c54a-203e-4b88-b250-aeea97fd497b" />


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
 pip install -r requirements.txt
  ```
## 4. Using dis-GNN
- via the usage file, to train a model run 
```bash
python -m main --config ./dis_gnn/config/config.yaml
```
