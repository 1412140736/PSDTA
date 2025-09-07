# PSDTA

conda create --name psdta python=3.8
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
conda install -c conda-forge rdkit==2022.09.5
pip install scipy biopython pandas biopandas timeout_decorator py3Dmol umap-learn plotly mplcursors lifelines reprint
pip install "saprot"



Create a train, valid and test csv file in a datafolde. The datafolder should contain at least a train.csv , valid.csv and test.csv file. 

For PDBBind v2016:
python main.py --datafolder dataset/pdb2016 --result_path result/PDB2016
