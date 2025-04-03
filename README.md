## Installation

### conda environment setup
```
conda create -n pts python=3.10
conda activate pts
pip install -r requirements.txt
conda install tsnecuda
```

### ZINC-250k download
```
mkdir data/
mkdir data/zinc250k
wget https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv -P ./data/zinc250k
```