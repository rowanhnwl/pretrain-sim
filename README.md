## Installation

### conda environment setup
```
conda create -n pts python
conda activate pts
pip install -r requirements.txt
```

### ZINC-250k download
```
mkdir zinc250k
wget https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv -P ./zinc250k
```