import json
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind

def get_dataframe(prop_dict):
    props = list(prop_dict.keys())

    df_dict = {prop: list(prop_dict[prop].values()) for prop in props}
    df = pd.DataFrame(df_dict)

    return df, props

p1 = "molecules/tpsa_high/molecules.json"
p2 = "molecules/tpsa_low/molecules.json"

dataset_path = "data/props/tpsa.json"

prop = "tpsa"
joint = False

with open(p1, "r") as f:
    p1_dict = json.load(f)

for (k, v) in list(p1_dict[prop].items())[::-1]:
    if v > 200:
        del p1_dict[prop][k]

with open(p2, "r") as f:
    p2_dict = json.load(f)

for (k, v) in list(p2_dict[prop].items())[::-1]:
    if v > 200:
        del p2_dict[prop][k]

with open(dataset_path, "r") as f:
    data_dict = json.load(f)

if joint:
    df1, props1 = get_dataframe(p1_dict)
    df2, props2 = get_dataframe(p2_dict)

    sns.kdeplot(data=df1, x=props1[0], y=props1[1], color="r", alpha=0.75, bw_adjust=0.5)
    sns.kdeplot(data=df2, x=props2[0], y=props2[1], color="b", alpha=0.75, bw_adjust=0.5)
else:
    p1_vals = list(p1_dict[prop].values())
    p2_vals = list(p2_dict[prop].values())
    data_vals = list(data_dict.values())

    max_val = max(data_vals)
    min_val = min(data_vals)

    p1_vals = [p for p in p1_vals if p >= min_val and p <= max_val]
    p2_vals = [p for p in p2_vals if p >= min_val and p <= max_val]

    p_val = ttest_ind(p1_vals, p2_vals)
    print(p_val)

    sns.kdeplot(p1_vals, color="b")
    sns.kdeplot(p2_vals, color="r")
    #sns.kdeplot(data_vals, color="g")

plt.savefig("double.png")