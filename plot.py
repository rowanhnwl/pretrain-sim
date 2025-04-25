import json
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

def get_dataframe(prop_dict):
    props = list(prop_dict.keys())

    df_dict = {prop: list(prop_dict[prop].values()) for prop in props}
    df = pd.DataFrame(df_dict)

    return df, props

p1 = "molecules/tpsa_high_xlogp_high/molecules.json"
p2 = "molecules/tpsa_low_xlogp_low/molecules.json"

with open(p1, "r") as f:
    p1_dict = json.load(f)

with open(p2, "r") as f:
    p2_dict = json.load(f)

df1, props1 = get_dataframe(p1_dict)
df2, props2 = get_dataframe(p2_dict)

sns.kdeplot(data=df1, x=props1[0], y=props1[1], color="r", alpha=0.75, bw_adjust=0.75)
sns.kdeplot(data=df2, x=props2[0], y=props2[1], color="b", alpha=0.75, bw_adjust=0.75)

plt.savefig("double.png")