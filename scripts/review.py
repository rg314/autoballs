import glob
import pandas as pd 
import os
from collections import defaultdict
import dabest
from autoballs.utils import mediun_axon_length_pixels
import matplotlib.pyplot as plt

files = glob.glob('results/**/csvs/*.csv')
pixel_um = 1.3

df_dict = defaultdict(list)
for file in files:
    basename = os.path.basename(file)
    type = basename.split('_')[0]
    
    df = pd.read_csv(file)
    med = mediun_axon_length_pixels(df) * pixel_um

    df_dict[type].append(med)


df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in df_dict.items() ]))


# multi_groups = dabest.load(df, idx=(
#     ("e01", "e1", "ve01"),
#     # ("e01", "ve01"),
#     ))
# multi_groups.mean_diff.plot()
    
# plt.show()