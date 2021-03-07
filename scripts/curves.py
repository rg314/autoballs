import pandas as pd
import numpy as np 
import glob
import matplotlib.pyplot as plt
from autoballs.utils import mediun_axon_length_pixels


path = '/home/ryan/Documents/GitHub/autoballs/results/20210226 Cam Franze_results/*.csv'

files = glob.glob(path)

for file in files:
    if 'e01' in file:
        df = pd.read_csv(file)
        # print(med)
        rad, inters = df[['Radius', 'Inters.']].T.values
        # print(file)

        ans = getMaxLength(inters)
        target = np.ones(ans-1)
        idx = find_subsequence(inters, target)[0]

        inters[idx:] = 0


        df['Radius'] = rad
        df['Inters.'] = inters
        

        med = mediun_axon_length_pixels(df)
        print(med*1.3213)