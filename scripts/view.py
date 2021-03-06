import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.stats.multicomp as multi

data = pd.read_csv('results/20210226 Cam Franze_results/res.csv')


sns.set_style("white")
sns.set_style("ticks")
ax = sns.boxplot(y='Median axon', x='Gel type', data=data, palette="Blues")
ax = sns.swarmplot(y='Median axon', x='Gel type', data=data, color=".25", size=10)
ax.set_ylabel('Axon length [um]')
ax.set_xlabel('Gel type [kPa]')

test = multi.MultiComparison(data['Median axon'], data['Gel type'])
res = test.tukeyhsd()
res_table1 = res.summary()
print(res_table1)


test = multi.pairwise_tukeyhsd(data['Median axon'], data['Gel type'], alpha=0.05)
res_table2 = test.summary()
print(res_table2)

plt.show()