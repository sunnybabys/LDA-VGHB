import numpy as np
import pandas as pd



f_r = np.load('./data2/rna5_feature.npy')
f_d = np.load('./data2/disease5_feature.npy')

all_associations = pd.read_csv('./data2' + '/pair.txt', sep=' ', names=['r', 'd', 'label'])

#label = pd.read_excel('./data1/lncRNADisease-lncRNA-disease associations matrix.xls',header=0,index_col=0)

#label.to_csv("./data1/label.csv",header=None,index=None)





dataset = []

for i in range(int(all_associations.shape[0])):
    r = all_associations.iloc[i, 0]
    c = all_associations.iloc[i, 1]
    label = all_associations.iloc[i, 2]
    dataset.append(np.hstack((f_r[r], f_d[c], label)))

all_dataset = pd.DataFrame(dataset)

all_dataset.to_csv("./data2/data5.csv",header=None,index=None)

print("Fnished!")