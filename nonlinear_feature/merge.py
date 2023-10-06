import numpy as np
import pandas as pd



rna_G =  np.loadtxt('../data/rna.txt')
rna_U = np.load("../data/r_feature.npy")


disease_G = np.loadtxt('../data/disease.txt')
disease_V = np.load("../data/d_feature.npy")


rna_Similarity = np.hstack((rna_G,rna_U))
disease_Similarity = np.hstack((disease_G,disease_V))

rna_feature = pd.DataFrame(rna_Similarity)
disease_feature = pd.DataFrame(disease_Similarity)


rna_feature.to_csv('../data/rna_feature.csv')
disease_feature.to_csv('../data/disease_feature.csv')