import numpy as np
import pandas as pd

# 读取数据

def RNA_intergrate(s1,s2,rate):
    G = pd.read_csv(s1,header=None,index_col=None)   #高斯相似性
    S = pd.read_excel(s2,header=0,index_col=0)

    G_matrix = G.values
    S_matrix = S.values

    result = rate*G_matrix + (1-rate)*S_matrix

    return result

def Dis_intergrate(s1,s2,rate):
    G = pd.read_csv(s1, header=None, index_col=None)  # 高斯相似性
    S = pd.read_excel(s2, header=0, index_col=0)

    G_matrix = G.values
    S_matrix = S.values

    result = rate * G_matrix + (1 - rate) * S_matrix

    return result


f1 = "./data2/rna_GaussianSimilarity.csv"
f2 = "./data2/MNDR-lncRNA functional similarity matrix.xls"

f3 = "./data2/disease_GaussianSimilarity.csv"
f4 = "./data2/MNDR-disease semantic similarity matrix.xls"
rate = 0.5
rna = RNA_intergrate(f1,f2,rate)
disease = Dis_intergrate(f3,f4,rate)
np.savetxt("./data2/inter_lncrna.csv",rna,fmt='%.16f',delimiter=',')
np.savetxt("./data2/inter_diease.csv",disease,fmt='%.16f',delimiter=',')
