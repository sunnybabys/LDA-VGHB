import numpy as np

s_l = np.load('./data1/r64_feature.npy')
v_l = np.load('./data1/r64_vfeature.npy')
s_d = np.load('./data1/d64_feature.npy')
v_d = np.load('./data1/d64_vfeature.npy')

rna_feature = np.hstack((s_l,v_l))
disease_feature = np.hstack((s_d,v_d))



np.save('./data1/rna64_feature.npy', rna_feature)
np.save('./data1/disease64_feature', disease_feature)

print("拼接完成！")



