import numpy as np
import pickle as pck
import pandas as pd
import torch

with open('test_exp.pck', 'rb') as file:
    explanation = pck.load(file)

explanation[0].detach().cpu()
explanation[1].detach().cpu()

reconstructed_explanation = [torch.zeros(explanation[0].shape), torch.zeros(explanation[1].shape)]

for exp_part in range(len(explanation)):
    importances = []
    loc_indexs = []
    sub_locs = []
    for s, imps in enumerate(explanation[exp_part]):
        importances.extend(imps.tolist())
        loc_indexs.extend([i for i in range(len(imps))])
        sub_locs.extend([s]*len(imps))


#df = pd.DataFrame(columns=['loc index', 'importance', 'sub_loc'])
    df = pd.DataFrame({'loc index': loc_indexs, 'importance': importances, 'sub_loc': sub_locs})
    df = df[df['importance'] > 0.0]
    df = df.sort_values(by=['importance'], ascending=False)



    df = df[:exp_size]
    for i, row in df.iterrows():
        reconstructed_explanation[exp_part][int(row['sub_loc']), int(row['loc index'])] = row['importance']




