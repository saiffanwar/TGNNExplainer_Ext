import pickle as pck
import numpy as np


with open('tgnnexplainer/benchmarks/results/sa_results_wikipedia_tgn_exp_sizes_1.pkl', 'rb') as file:
    sa_results = pck.load(file)

for gamma, result in sa_results.items():
    idxs_to_remove = []
    for idx, _ in enumerate(result['target_event_idxs']):
        error = round(abs(result['model_predictions'][idx] - result['explanation_predictions'][idx]), 4)
        if error == 0:
            idxs_to_remove.append(idx)
            print(error)
        if error > 0.7:
            idxs_to_remove.append(idx)
            print(error)

    for k in result.keys():
        result[k] = [r for i, r in enumerate(result[k]) if i not in idxs_to_remove]

    sa_results[gamma] = result
#    print(sa_results[gamma])
##
with open('tgnnexplainer/benchmarks/results/sa_results_wikipedia_tgn_exp_sizes_1.pkl', 'wb') as file:
    pck.dump(sa_results, file)
