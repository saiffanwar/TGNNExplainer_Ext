import pickle as pck
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

#data_dir = 'tgnnexplainer/src/dataset/data/'
#
#data = pd.read_csv(data_dir + 'wikipedia.csv')
fig, axes = plt.subplots(2, 2, figsize=(8, 7))

dataset = 'wikipedia'
models = ['tgn', 'tgat']
methods = ['STX-Search', 'TGNNExplainer', 'Temp-ME']#

for m, model in enumerate(models):
    combined_tgnne_results = None
    if model == 'tgn':
        result_batches = [1,2,3,4,5,15,17,20]
    elif model == 'tgat':
        result_batches = [1,2,3,4,5]
#for i in [1,2,3,4,5,15,17,20]:
    for i in result_batches:
        with open(f'tgnnexplainer/benchmarks/results/tgnne_results_{dataset}_{model}_{i}.pkl', 'rb') as file:
            tgnne_results = pck.load(file)
            print(tgnne_results)
            sub_keys = list(tgnne_results.values())[0].keys()
            if combined_tgnne_results is None:
                combined_tgnne_results = {k:{k_: [] for k_ in sub_keys} for k in tgnne_results.keys()}
            for k in tgnne_results.keys():
                for k_ in sub_keys:
                    combined_tgnne_results[k][k_].extend(tgnne_results[k][k_])
    tgnne_results = combined_tgnne_results
#    print(tgnne_results)

    with open(f'tgnnexplainer/benchmarks/results/sa_results_{dataset}_{model}_exp_sizes.pkl', 'rb') as file:
        sa_results = pck.load(file)

    with open(f'tgnnexplainer/benchmarks/results/temp_me_results_{dataset}_{model}_exp_sizes.pkl', 'rb') as file:
        temp_me_results = pck.load(file)


#

    colors = ['r', 'g', 'b']
    for i, (result, method) in enumerate(zip([sa_results, tgnne_results, temp_me_results], methods)):
        print('-------', method, '--------')
        errors = []
        sizes = []
        delta_fidelities = []
        for k in result.keys():
            errors.append(np.mean([abs(y-y_hat) for y, y_hat in zip(result[k]['model_predictions'], result[k]['explanation_predictions'])]))
            sizes.append(k)
            delta_fidelities.append(np.mean(result[k]['delta_fidelity']))
            print('Exp Size:', k, 'Avg Error:', errors[-1], 'Avg Delta Fidelity:', delta_fidelities[-1])
        axes[0][m].plot(sizes, errors, color=colors[i])
        axes[1][m].plot(sizes, delta_fidelities, color=colors[i])

axes[0][0].set_title('TGN - Wikipedia')
axes[0][1].set_title('TGAT - Wikipedia')

fig.legend(methods, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0))
fig.savefig(f'Figures/{dataset}_results.png')
