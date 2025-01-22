import pickle as pck
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

#data_dir = 'tgnnexplainer/src/dataset/data/'
#
#data = pd.read_csv(data_dir + 'wikipedia.csv')

combined_tgnne_results = None
for i in [0,15,17,20]:
    with open(f'tgnnexplainer/benchmarks/results/tgnne_results_{i}_wikipedia_tgn.pkl', 'rb') as file:
        tgnne_results = pck.load(file)
        sub_keys = list(tgnne_results.values())[0].keys()
        if combined_tgnne_results is None:
            combined_tgnne_results = {k:{k_: [] for k_ in sub_keys} for k in tgnne_results.keys()}
        for k in tgnne_results.keys():
            for k_ in sub_keys:
                combined_tgnne_results[k][k_].extend(tgnne_results[k][k_])


with open(f'tgnnexplainer/benchmarks/results/sa_results_wikipedia_tgn_exp_sizes.pkl', 'rb') as file:
    sa_results = pck.load(file)

with open(f'tgnnexplainer/benchmarks/results/temp_me_results_wikipedia_tgn_exp_sizes.pkl', 'rb') as file:
    temp_me_results = pck.load(file)


sa_errors = []
sa_sizes = []
sa_delta_fidelities = []
print('------- STX-Search --------')
for k in sa_results.keys():
#    for i in range(len(sa_results[k]['target_event_idxs'])):
#        sa_errors.append(abs(sa_results[k]['model_predictions'][i] - sa_results[k]['explanation_predictions'][i]))
#        sa_sizes.append(len(sa_results[k]['explanations'][i]))
    errors = [abs(y-y_hat) for y, y_hat in zip(sa_results[k]['model_predictions'], sa_results[k]['explanation_predictions'])]
#    errors = errors[:-3]
    sa_error = np.mean(errors)
    print('Exp Size: ', k, 'Avg Error', sa_error)
    sa_errors.append(sa_error)
    print(sa_results[k]['delta_fidelity'])
    sa_delta_fidelities.append(np.mean(sa_results[k]['delta_fidelity']))
#    combined_sa_results[k].append(sa_results[k][])

tgnne_results = combined_tgnne_results
print(tgnne_results.keys())
tgnne_errors = []
tgnne_sizes = []
tgnne_delta_fidelities = []
print('------- TGNNExplainer --------')
for k in tgnne_results.keys():

#    for i in range(len(tgnne_results[k]['target_event_idxs'])):
#        tgnne_errors.append(abs(tgnne_results[k]['model_predictions'][i] - tgnne_results[k]['explanation_predictions'][i]))
#        tgnne_sizes.append(len(tgnne_results[k]['explanations'][i]))
#
    tgnne_error = np.mean([abs(y-y_hat) for y, y_hat in zip(tgnne_results[k]['model_predictions'], tgnne_results[k]['explanation_predictions'])])
    print('Exp Size: ', k, 'Avg Error', tgnne_error)
    tgnne_errors.append(float(tgnne_error))
    tgnne_delta_fidelities.append(np.mean(tgnne_results[k]['delta_fidelity']))



print('------- Temp-ME --------')
temp_me_errors = []
temp_me_sizes = []
temp_me_delta_fidelities = []
for k in temp_me_results.keys():
    temp_me_error = np.mean([abs(y-y_hat) for y, y_hat in zip(temp_me_results[k]['model_predictions'], temp_me_results[k]['explanation_predictions'])])
    print('Exp Size: ', k, 'Avg Error', temp_me_error)
    temp_me_errors.append(temp_me_error)
    temp_me_delta_fidelities.append(np.mean(temp_me_results[k]['delta_fidelity']))

fig, axes = plt.subplots(2, 3)
#
axes[0][0].set_title('STX-Search')
axes[0][1].set_title('TGNNExplainer')
axes[0][2].set_title('Temp-ME')
axes[0][0].set_xlabel('Gamma (Exp Size Penalty)')
axes[0][1].set_xlabel('Exp Size')
axes[0][2].set_xlabel('Exp Size')
axes[0][0].set_ylabel('Avg Error')
axes[0][1].set_ylabel('Avg Error')
axes[0][2].set_ylabel('Avg Error')
axes[0][0].plot(sa_results.keys(), sa_errors, label='STX-Search')
axes[0][1].plot(tgnne_results.keys(), tgnne_errors, label='SA')
axes[0][2].plot(temp_me_results.keys(), temp_me_errors, label='TGNNE')

axes[1][0].set_title('STX-Search')
axes[1][1].set_title('TGNNExplainer')
axes[1][2].set_title('Temp-ME')
axes[1][0].set_xlabel('Gamma (Exp Size Penalty)')
axes[1][1].set_xlabel('Exp Size')
axes[1][2].set_xlabel('Exp Size')
axes[1][0].set_ylabel('Avg Delta Fidelity')
axes[1][1].set_ylabel('Avg Delta Fidelity')
axes[1][2].set_ylabel('Avg Delta Fidelity')
axes[1][0].plot(sa_results.keys(), sa_delta_fidelities, label='STX-Search')
axes[1][0].plot(tgnne_results.keys(), tgnne_delta_fidelities, label='SA')
axes[1][0].plot(temp_me_results.keys(), temp_me_delta_fidelities, label='TGNNE')

#plt.scatter([, sa_errors, label='STX-Search')
#plt.scatter(tgnne_sizes, tgnne_errors, label='SA')
#plt.plot([10, 20, 30], tgnne_errors, label='TGNNE')
plt.legend()
plt.show()
