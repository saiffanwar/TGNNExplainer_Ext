import pickle as pck
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

#data_dir = 'tgnnexplainer/src/dataset/data/'
#
#data = pd.read_csv(data_dir + 'wikipedia.csv')

def exp_sizes_results(dataset='reddit'):
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))

    models = ['tgat']
    methods = ['STX-Search', 'TGNNExplainer', 'Temp-ME']#

    for m, model in enumerate(models):
        combined_tgnne_results = None
        combined_sa_results = None
        if model == 'tgn':
            tgnne_result_batches = [1,2,3,4,5,15,17,20]

            with open(f'tgnnexplainer/benchmarks/results/sa_results_{dataset}_{model}_exp_sizes.pkl', 'rb') as file:
                sa_results = pck.load(file)
                print(sa_results.keys())
        elif model == 'tgat':
            tgnne_result_batches = [0]
            sa_result_batches = [0]

            for i in sa_result_batches:
                with open(f'tgnnexplainer/benchmarks/results/sa_results_{dataset}_{model}_exp_sizes_{i}.pkl', 'rb') as file:
                    sa_results = pck.load(file)
                    print(sa_results.keys())
                    sub_keys = list(sa_results.values())[0].keys()
                    if combined_sa_results is None:
                        combined_sa_results = {k:{k_: [] for k_ in sub_keys} for k in sa_results.keys()}
                    for k in sa_results.keys():
                        for k_ in sub_keys:
                            combined_sa_results[k][k_].extend(sa_results[k][k_])
            sa_results = combined_sa_results

#for i in [1,2,3,4,5,15,17,20]:
        for i in tgnne_result_batches:
            with open(f'tgnnexplainer/benchmarks/results/tgnne_results_{dataset}_{model}_{i}.pkl', 'rb') as file:
                tgnne_results = pck.load(file)
                sub_keys = list(tgnne_results.values())[0].keys()
                if combined_tgnne_results is None:
                    combined_tgnne_results = {k:{k_: [] for k_ in sub_keys} for k in tgnne_results.keys()}
                for k in tgnne_results.keys():
                    for k_ in sub_keys:
                        combined_tgnne_results[k][k_].extend(tgnne_results[k][k_])
        tgnne_results = combined_tgnne_results
#    print(tgnne_results)


#        with open(f'tgnnexplainer/benchmarks/results/temp_me_results_{dataset}_{model}_exp_sizes.pkl', 'rb') as file:
#            temp_me_results = pck.load(file)
        temp_me_results = None
#
#        # Get common target_idx results
#        for exp_size in sa_results.keys():
#            all_sa_target_idx = set(sa_results[exp_size]['target_event_idxs'])
#            all_tgnne_target_idx = set(tgnne_results[exp_size]['target_event_idxs'])
#            common_target_idx = all_sa_target_idx.intersection(all_tgnne_target_idx)
#
#            sa_results[exp_size]['model_predictions'] = [sa_results[exp_size]['model_predictions'][i] for i, target_idx in enumerate(all_sa_target_idx) if target_idx in common_target_idx]
#            sa_results[exp_size]['explanation_predictions'] = [sa_results[exp_size]['explanation_predictions'][i] for i, target_idx in enumerate(all_sa_target_idx) if target_idx in common_target_idx]
#            sa_results[exp_size]['delta_fidelity'] = [sa_results[exp_size]['delta_fidelity'][i] for i, target_idx in enumerate(all_sa_target_idx) if target_idx in common_target_idx]
#            sa_results[exp_size]['target_event_idxs'] = [target_idx for target_idx in all_sa_target_idx if target_idx in common_target_idx]
#
#            tgnne_results[exp_size]['model_predictions'] = [tgnne_results[exp_size]['model_predictions'][i] for i, target_idx in enumerate(all_tgnne_target_idx) if target_idx in common_target_idx]
#            tgnne_results[exp_size]['explanation_predictions'] = [tgnne_results[exp_size]['explanation_predictions'][i] for i, target_idx in enumerate(all_tgnne_target_idx) if target_idx in common_target_idx]
#            tgnne_results[exp_size]['delta_fidelity'] = [tgnne_results[exp_size]['delta_fidelity'][i] for i, target_idx in enumerate(all_tgnne_target_idx) if target_idx in common_target_idx]
#            tgnne_results[exp_size]['target_event_idxs'] = [target_idx for target_idx in all_tgnne_target_idx if target_idx in common_target_idx]
##
##
#
##
#
        colors = ['r', 'g', 'b']
        for i, (result, method) in enumerate(zip([sa_results, tgnne_results, temp_me_results][:2], methods[:2])):
            print('-------', method, '--------')
            errors = []
            sizes = []
            delta_fidelities = []
            for k in result.keys():
                print(result[k].keys())
                errors.append(np.mean([abs(y-y_hat) for y, y_hat in zip(result[k]['model_predictions'], result[k]['explanation_predictions'])]))
                sizes.append(k)
                if i == 0 and dataset == 'reddit':
                    e_delta_fidelities = [s[0] for s in result[k]['delta_fidelity'] if s[0] != np.inf]
                else:
                    e_delta_fidelities = [s for s in result[k]['delta_fidelity'] if s != np.inf]
                delta_fidelities.append(np.mean(e_delta_fidelities))
                print('Exp Size:', k, 'Avg Error:', errors[-1], 'Avg Delta Fidelity:', delta_fidelities[-1])
#            for r in range(len(result[k]['model_predictions'])):
#                errors.append(abs(result[k]['model_predictions'][r] - result[k]['explanation_predictions'][r]))
#                sizes.append(len(result[k]['explanations'][r]))
#                delta_fidelities.append(result[k]['delta_fidelity'][r])
#        if method == 'STX-Search':
#            sa_errors = errors
#        else:
#            tgnne_errors = errors

                print(len(result[k]['model_predictions']), len(result[k]['explanation_predictions']), len(result[k]['delta_fidelity']))
            axes[0][m].plot(sizes, errors, color=colors[i])
            axes[1][m].plot(sizes, delta_fidelities, color=colors[i])
#    pprint(list(zip(sa_errors, tgnne_errors)))

    axes[0][0].set_title('TGN - Wikipedia')
    axes[0][1].set_title('TGAT - Wikipedia')
    for ax in fig.get_axes():
        ax.set_yscale('log')

    fig.legend(methods, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(f'Figures/{dataset}_results.png')

exp_sizes_results()
exp_sizes_results('wikipedia')

def sa_gammas():

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    dataset = 'wikipedia'
    model = 'tgat'

    results_batches = [0,1]
    combined_sa_results = None

    for i in results_batches:
        with open(f'tgnnexplainer/benchmarks/results/sa_results_{dataset}_{model}_gammas_{i}.pkl', 'rb') as file:
            sa_results = pck.load(file)
            print(sa_results.keys())
            sub_keys = list(sa_results.values())[0].keys()
            if combined_sa_results is None:
                combined_sa_results = {k:{k_: [] for k_ in sub_keys} for k in sa_results.keys()}
            for k in sa_results.keys():
                for k_ in sub_keys:
                    combined_sa_results[k][k_].extend(sa_results[k][k_])
    sa_results = combined_sa_results

    for gamma in sa_results.keys():
        sizes = []
        errors = []
        delta_fidelities = []
#        pprint(list(zip(sa_results[gamma]['model_predictions'], sa_results[gamma]['explanation_predictions'])))
#        errors.append(np.mean([abs(y-y_hat) for y, y_hat in zip(sa_results[gamma]['model_predictions'], sa_results[gamma]['explanation_predictions'])]))
#        print('Gamma:', gamma, 'Avg Error:', errors[-1])
#        delta_fidelities.append(np.mean([s[0] for s in sa_results[gamma]['delta_fidelity']]))
#
#    axes[0].plot(sa_results.keys(), errors)
#    axes[1].plot(sa_results.keys(), delta_fidelities)
        for r in range(len(sa_results[gamma]['model_predictions'])):
            errors.append(abs(sa_results[gamma]['model_predictions'][r] - sa_results[gamma]['explanation_predictions'][r]))
            sizes.append(len(sa_results[gamma]['explanations'][r]))
            delta_fidelities.append(sa_results[gamma]['delta_fidelity'][r][0])
        print(sizes)
        print(errors)
        print(delta_fidelities)
        axes[0][0].scatter(sizes, errors, label=gamma)
        axes[1][0].scatter(sizes, delta_fidelities)
    axes[0][0].set_yscale('log')
    axes[1][0].set_yscale('log')

    delta_fidelities = []
    sizes = []
    errors = []
    for gamma in sa_results.keys():
        errors.append(np.mean([abs(y-y_hat) for y, y_hat in zip(sa_results[gamma]['model_predictions'], sa_results[gamma]['explanation_predictions'])]))
        print('Gamma:', gamma, 'Avg Error:', errors[-1])
        g_delta_fidelities = [s[0] for s in sa_results[gamma]['delta_fidelity'] if s[0] != np.inf]
        print(g_delta_fidelities)

        delta_fidelities.append(np.mean(g_delta_fidelities))

    axes[0][1].plot(sa_results.keys(), errors)
    print(delta_fidelities)
    axes[1][1].plot(sa_results.keys(), delta_fidelities)

    fig.legend(title='Gamma', loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0))

    plt.show()

#sa_gammas()


