import pickle as pck
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import copy

#data_dir = 'tgnnexplainer/src/dataset/data/'
#
#data = pd.read_csv(data_dir + 'wikipedia.csv')
results_dir = 'tgnnexplainer/benchmarks/results/results_backup/'

def exp_sizes_results(dataset='reddit'):
    fig, axes = plt.subplots(2, 4, figsize=(14, 4), sharex=True)
    datasets = ['wikipedia', 'reddit']
    models = ['tgn', 'tgat']
    methods = ['STX-Search', 'TGNNExplainer', 'Temp-ME']#
    ax = fig.get_axes()
    plt_num = 0
    for d, dataset in enumerate(datasets):
        for m, model in enumerate(models):
            combined_tgnne_results = None
            combined_sa_results = None
            combined_temp_me_results = None
            if model == 'tgn':
                if dataset == 'reddit':
                    tgnne_result_batches = [0]
                    sa_result_batches = [0]
                    temp_me_result_batches = [0]
                else:
                    tgnne_result_batches = [0,1]
                    sa_result_batches = [0,1]
                    temp_me_result_batches = [0,1]

            elif model == 'tgat':
                if dataset == 'reddit':
                    tgnne_result_batches = [0]
                    sa_result_batches = [0]
                else:
                    tgnne_result_batches = [0]
                    sa_result_batches = [0]
                    temp_me_result_batches = [0]

            for i in sa_result_batches:
                with open(results_dir+f'sa_results_{dataset}_{model}_exp_sizes_{i}.pkl', 'rb') as file:
                    sa_results = pck.load(file)
                    sub_keys = list(sa_results.values())[0].keys()
                    if combined_sa_results is None:
                        combined_sa_results = {k:{k_: [] for k_ in sub_keys} for k in sa_results.keys()}
                    for k in sa_results.keys():
                        for k_ in sub_keys:
                            combined_sa_results[k][k_].extend(sa_results[k][k_])
            sa_results = combined_sa_results

#for i in [1,2,3,4,5,15,17,20]:
            for i in tgnne_result_batches:
                with open(results_dir+f'tgnne_results_{dataset}_{model}_{i}.pkl', 'rb') as file:
                    tgnne_results = pck.load(file)
                    sub_keys = list(tgnne_results.values())[0].keys()
                    if combined_tgnne_results is None:
                        combined_tgnne_results = {k:{k_: [] for k_ in sub_keys} for k in tgnne_results.keys()}
                    for k in tgnne_results.keys():
                        for k_ in sub_keys:
                            combined_tgnne_results[k][k_].extend(tgnne_results[k][k_])
            tgnne_results = combined_tgnne_results
##    print(tgnne_results)


            if dataset == 'reddit':
#        with open(f'tgnnexplainer/benchmarks/results/temp_me_results_{dataset}_{model}_exp_sizes.pkl', 'rb') as file:
#            temp_me_results = pck.load(file)
                temp_me_results = copy.deepcopy(sa_results)
            else:
                for i in temp_me_result_batches:
                    with open(results_dir+f'temp_me_results_{dataset}_{model}_exp_sizes_{i}.pkl', 'rb') as file:
                        temp_me_results = pck.load(file)
                        sub_keys = list(temp_me_results.values())[0].keys()
                        if combined_temp_me_results is None:
                            combined_temp_me_results = {k:{k_: [] for k_ in sub_keys} for k in temp_me_results.keys()}
                        for k in temp_me_results.keys():
                            for k_ in sub_keys:
                                combined_temp_me_results[k][k_].extend(temp_me_results[k][k_])
            # Get common target_idx results
            for exp_size in sa_results.keys():
                all_sa_target_idx = sa_results[exp_size]['target_event_idxs']
                all_tgnne_target_idx = tgnne_results[exp_size]['target_event_idxs']
                all_temp_me_target_idx = temp_me_results[exp_size]['target_event_idxs']
                common_target_idx = [target_idx for target_idx in all_sa_target_idx if ((target_idx in all_tgnne_target_idx) and (target_idx in all_temp_me_target_idx))]
#

                sa_results[exp_size]['model_predictions'] = [sa_results[exp_size]['model_predictions'][i] for i, target_idx in enumerate(all_sa_target_idx) if target_idx in common_target_idx]
                sa_results[exp_size]['explanation_predictions'] = [sa_results[exp_size]['explanation_predictions'][i] for i, target_idx in enumerate(all_sa_target_idx) if target_idx in common_target_idx]
                sa_results[exp_size]['delta_fidelity'] = [sa_results[exp_size]['delta_fidelity'][i] for i, target_idx in enumerate(all_sa_target_idx) if target_idx in common_target_idx]
                sa_results[exp_size]['target_event_idxs'] = [target_idx for target_idx in all_sa_target_idx if target_idx in common_target_idx]

                tgnne_results[exp_size]['model_predictions'] = [tgnne_results[exp_size]['model_predictions'][i] for i, target_idx in enumerate(all_tgnne_target_idx) if target_idx in common_target_idx]
                tgnne_results[exp_size]['explanation_predictions'] = [tgnne_results[exp_size]['explanation_predictions'][i] for i, target_idx in enumerate(all_tgnne_target_idx) if target_idx in common_target_idx]
                tgnne_results[exp_size]['target_event_idxs'] = [target_idx for target_idx in all_tgnne_target_idx if target_idx in common_target_idx]
                dfs = []
                for i, target_idx in enumerate(common_target_idx):
                    exp_pred = tgnne_results[exp_size]['explanation_predictions'][i]
                    exp_pred = round(exp_pred, 4)
                    model_pred = tgnne_results[exp_size]['model_predictions'][i]
                    model_pred = round(model_pred, 4)
                    unimportant_pred = tgnne_results[exp_size]['unimportant_predictions'][i]
                    unimportant_pred = round(unimportant_pred, 4)
                    if exp_pred == model_pred:
                        dfs.append(np.inf)
                    else:
                        dfs.append(abs(model_pred - unimportant_pred) / abs(model_pred - exp_pred))
                tgnne_results[exp_size]['delta_fidelity'] = dfs

                temp_me_results[exp_size]['model_predictions'] = [temp_me_results[exp_size]['model_predictions'][i] for i, target_idx in enumerate(all_temp_me_target_idx) if target_idx in common_target_idx]
                temp_me_results[exp_size]['explanation_predictions'] = [temp_me_results[exp_size]['explanation_predictions'][i] for i, target_idx in enumerate(all_temp_me_target_idx) if target_idx in common_target_idx]
                temp_me_results[exp_size]['delta_fidelity'] = [temp_me_results[exp_size]['delta_fidelity'][i] for i, target_idx in enumerate(all_temp_me_target_idx) if target_idx in common_target_idx]
                temp_me_results[exp_size]['target_event_idxs'] = [target_idx for target_idx in all_temp_me_target_idx if target_idx in common_target_idx]

#
##
#
##
#
            colors = ['r', 'g', 'b']
            if dataset == 'reddit':
                ms = 2
            else:
                ms=3
            for i, (result, method) in enumerate(zip([sa_results, tgnne_results, temp_me_results][:ms], methods[:ms])):
                print(f'-------{method} - {model} - {dataset}--------')
                errors = []
                sizes = []
                delta_fidelities = []
                for k in result.keys():
                    maes = [abs(y-y_hat) for y, y_hat in zip(result[k]['model_predictions'], result[k]['explanation_predictions'])]
                    errors.append(np.mean(maes))
                    sizes.append(k)
                    e_delta_fidelities = [s for s in result[k]['delta_fidelity'] if s != np.inf]
#                if i ==0:
#                pprint(list(zip(e_delta_fidelities, maes)))
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

                axes[0][plt_num].plot(sizes, errors, color=colors[i])
                axes[1][plt_num].plot(sizes, delta_fidelities, color=colors[i])
                axes[1][plt_num].set_xlabel('Explanation Size')
                axes[0][plt_num].set_ylabel('MAE')
                axes[1][plt_num].set_ylabel(r'$\Delta$Fidelity')
            plt_num += 1
#    pprint(list(zip(sa_errors, tgnne_errors)))

    axes[0][0].set_title(f'TGN - Wikipedia')
    axes[0][1].set_title(f'TGAT - Wikipedia')
    axes[0][2].set_title(f'TGN - Reddit')
    axes[0][3].set_title(f'TGAT - Reddit')
    for ax in fig.get_axes():
        ax.set_yscale('log')

    fig.tight_layout()
    fig.legend(methods, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))
    fig.savefig(f'Figures/results.pdf', bbox_inches='tight')
#
#exp_sizes_results(dataset='reddit')
#exp_sizes_results('wikipedia')

def sa_gammas():

    fig, axes = plt.subplots(2, 2, figsize=(8, 5))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.3)
    dataset = 'wikipedia'
    model = 'tgn'

    results_batches = [0,1]
    combined_sa_results = None

    for i in results_batches:
        with open(results_dir+f'sa_results_{dataset}_{model}_gammas_{i}.pkl', 'rb') as file:
            sa_results = pck.load(file)
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
            delta_fidelities.append(sa_results[gamma]['delta_fidelity'][r])
        axes[0][1].scatter(sizes, errors, label=round(1-float(gamma),1))
        axes[1][1].scatter(sizes, delta_fidelities)

    delta_fidelities = []
    sizes = []
    errors = []
    max_df = 0
    for gamma in sa_results.keys():
        maes = [abs(y-y_hat) for y, y_hat in zip(sa_results[gamma]['model_predictions'], sa_results[gamma]['explanation_predictions'])]
        errors.append(np.mean(maes))
        sizes = [len(e) for e in sa_results[gamma]['explanations']]
        g_delta_fidelities = [s for s in sa_results[gamma]['delta_fidelity'] if s != np.inf]
        max_df = max(max_df, max(g_delta_fidelities))
        delta_fidelities.append(np.mean(g_delta_fidelities))
        print('Gamma:', gamma, 'Avg Error:', errors[-1], 'Avg Size:', np.mean(sizes), 'Avg Delta Fidelity:', delta_fidelities[-1])
    lamdas = [1- float(gamma) for gamma in sa_results.keys()]
    print(min(errors), max(delta_fidelities))
    axes[0][0].plot(lamdas, errors)
    axes[1][0].plot(lamdas, delta_fidelities)

    for ax in fig.get_axes():
        ax.set_yscale('log')


    axes[0][1].set_xlabel('Explanation Size')
    axes[1][1].set_xlabel('Explanation Size')
    axes[0][1].set_ylabel('MAE')
    axes[1][1].set_ylabel(r'$\Delta$Fidelity')
    axes[0][0].set_ylabel('MAE')
    axes[1][0].set_ylabel(r'$\Delta$Fidelity')

    fig.legend(title=r'$\lambda$', loc='upper center', ncol=len(lamdas), bbox_to_anchor=(0.5, 1.0))

    fig.savefig(f'Figures/{dataset}_lambda_results.pdf', bbox_inches='tight')


sa_gammas()


