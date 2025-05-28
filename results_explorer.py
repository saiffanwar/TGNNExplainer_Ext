import pickle as pck
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import copy
import seaborn as sns

#data_dir = 'tgnnexplainer/src/dataset/data/'
#
#data = pd.read_csv(data_dir + 'wikipedia.csv')
results_dir = 'tgnnexplainer/benchmarks/results/'
plt.style.use('seaborn-v0_8')

def exp_sizes_results(dataset='reddit'):
    fig, axes = plt.subplots(2, 4, figsize=(14, 4), sharex=True)
    datasets = ['wikipedia', 'reddit']
    models = ['tgn', 'tgat']
    methods = ['STX-Search', 'TGNNExplainer', 'Temp-ME', 'PGExplainer']#
    ax = fig.get_axes()
    plt_num = 0
    for d, dataset in enumerate(datasets):
        for m, model in enumerate(models):
            combined_tgnne_results = None
            combined_sa_results = None
            combined_temp_me_results = None
            combined_pg_results = None
            combined_sa_gammas = None
            sa_gammas_result_batches = [0]
            if model == 'tgn':
                if dataset == 'reddit':
                    tgnne_result_batches = [0,1,2,3]
                    sa_result_batches = [0,1,2,3]
                    temp_me_result_batches = [0]
                    pg_result_batches = [0]
                else:
                    tgnne_result_batches = [0,1,2,3]
                    sa_result_batches = [0,1,2,3]
                    temp_me_result_batches = [0]
                    pg_result_batches = [0]

            elif model == 'tgat':
                if dataset == 'reddit':
                    tgnne_result_batches = [0,1,2,3]
                    sa_result_batches = [0,1,2,3]
                    temp_me_result_batches = [0]
                    pg_result_batches = [0]
                else:
                    tgnne_result_batches = [0,1,2,3]
                    sa_result_batches = [0,1,2,3]
                    temp_me_result_batches = [0]
                    pg_result_batches = [0]

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

            for i in temp_me_result_batches:
                with open(results_dir+f'temp_me_results_{dataset}_{model}_exp_sizes_{i}.pkl', 'rb') as file:
                    temp_me_results = pck.load(file)
                    sub_keys = list(temp_me_results.values())[0].keys()
                    if combined_temp_me_results is None:
                        combined_temp_me_results = {k:{k_: [] for k_ in sub_keys} for k in temp_me_results.keys()}
                    for k in temp_me_results.keys():
                        for k_ in sub_keys:
                            combined_temp_me_results[k][k_].extend(temp_me_results[k][k_])
            temp_me_results = combined_temp_me_results

            for i in pg_result_batches:
                with open(results_dir+f'pg_results_{dataset}_{model}_{i}.pkl', 'rb') as file:
                    pg_results = pck.load(file)
                    sub_keys = list(pg_results.values())[0].keys()
                    if combined_pg_results is None:
                        combined_pg_results = {k:{k_: [] for k_ in sub_keys} for k in pg_results.keys()}
                    for k in pg_results.keys():
                        for k_ in sub_keys:
                            combined_pg_results[k][k_].extend(pg_results[k][k_])
            pg_results = combined_pg_results


            combined_sa_gammas = None
            for i in sa_gammas_result_batches:
                with open(results_dir+f'sa_results_{dataset}_{model}_gammas_0.pkl', 'rb') as file:
                    sa_gammas = pck.load(file)
                    print(sa_gammas.keys())
                    sub_keys = list(sa_gammas.values())[0].keys()
                    if combined_sa_gammas is None:
                        combined_sa_gammas = {k:{k_: [] for k_ in sub_keys} for k in sa_gammas.keys()}
                    for k in sa_gammas.keys():
                        for k_ in sub_keys:
                            combined_sa_gammas[k][k_].extend(sa_gammas[k][k_])




            # Get common target_idx results
#            for exp_size in sa_results.keys():
#                all_sa_target_idx = sa_results[exp_size]['target_event_idxs']
#                all_tgnne_target_idx = tgnne_results[exp_size]['target_event_idxs']
#                all_temp_me_target_idx = temp_me_results[exp_size]['target_event_idxs']
#                common_target_idx = [target_idx for target_idx in all_sa_target_idx if ((target_idx in all_tgnne_target_idx) and (target_idx in all_temp_me_target_idx))]
##
#
#                sa_results[exp_size]['model_predictions'] = [sa_results[exp_size]['model_predictions'][i] for i, target_idx in enumerate(all_sa_target_idx) if target_idx in common_target_idx]
#                sa_results[exp_size]['explanation_predictions'] = [sa_results[exp_size]['explanation_predictions'][i] for i, target_idx in enumerate(all_sa_target_idx) if target_idx in common_target_idx]
#                sa_results[exp_size]['delta_fidelity'] = [sa_results[exp_size]['delta_fidelity'][i] for i, target_idx in enumerate(all_sa_target_idx) if target_idx in common_target_idx]
#                sa_results[exp_size]['target_event_idxs'] = [target_idx for target_idx in all_sa_target_idx if target_idx in common_target_idx]
#
#                tgnne_results[exp_size]['model_predictions'] = [tgnne_results[exp_size]['model_predictions'][i] for i, target_idx in enumerate(all_tgnne_target_idx) if target_idx in common_target_idx]
#                tgnne_results[exp_size]['explanation_predictions'] = [tgnne_results[exp_size]['explanation_predictions'][i] for i, target_idx in enumerate(all_tgnne_target_idx) if target_idx in common_target_idx]
#                tgnne_results[exp_size]['target_event_idxs'] = [target_idx for target_idx in all_tgnne_target_idx if target_idx in common_target_idx]
#                dfs = []
#                for i, target_idx in enumerate(common_target_idx):
#                    exp_pred = tgnne_results[exp_size]['explanation_predictions'][i]
#                    exp_pred = round(exp_pred, 4)
#                    model_pred = tgnne_results[exp_size]['model_predictions'][i]
#                    model_pred = round(model_pred, 4)
#                    unimportant_pred = tgnne_results[exp_size]['unimportant_predictions'][i]
#                    unimportant_pred = round(unimportant_pred, 4)
#                    if exp_pred == model_pred:
#                        dfs.append(np.inf)
#                    else:
#                        dfs.append(abs(model_pred - unimportant_pred) / abs(model_pred - exp_pred))
#                tgnne_results[exp_size]['delta_fidelity'] = dfs

#                temp_me_results[exp_size]['model_predictions'] = [temp_me_results[exp_size]['model_predictions'][i] for i, target_idx in enumerate(all_temp_me_target_idx) if target_idx in common_target_idx]
#                temp_me_results[exp_size]['explanation_predictions'] = [temp_me_results[exp_size]['explanation_predictions'][i] for i, target_idx in enumerate(all_temp_me_target_idx) if target_idx in common_target_idx]
#                temp_me_results[exp_size]['delta_fidelity'] = [temp_me_results[exp_size]['delta_fidelity'][i] for i, target_idx in enumerate(all_temp_me_target_idx) if target_idx in common_target_idx]
#                temp_me_results[exp_size]['target_event_idxs'] = [target_idx for target_idx in all_temp_me_target_idx if target_idx in common_target_idx]
#
#
##
#
##
#
            if dataset == 'wikipedia' and model == 'tgn':
                lam = 0.1
            else:
                lam = 0.9
            colors = ['r', 'g', 'b', 'purple']
            if dataset == 'reddit':
                ms = 4
            else:
                ms=4
            sa_gammas_results = {e:[[],[]] for e in sa_results.keys()}
            sa_gammas_exp_sizes = [len(e) for e in sa_gammas[lam]['explanations']]
            sa_gammas_errors = [abs(y-y_hat) for y, y_hat in zip(sa_gammas[lam]['model_predictions'], sa_gammas[lam]['explanation_predictions'])]
            sa_gammas_delta_fidelities = [s for s in sa_gammas[lam]['delta_fidelity']]
#            for size, error, delta_fide in zip(sa_gammas_exp_sizes, sa_gammas_errors, sa_gammas_delta_fidelities):
#                sa_gammas_results[size][0].append(error)
#                if delta_fide != np.inf:
#                    sa_gammas_results[size][1].append(delta_fide)


            sa_g_mean_errors = {k:np.mean(v[0]) for k, v in sa_gammas_results.items()}
            sa_g_mean_delta_fidelities = {k:np.mean(v[1]) for k, v in sa_gammas_results.items()}
            print(f'------- {model} - {dataset} --------')
            print(f'SA-Gamma: --- MAE: {min(sa_gammas_errors):.7f} ({np.mean(sa_gammas_exp_sizes)}), Delta Fidelity: {max(sa_gammas_delta_fidelities):.1f}')


            for i, (result, method) in enumerate(zip([sa_results, tgnne_results, temp_me_results, pg_results][:ms], methods[:ms])):
#                print(f'-------{method} - {model} - {dataset}--------')
                min_error = np.inf
                max_df = 0
                errors = []
                sizes = []
                sa_gammas_errors = []
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

                print(f'{method}: ---- MAE: {min(errors):.4f} ({list(result.keys())[errors.index(min(errors))]}), Delta Fidelity: {max(delta_fidelities):.1f} ({list(result.keys())[delta_fidelities.index(max(delta_fidelities))]})')
                print(errors)
                axes[0][plt_num].plot(sizes, errors, color=colors[i], label=method)
                axes[1][plt_num].plot(sizes, delta_fidelities, color=colors[i], label='__nolegend__')

                axes[1][plt_num].set_xlabel('Explanation Size')
                axes[0][plt_num].set_ylabel('MAE')
                axes[1][plt_num].set_ylabel(r'$\alpha$-Fidelity')
#            axes[0][plt_num].plot(sa_gammas_results.keys(), sa_g_mean_errors.values(), color='black', label='__nolegend__')
#            axes[1][plt_num].plot(sa_gammas_results.keys(), sa_g_mean_delta_fidelities.values(), label='SA-Gamma', color='black')


            plt_num += 1
#    pprint(list(zip(sa_errors, tgnne_errors)))

    axes[0][0].set_title(f'TGN - Wikipedia')
    axes[0][1].set_title(f'TGAT - Wikipedia')
    axes[0][2].set_title(f'TGN - Reddit')
    axes[0][3].set_title(f'TGAT - Reddit')
    for ax in fig.get_axes():
        ax.set_yscale('log')

    fig.tight_layout()
    fig.legend(methods, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.05))
    fig.savefig(f'Figures/results.pdf', bbox_inches='tight')
#
#exp_sizes_results(dataset='reddit')
#exp_sizes_results('wikipedia')

def sa_gammas():


    results_batches = [0]
    for dataset in ['wikipedia']:
        for model in ['tgn']:
            combined_sa_results = None
            fig, axes = plt.subplots(2, 2, figsize=(9, 5))
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.3)
            for i in results_batches:
                print(f'-------{dataset} - {model}--------')
                with open(results_dir+f'sa_results_{dataset}_{model}_gammas_{i}.pkl', 'rb') as file:
                    sa_results = pck.load(file)
                    sub_keys = list(sa_results.values())[0].keys()
                    if combined_sa_results is None:
                        combined_sa_results = {k:{k_: [] for k_ in sub_keys} for k in sa_results.keys()}
                    for k in sa_results.keys():
                        for k_ in sub_keys:
                            combined_sa_results[k][k_].extend(sa_results[k][k_])
            sa_results = combined_sa_results
            print(sa_results[0.1]['delta_fidelity'])

            colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
            for g_num, gamma in enumerate(sa_results.keys()):
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
                axes[0][1].errorbar(np.mean(sizes), np.mean(errors), xerr=np.std(sizes), fmt='o', color=colours[g_num])
                axes[1][1].errorbar(np.mean(sizes), np.mean(delta_fidelities), xerr=np.std(sizes), fmt='o', color=colours[g_num])
#                axes[0][1].scatter(sizes, errors, label=round(1-float(gamma),1), alpha=0.3, color=colours[g_num])
#                axes[1][1].scatter(sizes, delta_fidelities, alpha=0.3, color=colours[g_num])

            delta_fidelities = []
            all_sizes = []
            errors = []
            max_df = 0
            for g_num, gamma in enumerate(sa_results.keys()):
                maes = [abs(y-y_hat) for y, y_hat in zip(sa_results[gamma]['model_predictions'], sa_results[gamma]['explanation_predictions'])]
                errors.append(np.mean(maes))
                sizes = [len(e) for e in sa_results[gamma]['explanations']]
                g_delta_fidelities = [s for s in sa_results[gamma]['delta_fidelity'] if s != np.inf]
                max_df = max(max_df, max(g_delta_fidelities))
                delta_fidelities.append(np.mean(g_delta_fidelities))
                print('Gamma:', gamma, 'Avg Error:', errors[-1], 'Avg Size:', np.mean(sizes), 'Avg Delta Fidelity:', delta_fidelities[-1])
#                axes[1][0].plot(lamdas, delta_fidelities)
                all_sizes.append(np.mean(sizes))
                axes[1][1].scatter(all_sizes[-1], delta_fidelities[-1], s=100, color=colours[g_num], label=round(1-float(gamma),1))
                axes[0][1].scatter(all_sizes[-1], errors[-1], s=100, color=colours[g_num], label='__nolegend__')
            lamdas = [1- float(gamma) for gamma in sa_results.keys()]
            print(min(errors), max(delta_fidelities), all_sizes)
            axes[0][0].plot(lamdas, errors, c='black', zorder=1,alpha=0.8)
            axes[1][0].plot(lamdas, delta_fidelities, c='black',zorder=1, alpha=0.8)
            axes[0][0].scatter(lamdas, errors, color=colours)
            axes[1][0].scatter(lamdas, delta_fidelities, color=colours)
#            ax2 = axes[1][0].twinx()
#            ax2.plot(lamdas, all_sizes, color='black', linestyle='--')
#            ax2.set_ylabel('Explanation Size')

            for ax in fig.get_axes():
                ax.set_yscale('log')
                ax.grid(axis='y')

#            ax2.set_yscale('linear')

            axes[0][1].set_xlabel('Explanation Size')
            axes[1][1].set_xlabel('Explanation Size')
            axes[0][0].set_xlabel(r'$\lambda$')
            axes[1][0].set_xlabel(r'$\lambda$')
            axes[0][1].set_ylabel('MAE')
            axes[1][1].set_ylabel(r'$\alpha$-Fidelity')
            axes[0][0].set_ylabel('MAE')
            axes[1][0].set_ylabel(r'$\alpha$-Fidelity')

            fig.legend(title=r'$\lambda$', loc='upper center', ncol=len(lamdas)/2, bbox_to_anchor=(0.5, 1.05))

#    plt.tight_layout()
            fig.savefig(f'Figures/{dataset}_{model}_lambda_results.pdf', bbox_inches='tight')


#sa_gammas()


def model_losses():

    fig, axes = plt.subplots(4, 1, figsize=(9, 5))

    plot_num = 0
    for dataset in ['wikipedia', 'reddit']:
        for model in ['tgn']:
            with open(f'losses_{dataset}.pkl', 'rb') as file:
                losses = pck.load(file)
                axes[plot_num].plot(range(len(losses)), losses, label=f'{model} - {dataset}')
            plot_num += 1

    plt.show()


model_losses()

