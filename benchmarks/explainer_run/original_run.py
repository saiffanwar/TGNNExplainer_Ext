import torch
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import multiprocessing as mp
from multiprocessing import Process
import pickle as pck

from src.dataset.tg_dataset import load_tg_dataset, load_explain_idx
from src.dataset.utils_dataset import construct_tgat_neighbor_finder

from src.models.ext.tgat.module import TGAN
from src.models.ext.tgn.model.tgn import TGN
from src.models.ext.tgn.utils.data_processing import compute_time_statistics
from src.method.sa_explainer import SA_Explainer
from src import ROOT_DIR


import warnings
warnings.filterwarnings("ignore")

def start_multi_process(explainer, target_event_idxs, parallel_degree):
    mp.set_start_method('spawn')
    process_list = []
    size = len(target_event_idxs)//parallel_degree
    split = [ i* size for i in range(parallel_degree) ] + [len(target_event_idxs)]
    return_dict = mp.Manager().dict()
    for i in range(parallel_degree):
        p = Process(target=explainer[i], kwargs={ 'event_idxs': target_event_idxs[split[i]:split[i+1]], 'return_dict': return_dict})
        process_list.append(p)
        p.start()

    for p in process_list:
        p.join()

    explain_results = [return_dict[event_idx] for event_idx in target_event_idxs ]
    return explain_results

# def start_multi_process(explainer, target_event_idxs, parallel_degree):
#     mp.set_start_method('spawn')
#     return_dict = mp.Manager().dict()
#     pool = mp.Pool(parallel_degree)
#     for i, e_idx in enumerate(target_event_idxs):
#         pool.apply_async( partial(explainer[i%parallel_degree], event_idxs=[e_idx,], return_dict=return_dict, device=i%4) )

#     pool.close()
#     pool.join()

#     import ipdb; ipdb.set_trace()
#     explain_results = [return_dict[event_idx] for event_idx in target_event_idxs ]
#     return explain_results


@hydra.main(config_path='config', config_name='config')
def pipeline(config: DictConfig):
    # model config
    config.models.param = config.models.param[config.datasets.dataset_name]
    config.models.ckpt_path = str(ROOT_DIR/'models'/'checkpoints'/f'{config.models.model_name}_{config.datasets.dataset_name}_best.pth')

    # dataset config
    config.datasets.dataset_path = str(ROOT_DIR/'dataset'/'data'/f'{config.datasets.dataset_name}.csv')
    config.datasets.explain_idx_filepath = str(ROOT_DIR/'dataset'/'explain_index'/f'{config.datasets.explain_idx_filename}.csv')

    # explainer config
    config.explainers.param = config.explainers.param[config.datasets.dataset_name]
    config.explainers.results_dir = str(ROOT_DIR.parent/'benchmarks'/'results')
    config.explainers.mcts_saved_dir = str(ROOT_DIR/'saved_mcts_results')
    config.explainers.explainer_ckpt_dir = str(ROOT_DIR/'explainer_ckpts')

    # import ipdb; ipdb.set_trace()

    if torch.cuda.is_available() and config.explainers.use_gpu:
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')

    # DONE: only use tgat processed data
    events, edge_feats, node_feats = load_tg_dataset(config.datasets.dataset_name)
    target_event_idxs = load_explain_idx(config.datasets.explain_idx_filepath, start=0)
    ngh_finder = construct_tgat_neighbor_finder(events)

    if config.models.model_name == 'tgat':
        model = TGAN(ngh_finder, node_feats, edge_feats,
                     device=device,
                     attn_mode=config.models.param.attn_mode,
                     use_time=config.models.param.use_time,
                     agg_method=config.models.param.agg_method,
                     num_layers=config.models.param.num_layers,
                     n_head=config.models.param.num_heads,
                     num_neighbors=config.models.param.num_neighbors,
#                     num_neighbors=50,
                     drop_out=config.models.param.dropout
                     )
    elif config.models.model_name == 'tgn': # DONE: added tgn
        mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = compute_time_statistics(events.u.values, events.i.values, events.ts.values )
        model = TGN(ngh_finder, node_feats, edge_feats,
                    device=device,
                    n_layers=config.models.param.num_layers,
                    n_heads=config.models.param.num_heads,
                    dropout=config.models.param.dropout,
                    use_memory=True, # True
                    forbidden_memory_update=True, # True
                    memory_update_at_start=False, # False
                    message_dimension=config.models.param.message_dimension,
                    memory_dimension=config.models.param.memory_dimension,
                    embedding_module_type='graph_attention', # fix
                    message_function='identity', # fix
                    mean_time_shift_src=mean_time_shift_src,
                    std_time_shift_src=std_time_shift_src,
                    mean_time_shift_dst=mean_time_shift_dst,
                    std_time_shift_dst=std_time_shift_dst,
                    n_neighbors=config.models.param.num_neighbors,
                    aggregator_type='last', # fix
                    memory_updater_type='gru', # fix
                    use_destination_embedding_in_message=False,
                    use_source_embedding_in_message=False,
                    dyrep=False,
                    )
    else:
        raise NotImplementedError('Not supported.')

    # load model checkpoints
    import numpy as np
    state_dict = torch.load(config.models.ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    print('USING EXPLAINER:', config.explainers.explainer_name)
    print('USING MODEL:', config.models.model_name)
    # construct a pg_explainer_tg, the mcts_tg explainer may use it
    if config.explainers.explainer_name in ['tgnnexplainer', 'sa_explainer']: # DONE: test this 'use_pg_explainer'
        print('Initializing TGNNExplainer')
        from src.method.tgnnexplainer import TGNNExplainer
        from src.method.other_baselines_tg import PGExplainerExt
        pg_explainer_model, explainer_ckpt_path = PGExplainerExt.expose_explainer_model(model, # load a trained mlp model
                                model_name=config.models.model_name,
                                explainer_name='pg_explainer_tg', # fixed
                                dataset_name=config.datasets.dataset_name,
                                ckpt_dir=config.explainers.explainer_ckpt_dir,
                                device=device,
                                )
        assert config.explainers.parallel_degree >= 1
        explainer = [TGNNExplainer(model,
                                config.models.model_name,
                                config.explainers.explainer_name,
                                config.datasets.dataset_name,
                                events,
                                config.explainers.param.explanation_level,
                                device=device,
                                results_dir=config.explainers.results_dir,
                                debug_mode=config.explainers.debug_mode,
                                save_results=config.explainers.results_save,
                                mcts_saved_dir=config.explainers.mcts_saved_dir,
                                load_results=config.explainers.load_results,
                                rollout=config.explainers.param.rollout,
                                min_atoms=config.explainers.param.min_atoms,
                                c_puct=config.explainers.param.c_puct,
                                pg_explainer_model=pg_explainer_model if config.explainers.use_pg_explainer else None,
                                pg_positive=config.explainers.pg_positive,
        ) for i in range(config.explainers.parallel_degree)]

    elif config.explainers.explainer_name == 'attn_explainer_tg':
        from src.method.attn_explainer_tg import AttnExplainerTG
        explainer = AttnExplainerTG(
                                model,
                                config.models.model_name,
                                config.explainers.explainer_name,
                                config.datasets.dataset_name,
                                events,
                                config.explainers.param.explanation_level,
                                device=device,
                                results_dir=config.explainers.results_dir,
                                debug_mode=config.explainers.debug_mode,
        )
    elif config.explainers.explainer_name == 'pbone_explainer_tg':
        from src.method.other_baselines_tg import PBOneExplainerTG
        explainer = PBOneExplainerTG(
                                model,
                                config.models.model_name,
                                config.explainers.explainer_name,
                                config.datasets.dataset_name,
                                events,
                                config.explainers.param.explanation_level,
                                device=device,
                                results_dir=config.explainers.results_dir,
                                debug_mode=config.explainers.debug_mode,
        )
    elif config.explainers.explainer_name == 'pg_explainer_tg':
        from src.method.other_baselines_tg import PGExplainerExt
        print('here')
        explainer = PGExplainerExt(
                                model,
                                config.models.model_name,
                                config.explainers.explainer_name,
                                config.datasets.dataset_name,
                                events,
                                config.explainers.param.explanation_level,
                                device=device,
                                results_dir=config.explainers.results_dir,
                                train_epochs=config.explainers.param.train_epochs,
                                explainer_ckpt_dir=config.explainers.explainer_ckpt_dir,
                                reg_coefs=config.explainers.param.reg_coefs,
                                batch_size=config.explainers.param.batch_size,
                                lr=config.explainers.param.lr,
                                debug_mode=config.explainers.debug_mode,
        )



#    print(explainer.computation_graph_events)
    # run the explainer
    target_event_idxs = target_event_idxs[:10]



    start_time = time.time()
    if config.explainers.explainer_name == 'tgnnexplainer' and config.explainers.parallel_degree == 1:
        explainer = explainer[0]
        tgnne_results = {'target_event_idxs': [], 'explanations': [], 'explanation_predictions': [], 'model_predictions': []}
        explain_results = explainer(event_idxs=target_event_idxs, results_dict=tgnne_results)


    elif config.explainers.explainer_name == 'sa_explainer' and config.explainers.parallel_degree == 1:
        print('Running SA explainer')
        explainer = explainer[0]
        results = {'SA': None, 'TGNNE': None}
        sa_results = {'target_event_idxs': [], 'explanations': [], 'explanation_predictions': [], 'model_predictions': []}
        tgnne_results = {'target_event_idxs': [], 'explanations': [], 'explanation_predictions': [], 'model_predictions': []}


        print(model)
        sa_explainer = SA_Explainer(model, tgnnexplainer=explainer, model_name=config.models.model_name)
        sa_results = sa_explainer(target_event_idxs, num_iter=500, sa_results=sa_results, results_dir=config.explainers.results_dir)
#                print(f'Model Prediction: {model_pred} Explanation Prediction: {exp_pred}')
#                print(f'Event {event_idx} Score: {score} Explanation: {exp} Sparsity: {len(exp)}')
#
        explainer.rollout=100
        explain_results, results = explainer(event_idxs=target_event_idxs, results_dict=tgnne_results, results_dir=config.explainers.results_dir)

        results['SA'] = sa_results
        results['TGNNE'] = tgnne_results
        with open(config.explainers.results_dir + f'/results_{len(target_event_idxs)}.pkl', 'wb') as f:
            pck.dump(results, f)

#                print(results)

    elif config.explainers.explainer_name == 'tgnnexplainer' and config.explainers.parallel_degree > 1:
        explain_results = start_multi_process(explainer, target_event_idxs, config.explainers.parallel_degree)
    else:
        explain_results = explainer(event_idxs=target_event_idxs)
    end_time = time.time()
    print(f'runtime: {end_time - start_time:.2f}s')

    # exit(0)

    # compute metric values and save
#    if config.explainers.explainer_name == 'tgnnexplainer' or 'sa_explainer':
#        from src.evaluation.metrics_tg import EvaluatorMCTSTG
#        evaluator = EvaluatorMCTSTG(model_name=config.models.model_name,
#                                    explainer_name=config.explainers.explainer_name,
#                                    dataset_name=config.datasets.dataset_name,
#                                    explainer=explainer[0] if isinstance(explainer, list) else explainer,
#                                    results_dir=config.explainers.results_dir
#                                    )
#    elif config.explainers.explainer_name in ['attn_explainer_tg', 'pbone_explainer_tg', 'pg_explainer_tg']:
#        from src.evaluation.metrics_tg import EvaluatorAttenTG
#        evaluator = EvaluatorAttenTG(model_name=config.models.model_name,
#                                    explainer_name=config.explainers.explainer_name,
#                                    dataset_name=config.datasets.dataset_name,
#                                    explainer=explainer,
#                                    results_dir=config.explainers.results_dir
#                                    ) # DONE: updated
#    else:
#        raise NotImplementedError
#
#    if config.evaluate:
#        evaluator.evaluate(explain_results, target_event_idxs)
#    else:
#        print('no evaluate.')
    # import ipdb; ipdb.set_trace()
    # exit(0)


if __name__ == '__main__':
    pipeline()


