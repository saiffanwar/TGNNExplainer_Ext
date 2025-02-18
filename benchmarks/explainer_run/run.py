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
from src.method.temp_exp_main import load_data, train, TempME_Executor

from src.method.temp_me_utils import RandEdgeSampler, load_subgraph, load_subgraph_margin, get_item, get_item_edge, NeighborFinder
from src import ROOT_DIR
import sys
import h5py
import os.path as osp

import warnings
warnings.filterwarnings("ignore")


degree_dict = {"wikipedia": 20, "reddit": 20, "uci": 30, "mooc": 60, "enron": 30, "canparl": 30, "uslegis": 30}

class ArgDefaults:
    def __init__(self):
        self.gpu = 0
        self.preload = True
        self.base_type = "tgn"
        self.data = "wikipedia"
        self.bs = 100
        self.test_bs = 100
        self.n_degree = 20
        self.n_head = 4
        self.n_epoch = 150
        self.out_dim = 40
        self.hid_dim = 64
        self.temp = 0.07
        self.prior_p = 0.3
        self.lr = 1e-2
        self.drop_out = 0.1
        self.if_attn = True
        self.if_bern = True
        self.save_model = True
        self.test_threshold = False
        self.verbose = 1
        self.weight_decay = 0
        self.beta = 0.5
        self.lr_decay = 0.999
        self.task_type = "temporal explanation"
#except:
#    parser.print_help()
#    sys.exit(0)

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

@hydra.main(config_path='config', config_name='config')
def pipeline(config: DictConfig):
    # model config
    config.models.param = config.models.param[config.datasets.dataset_name]
    config.models.ckpt_path = str(ROOT_DIR/'models'/'checkpoints'/f'{config.models.model_name}_{config.datasets.dataset_name}_best.pth')

    # dataset config
    config.datasets.dataset_path = str(ROOT_DIR/'dataset'/'data'/f'{config.datasets.dataset_name}.csv')
    data_dir = str(ROOT_DIR/'dataset'/'data')
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
                     drop_out=config.models.param.dropout,
                     mode=['temp_me' if config.explainers.explainer_name == 'temp_me' else 'tgnne'][0],
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
                    mode=['temp_me' if config.explainers.explainer_name == 'temp_me' else 'tgnne'][0],
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
                                exp_size=20
        )
    if config.explainers.explainer_name == 'temp_me':

        args = ArgDefaults()
        args.base_type = config.models.model_name
        args.data = config.datasets.dataset_name
        args.device = device
        args.n_degree = degree_dict[args.data]
        args.ratios = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.30]
        args.temp_me_train = True
        args.data_dir = data_dir
        base_model = model

        if args.base_type == "tgn":
            base_model.forbidden_memory_update = True
        pre_load_train = h5py.File(osp.join(data_dir, f'{args.data}_train_cat.h5'), 'r')
        pre_load_test = h5py.File(osp.join(data_dir, f'{args.data}_test_cat.h5'), 'r')

        train_pack = load_subgraph_margin(args, pre_load_train)
        test_pack = load_subgraph_margin(args, pre_load_test)

        train_edge = np.load(osp.join(data_dir, f'{args.data}_train_edge.npy'))
        test_edge = np.load(osp.join(data_dir, f'{args.data}_test_edge.npy'))

        if args.temp_me_train:
#            execute_temp_me_trainer(model, args)
            train(args, base_model, train_pack=train_pack, test_pack=test_pack, train_edge=train_edge, test_edge=test_edge)

        else:
            temp_me_explainer = TempME_Executor(args, model, train_pack, test_pack, train_edge, test_edge, results_dir=config.explainers.results_dir)
#        if args.preload:
#            explainer.load_state_dict(torch.load(f'~/../../tgnnexplainer/src/method/params/explainer/{config.models.model_name}/{config.datasets.dataset_name}.pt'))
#        train(args, base_model, train_pack=train_pack, test_pack=test_pack, train_edge=train_edge, test_edge=test_edge)



    experiment_result_batch = 60
    if config.explainers.explainer_name in ['tgnnexplainer', 'sa_explainer', 'temp_me']:
        target_event_idxs = target_event_idxs[config.results_batch*experiment_result_batch:config.results_batch*experiment_result_batch+experiment_result_batch]
        print(config.results_batch*experiment_result_batch, config.results_batch*experiment_result_batch+experiment_result_batch)
#


    start_time = time.time()
    if config.explainers.explainer_name == 'tgnnexplainer' and config.explainers.parallel_degree == 1:
        explainer = explainer[0]
        explainer.rollout=10
        explainer(event_idxs=target_event_idxs, results_dir=config.explainers.results_dir, results_batch=config.results_batch)

    elif config.explainers.explainer_name == 'temp_me':

        if not args.temp_me_train:
            temp_me_explainer(target_event_idxs=target_event_idxs, results_batch=config.results_batch)

    elif config.explainers.explainer_name == 'sa_explainer' and config.explainers.parallel_degree == 1:
        print('Running SA explainer')
        explainer = explainer[0]

        sa_explainer = SA_Explainer(model, tgnnexplainer=explainer, model_name=config.models.model_name, dataset_name=config.datasets.dataset_name)
        sa_explainer(target_event_idxs, num_iter=500, results_dir=config.explainers.results_dir, results_batch=config.results_batch)
#                print(f'Model Prediction: {model_pred} Explanation Prediction: {exp_pred}')
#                print(f'Event {event_idx} Score: {score} Explanation: {exp} Sparsity: {len(exp)}')
#
#        explainer.rollout=100
#        explain_results, results = explainer(event_idxs=target_event_idxs, results_dict=tgnne_results, results_dir=config.explainers.results_dir)
#
#        results['SA'] = sa_results
#        results['TGNNE'] = tgnne_results
#        with open(config.explainers.results_dir + f'/results_{len(target_event_idxs)}.pkl', 'wb') as f:
#            pck.dump(results, f)

#                print(results)

    elif config.explainers.explainer_name == 'tgnnexplainer' and config.explainers.parallel_degree > 1:
        explain_results = start_multi_process(explainer, target_event_idxs, config.explainers.parallel_degree)
    else:
        tgnne_results = {'target_event_idxs': [], 'explanations': [], 'explanation_predictions': [], 'model_predictions': []}
        explainer(event_idxs=target_event_idxs)
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


