from random import random
from typing import Union
from pandas import DataFrame
from pathlib import Path
from sklearn.utils import shuffle
from tqdm import tqdm
import pickle as pck
import numpy as np
import torch
import torch.nn as nn

from src.method.base_explainer_tg import BaseExplainerTG
from src.evaluation.metrics_tg_utils import fidelity_inv_tg
from src.method.tg_score import _set_tgat_data
from src.models.ext.tgat.module import TGAN
from src.models.ext.tgn.model.tgn import TGN


def _create_explainer_input(model: Union[TGAN, TGN], model_name, all_events, candidate_events=None, event_idx=None, device=None):
    # DONE: explainer input should have both the target event and the event that we want to assign a weight to.

    if model_name in ['tgat', 'tgn']:
        print('Event idx: ', event_idx)
        event_idx_u, event_idx_i, event_idx_t = _set_tgat_data(all_events, event_idx)
        # event_idx_new = _set_tgat_events_idxs(event_idx)
        event_idx_new = event_idx
        t_idx_u_emb = model.node_raw_embed[ torch.tensor(event_idx_u, dtype=torch.int64, device=device), : ]
        t_idx_i_emb = model.node_raw_embed[ torch.tensor(event_idx_i, dtype=torch.int64, device=device), : ]
        # import ipdb; ipdb.set_trace()
        t_idx_t_emb = model.time_encoder( torch.tensor(event_idx_t, dtype=torch.float32, device=device).reshape((1, -1)) ).reshape((1, -1))
        t_idx_e_emb = model.edge_raw_embed[ torch.tensor([event_idx_new, ], dtype=torch.int64, device=device), : ]

        target_event_emb = torch.cat([t_idx_u_emb,t_idx_i_emb, t_idx_t_emb, t_idx_e_emb ], dim=1)

        candidate_events_u, candidate_events_i, candidate_events_t = _set_tgat_data(all_events, candidate_events)
        candidate_events_new = candidate_events

        candidate_u_emb = model.node_raw_embed[ torch.tensor(candidate_events_u, dtype=torch.int64, device=device), : ]
        candidate_i_emb = model.node_raw_embed[ torch.tensor(candidate_events_i, dtype=torch.int64, device=device), : ]
        candidate_t_emb = model.time_encoder( torch.tensor(candidate_events_t, dtype=torch.float32, device=device).reshape((1, -1)) ).reshape((len(candidate_events_t), -1))
        candidate_e_emb = model.edge_raw_embed[ torch.tensor(candidate_events_new, dtype=torch.int64, device=device), : ]

        candiadte_events_emb = torch.cat([candidate_u_emb, candidate_i_emb, candidate_t_emb, candidate_e_emb], dim=1)

        input_expl = torch.cat([ target_event_emb.repeat(candiadte_events_emb.shape[0], 1),  candiadte_events_emb], dim=1)
        # import ipdb; ipdb.set_trace()
        print(input_expl.shape, input_expl.device)

        return input_expl

    else:
        raise NotImplementedError


class PGExplainerExt(BaseExplainerTG):
    def __init__(self, model, model_name: str, explainer_name: str, dataset_name: str,
                 all_events: DataFrame,  explanation_level: str, device, verbose: bool = True, results_dir = None, debug_mode=True,
                 # specific params for PGExplainerExt
                 train_epochs: int = 50, explainer_ckpt_dir = None, reg_coefs = None, batch_size = 64, lr=1e-4, exp_size=20,
                ):
        super(PGExplainerExt, self).__init__(model=model,
                                              model_name=model_name,
                                              explainer_name=explainer_name,
                                              dataset_name=dataset_name,
                                              all_events=all_events,
                                              explanation_level=explanation_level,
                                              device=device,
                                              verbose=verbose,
                                              results_dir=results_dir,
                                              debug_mode=debug_mode,
                                              )
        self.train_epochs = train_epochs
        self.explainer_ckpt_dir = explainer_ckpt_dir
        self.reg_coefs = reg_coefs
        self.batch_size = batch_size
        self.lr = lr
        self.expl_input_dim = None
        self.exp_size = exp_size
        self._init_explainer()

    @staticmethod
    def _create_explainer(model, model_name, device):
        if model_name == 'tgat':
            expl_input_dim = model.model_dim * 8 # 2 * (dim_u + dim_i + dim_t + dim_e)
        elif model_name == 'tgn':
            expl_input_dim = model.n_node_features * 8
        else:
            raise NotImplementedError

        explainer_model = nn.Sequential(
            nn.Linear(expl_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  ##### version 1
            # nn.Sigmoid(), ##### version 2
        )
        explainer_model = explainer_model.to(device)
        return explainer_model

    @staticmethod
    def _ckpt_path(ckpt_dir, model_name, dataset_name, explainer_name, epoch=0):
        if epoch is None:
            return Path(ckpt_dir)/f'{model_name}_{dataset_name}_{explainer_name}_expl_ckpt.pt'
        else:
            return Path(ckpt_dir)/f'{model_name}_{dataset_name}_{explainer_name}_expl_ckpt_ep{epoch}.pt'

    def _init_explainer(self):
        self.explainer_model = self._create_explainer(self.model, self.model_name, self.device)

    def __call__(self, node_idxs: Union[int, None] = None, event_idxs: Union[int, None] = None, results_batch= None):
        self.explainer_ckpt_path = self._ckpt_path(self.explainer_ckpt_dir, self.model_name, self.dataset_name, self.explainer_name)
        print(self.results_dir)
        print('Explainer Path: ', self.explainer_ckpt_path)
        self.explain_event_idxs = event_idxs

        if not self.explainer_ckpt_path.exists():
            self._train() # we need to train the explainer first
        else:
            state_dict = torch.load(self.explainer_ckpt_path, map_location=self.device)
            self.explainer_model.load_state_dict(state_dict)

        rb = [str(results_batch) if results_batch is not None else ''][0]
        exp_sizes = [10,20,30,40, 50, 60, 70, 80, 90, 100]
        pg_results = {s: {'target_event_idxs': [], 'explanations': [], 'explanation_predictions': [], 'model_predictions': [], 'delta_fidelity': []} for s in exp_sizes}
        filename = f'/pg_results_{self.dataset_name}_{self.model_name}_{rb}'

        for i, event_idx in enumerate(event_idxs):
            print(f'\nexplain {i}-th: {event_idx}')
            self._initialize(event_idx)
            if len(self.computation_graph_events) <= 100:
                continue
            for exp_size in exp_sizes:
                self.candidate_events = self.computation_graph_events
                exp_events, model_pred, exp_pred, complement_pred = self.explain(event_idx=event_idx, exp_size=exp_size)
                exp_pred = exp_pred.cpu().detach().numpy().flatten()[0]
                model_pred = model_pred.cpu().detach().numpy().flatten()[0]
                complement_pred = complement_pred.cpu().detach().numpy().flatten()[0]
                pg_results[exp_size]['target_event_idxs'].append(event_idx)
                pg_results[exp_size]['explanations'].append(exp_events)
                pg_results[exp_size]['explanation_predictions'].append(exp_pred)
                pg_results[exp_size]['model_predictions'].append(model_pred)

                if abs(exp_pred - model_pred) == 0:
                    delta_fidelity = np.inf
                else:
                    delta_fidelity = abs(complement_pred - model_pred)/abs(exp_pred - model_pred)
                pg_results[exp_size]['delta_fidelity'].append(delta_fidelity)

                print(f'Event idx: {event_idx}, exp_size: {exp_size}, exp_pred: {exp_pred}, model_pred: {model_pred}, complement_pred: {complement_pred}, delta_fidelity: {delta_fidelity}')

        for exp_size in exp_sizes:
            print(f'exp_size: {exp_size}, delta_fidelity: {np.mean(pg_results[exp_size]["delta_fidelity"])}')
            errors = [abs(pg_results[exp_size]['explanation_predictions'][i] - pg_results[exp_size]['model_predictions'][i]) for i in range(len(pg_results[exp_size]['explanation_predictions']))]
            print(f'exp_size: {exp_size}, avg_error: {np.mean(errors)}')

        with open(str(self.results_dir)+f'/{filename}.pkl', 'wb') as f:
            pck.dump(pg_results, f)

        # import ipdb; ipdb.set_trace()
        return pg_results


    def _tg_predict(self, event_idx, use_explainer=False, exp_size=None):
        if self.model_name in ['tgat', 'tgn']:
            src_idx_l, target_idx_l, cut_time_l = _set_tgat_data(self.all_events, event_idx)
            edge_weights = None
            complement_output = None
            if use_explainer:
                # candidate_events_new = _set_tgat_events_idxs(self.candidate_events) # these temporal edges to alter attn weights in tgat
                input_expl = _create_explainer_input(self.model, self.model_name, self.all_events, \
                    candidate_events=self.candidate_events, event_idx=event_idx, device=self.device)
                # import ipdb; ipdb.set_trace()
                edge_weights = self.explainer_model(input_expl)
                candidate_weights_dict = {'candidate_events': torch.tensor(self.candidate_events, dtype=torch.int64, device=self.device),
                                    'edge_weights': edge_weights,
                }
                if exp_size is not None:
                    all_scores = candidate_weights_dict['edge_weights'].cpu().detach().numpy().flatten()
                    sorted_indices = np.argsort(all_scores)
                    best_events = [self.candidate_events[i] for i in sorted_indices[:exp_size]]
                    candidate_weights_dict['candidate_events'] = torch.tensor(best_events, dtype=torch.int64, device=self.device)
                    candidate_weights_dict['edge_weights'] = torch.tensor([all_scores[i] for i in sorted_indices[:exp_size]], dtype=torch.float32, device=self.device)

                    output = self.model.get_prob( src_idx_l, target_idx_l, cut_time_l, logit=True, candidate_weights_dict=candidate_weights_dict, edge_idx_preserve_list=candidate_weights_dict['candidate_events'].detach().cpu().numpy().flatten())

                    remaining_events = [self.candidate_events[i] for i in sorted_indices[exp_size:]]
                    remaining_scores = [all_scores[i] for i in sorted_indices[exp_size:]]
                    remaining_candidates_weights_dict= {'candidate_events': torch.tensor(remaining_events, dtype=torch.int64, device=self.device), 'edge_weights': torch.tensor(remaining_scores, dtype=torch.float32, device=self.device)}
                    complement_output = self.model.get_prob( src_idx_l, target_idx_l, cut_time_l, logit=True, candidate_weights_dict=remaining_candidates_weights_dict, edge_idx_preserve_list=remaining_candidates_weights_dict['candidate_events'].detach().cpu().numpy().flatten())
                    return output, edge_weights, complement_output, candidate_weights_dict['candidate_events'] if use_explainer else None,

            else:
                candidate_weights_dict = None


            # NOTE: use the 'src_ngh_eidx_batch' in module to locate mask fill positions
                output = self.model.get_prob( src_idx_l, target_idx_l, cut_time_l, logit=True, candidate_weights_dict=candidate_weights_dict)

            return output, edge_weights
        else:
            raise NotImplementedError

    def _loss(self, masked_pred, original_pred, mask, reg_coefs):
        # TODO: improve the loss?
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]

        # Regularization losses
        size_loss = torch.sum(mask) * size_reg
        # mask_ent_reg = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        # mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        if original_pred > 0: # larger better
            error_loss = masked_pred - original_pred
        else:
            error_loss = original_pred - masked_pred
        error_loss = error_loss * -1 # to minimize

        # cce_loss = torch.nn.functional.mse_loss(masked_pred, original_pred)

        #return cce_loss
        # return cce_loss + size_loss + mask_ent_loss
        # return cce_loss + size_loss
        # import ipdb; ipdb.set_trace()

        return error_loss

    def _obtain_train_idxs(self,):
        size = 100
        # np.random.seed( np.random.randint(10000) )
        if self.dataset_name in ['wikipedia', 'reddit']:
            train_e_idxs = np.random.randint(int(len(self.all_events)*0.2), int(len(self.all_events)*0.6), (size, ))
            train_e_idxs = shuffle(train_e_idxs) # TODO: not shuffle?
        elif self.dataset_name in ['simulate_v1', 'simulate_v2']:
            positive_indices = self.all_events.label == 1
            pos_events = self.all_events[positive_indices].e_idx.values
            train_e_idxs = np.random.choice(pos_events, size=size, replace=False)

        return train_e_idxs



    def _train(self,):
        self.explainer_model.train()
        optimizer = torch.optim.Adam(self.explainer_model.parameters(), lr=self.lr)

        # train_e_idxs = np.random.randint(int(len(self.all_events)*0.2), int(len(self.all_events)*0.6), (2000, )) # NOTE: set train event idxs


        for e in range(self.train_epochs):
            train_e_idxs = self._obtain_train_idxs()

            optimizer.zero_grad()
            loss = torch.tensor([0], dtype=torch.float32, device=self.device)
            loss_list = []
            counter = 0
            skipped_num = 0

            for i, event_idx in tqdm(enumerate(train_e_idxs), total=len(train_e_idxs), desc=f'epoch {e}' ): # training
                self._initialize(event_idx) # NOTE: needed
                self.candidate_events = self.computation_graph_events
                if len(self.candidate_events) <= 100: # skip bad samples
                    skipped_num += 1
                    continue

                original_pred, mask_values_ = self._tg_predict(event_idx, use_explainer=False)
                masked_pred, mask_values,_, _ = self._tg_predict(event_idx, use_explainer=True, exp_size=100)

                id_loss = self._loss(masked_pred, original_pred, mask_values, self.reg_coefs)
                # import ipdb; ipdb.set_trace()
                id_loss = id_loss.flatten()
                assert len(id_loss) == 1

                loss += id_loss
                loss_list.append(id_loss.cpu().detach().item())
                counter += 1

                if counter % self.batch_size == 0:
                    loss = loss/self.batch_size
                    loss.backward()
                    optimizer.step()
                    loss = torch.tensor([0], dtype=torch.float32, device=self.device)
                    optimizer.zero_grad()

            # import ipdb; ipdb.set_trace()
            state_dict = self.explainer_model.state_dict()
            ckpt_save_path = self._ckpt_path(self.explainer_ckpt_dir, self.model_name, self.dataset_name, self.explainer_name, epoch=e)
            torch.save(state_dict, ckpt_save_path)
            tqdm.write(f"epoch {e} loss epoch {np.mean(loss_list)}, skipped: {skipped_num}, ckpt saved: {ckpt_save_path}")

        state_dict = self.explainer_model.state_dict()
        torch.save(state_dict, self.explainer_ckpt_path)
        print('train finished')
        print(f'explainer ckpt saved at {str(self.explainer_ckpt_path)}')

    def explain(self, node_idx=None, event_idx=None, exp_size=None):
        self.explainer_model.eval()
        input_expl = _create_explainer_input(self.model, self.model_name, self.all_events, \
            candidate_events=self.candidate_events, event_idx=event_idx, device=self.device)
        event_idx_scores = self.explainer_model(input_expl) # compute importance scores
        event_idx_scores = event_idx_scores.cpu().detach().numpy().flatten()

        # the same as Attn explainer
        candidate_weights = { self.candidate_events[i]: event_idx_scores[i] for i in range(len(self.candidate_events)) }
        candidate_weights = dict( sorted(candidate_weights.items(), key=lambda x: x[1], reverse=True) ) # NOTE: descending, important
        model_pred, _ = self._tg_predict(event_idx, use_explainer=False)
        exp_pred,edge_weights, complement_pred, exp_events = self._tg_predict(event_idx, use_explainer=True, exp_size=exp_size)
#        print(f'event_idx: {event_idx}, candidate_weights: {candidate_weights}')

        return exp_events, model_pred, exp_pred, complement_pred

    @staticmethod
    def expose_explainer_model(model, model_name, explainer_name, dataset_name, ckpt_dir, device):
        explainer_model = PGExplainerExt._create_explainer(model, model_name, device)
        explainer_ckpt_path = PGExplainerExt._ckpt_path(ckpt_dir, model_name, dataset_name, explainer_name)

        state_dict = torch.load(explainer_ckpt_path, map_location=device)
        explainer_model.load_state_dict(state_dict)

        return explainer_model, explainer_ckpt_path

class PBOneExplainerTG(BaseExplainerTG):
    """
    perturb only one event to evaluate its influence, then leverage the rank info.
    """
    def __init__(self, model, model_name: str, explainer_name: str, dataset_name: str,
                 all_events: DataFrame,  explanation_level: str, device, verbose: bool = True, results_dir = None, debug_mode=True,
                ):
        super(PBOneExplainerTG, self).__init__(model=model,
                                              model_name=model_name,
                                              explainer_name=explainer_name,
                                              dataset_name=dataset_name,
                                              all_events=all_events,
                                              explanation_level=explanation_level,
                                              device=device,
                                              verbose=verbose,
                                              results_dir=results_dir,
                                              debug_mode=debug_mode,
                                              )
        # assert model_name in ['tgat', 'tgn']


    def _agg_attention(self, atten_weights_list):
        e_idx_weight_dict = {}
        for item in atten_weights_list:
            src_ngh_eidx = item['src_ngh_eidx']
            weights = item['attn_weight'].mean(dim=0)

            src_ngh_eidx = src_ngh_eidx.detach().cpu().numpy().flatten()
            weights = weights.detach().cpu().numpy().flatten()

            for e_idx, w in zip(src_ngh_eidx, weights):
                if e_idx_weight_dict.get(e_idx, None) is None:
                    e_idx_weight_dict[e_idx] = [w,]
                else:
                    e_idx_weight_dict[e_idx].append(w)

        for e_idx in e_idx_weight_dict.keys():
            e_idx_weight_dict[e_idx] = np.mean(e_idx_weight_dict[e_idx])

        return e_idx_weight_dict


    def explain(self, node_idx=None, event_idx=None):
        # perturb each one
        e_idx_logit_dict = {}
        base_events = self.base_events
        for e_idx in self.candidate_events:
            preserved = [idx for idx in self.candidate_events if idx != e_idx]
            total = base_events + preserved
            score = self.tgnn_reward_wraper._compute_gnn_score(total, event_idx)
            e_idx_logit_dict[e_idx] = score

        # import ipdb; ipdb.set_trace()
        # compute an importance score for ranking them
        e_idx_score_dict = {}
        ori_score = self.tgnn_reward_wraper.original_scores
        for e_idx in e_idx_logit_dict.keys():
            # e_idx_score_dict[e_idx] = fidelity_inv_tg(ori_score, e_idx_logit_dict[e_idx]) # the larger the fidelity, the larger the unimportance.
            e_idx_score_dict[e_idx] = fidelity_inv_tg(ori_score, e_idx_logit_dict[e_idx]) * -1 # the larger the score, the larger the importance

        # the same as Attn explainer
        candidate_weights = { e_idx: e_idx_score_dict[e_idx] for e_idx in self.candidate_events }
        candidate_weights = dict( sorted(candidate_weights.items(), key=lambda x: x[1], reverse=True) ) # NOTE: descending, important

        return candidate_weights

    def __call__(self, node_idxs: Union[int, None] = None, event_idxs: Union[int, None] = None):
        results_list = []
        for i, event_idx in enumerate(event_idxs):
            print(f'\nexplain {i}-th: {event_idx}')
            self._initialize(event_idx)

            candidate_weights = self.explain(event_idx=event_idx)
            results_list.append( [ list(candidate_weights.keys()), list(candidate_weights.values()) ] )
            # import ipdb; ipdb.set_trace()
            self._save_candidate_scores(candidate_weights, event_idx)

        return results_list

