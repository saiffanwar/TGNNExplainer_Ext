import numpy as np
import torch
import math
import os
import random
from tqdm import tqdm
import copy
from matplotlib import pyplot as plt
from pprint import pprint
import pickle as pck
from src.method.tg_score import _set_tgat_data
import time

#from visualisation_utils import graph_visualiser

class SimulatedAnnealing:
    def __init__(self, explainer, target_event_idx, candidate_events, exp_size, score_func, expmode='fidelity', verbose=False):
        self.explainer = explainer
        self.expmode = expmode
        self.candidate_events = candidate_events
        self.target_event_idx = target_event_idx
        self.best_events = []
        self.best_score = np.inf
        if self.expmode == 'fidelity':
            self.starting_temperature = 1
        elif self.expmode == 'fidelity+size':
            self.starting_temperature = 1
        self.temperature = self.starting_temperature
        self.cooling_rate = 0.99
        self.exp_size = exp_size
        self.objective_function = score_func
        self.acceptance_probabilities = []
        self.scores = []
        self.actions = []
        self.exp_sizes = []
        self.results_dir = f'{os.getcwd()}/results/{self.explainer.dataset}/'
        self.verbose = verbose

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def reinitialize(self):
        self.best_score = np.inf
        self.best_events = []
        self.expmode = 'fidelity+size'


    def acceptance_probability(self, delta, temperature, alpha=10):
        return np.exp(-alpha*(delta)/temperature)


    def annealing_iteration(self, *args, mode, iteration=0):
        if mode == 'fidelity':
            current_events, current_score, current_absolute_error = args
        elif mode == 'fidelity+size':
            current_events, current_score, current_percentage_error, current_percentage_size, current_absolute_error = args

        new_events = self.perturb(current_events)
        if self.expmode == 'fidelity':
            new_score, new_absolute_error, model_pred, exp_pred = self.objective_function(new_events, self.expmode)
        elif self.expmode == 'fidelity+size':
            new_score, new_percentage_error, new_percentage_size, new_absolute_error, model_pred, exp_pred = self.objective_function(new_events, self.expmode)

        delta = new_score - current_score

        if self.verbose == True:
            if iteration%10==0:
                print('Exp Size: ', len(new_events))
                print("New score: ", new_score)
                print("Current score: ", current_score)
                print("Delta: ", delta)
                print("Temperature: ", self.temperature)
                print('Model pred:', model_pred, 'Exp pred:', exp_pred)
                print('Best score: ', self.best_score, 'Best pred:', self.best_pred)

        if delta > 0:
            current_score = copy.copy(new_score)
            current_events = copy.copy(new_events)
            current_absolute_error = new_absolute_error
            if self.expmode == 'fidelity+size':
                current_percentage_error = new_percentage_error
                current_percentage_size = new_percentage_size
            if new_score > self.best_score:
                self.best_score = copy.copy(new_score)
                self.best_events = copy.copy(new_events)
                self.best_pred = exp_pred
            self.acceptance_probabilities.append(None)
            self.actions.append(True)
        else:
            acceptance_probability = self.acceptance_probability(delta, self.temperature)
            self.acceptance_probabilities.append(acceptance_probability)
            rand_val = np.random.rand()
            accept = rand_val < acceptance_probability
            if accept:
                current_score = copy.copy(new_score)
                current_events = copy.copy(new_events)
                self.actions.append(True)
            else:
                self.actions.append(False)
        self.exp_sizes.append(len(current_events))
        self.temperature *= self.cooling_rate

#        if iteration%50==0:
#        with open(f'{self.results_dir}best_result_{self.target_event_idx}_{self.exp_size}_{self.expmode}.pck.tmp', 'wb') as f:
#            pck.dump((self.explainer, self), f)
        if self.expmode == 'fidelity':
            self.scores.append([current_score, new_score, 0, 0, current_absolute_error])
            return current_events, current_score, current_absolute_error
        elif self.expmode == 'fidelity+size':
            self.scores.append([current_score, new_score, current_percentage_error, current_percentage_size, current_absolute_error])
            return current_events, current_score, current_percentage_error, current_percentage_size, current_absolute_error


    def run(self, iterations=10000, expmode='fidelity', best_events=None):
        self.expmode = expmode
        # Initialize with random events if only optimising fidelity
        if self.exp_size >= len(self.candidate_events):
            self.exp_size = int(len(self.candidate_events))

        if self.expmode == 'fidelity':
            initial_events = list(np.random.choice(self.candidate_events, self.exp_size, replace=False))
            initial_events = [int(e) for e in initial_events]

        elif self.expmode == 'fidelity+size':
#            self.explainer, sa = pck.load(open(f'{self.results_dir}best_result_{self.target_event_idx}_{self.exp_size}_fidelity.pck', 'rb'))
            initial_events = best_events

        if self.expmode == 'fidelity':
            initial_score, initial_absolute_error, self.model_pred, exp_pred = self.objective_function(initial_events, self.expmode)
            self.best_score = initial_score
            self.best_events = initial_events
            self.best_pred = exp_pred
        elif self.expmode == 'fidelity+size':
            initial_score, initial_percentage_error, initial_percentage_size, initial_absolute_error, self.model_pred, exp_pred = self.objective_function(initial_events, self.expmode)
            self.best_score = initial_score
            self.best_pred = exp_pred


        current_events = initial_events
        current_score = initial_score
        current_absolute_error = initial_absolute_error
        if self.expmode == 'fidelity+size':
            current_percentage_error = initial_percentage_error
            current_percentage_size = initial_percentage_size


        if len(self.best_events) != int(len(self.candidate_events)):
            for i in tqdm(range(iterations)):
                if self.expmode == 'fidelity':
                    args = (current_events, current_score, current_absolute_error)
                    current_events, current_score, current_absolute_error = self.annealing_iteration(*args, mode=self.expmode, iteration=i)
                elif self.expmode == 'fidelity+size':
                    args = (current_events, current_score, current_percentage_error, current_percentage_size, current_absolute_error)
                    current_events, current_score, current_percentage_error, current_percentage_size, current_absolute_error = self.annealing_iteration(*args, mode=self.expmode, iteration=i)
#            os.rename(f'{self.results_dir}best_result_{self.target_event_idx}_{self.exp_size}_{self.expmode}.pck.tmp', f'{self.results_dir}best_result_{self.target_event_idx}_{self.exp_size}_{self.expmode}.pck')
        self.best_events = [int(e) for e in self.best_events]



        return self.best_score, self.best_events, self.model_pred, self.best_pred

    def add_event(self, current_events, num_events=2):
        new_events = copy.copy(current_events)
        available_events = list(set(self.candidate_events) - set(new_events))
        new_event = np.random.choice(available_events, num_events, replace=False)
        [new_events.append(e) for e in new_event]
        return new_events

    def remove_event(self, current_events, num_events=1):
        new_events = copy.copy(current_events)
        events_to_remove = np.random.choice(new_events, num_events, replace=False)
        [new_events.remove(e) for e in events_to_remove]
        return new_events

    def replace_event(self, current_events, num_events=2):
        new_events = copy.copy(current_events)
        available_events = list(set(self.candidate_events) - set(new_events))
        if len(available_events) < num_events:
            return new_events
        new_event = np.random.choice(available_events, num_events, replace=False)
        events_to_remove = np.random.choice(new_events, num_events, replace=False)
        for i in range(num_events):
            new_events.remove(events_to_remove[i])
            new_events.append(new_event[i])
#        new_events.remove(events_to_remove)
#        new_events.append(new_event)
        return new_events

    def perturb(self, current_events, num_events=2):
        if self.expmode == 'fidelity':
            return self.replace_event(current_events, num_events)
        elif self.expmode == 'fidelity+size':
            if len(current_events) <= 2:
                move = np.random.choice([self.add_event, self.replace_event])
            else:
                move = np.random.choice([self.add_event, self.remove_event, self.replace_event])
            try:
                return move(current_events)
            except:
                return self.perturb(current_events)

class SA_Explainer:

    def __init__(self, model, tgnnexplainer=None, dataset='wikipedia', model_name='tgat'):
        self.task='traffic_state_pred'
        self.model = model
        self.model_name=model_name
        self.dataset_name=dataset
        self.saved_model=True
        self.train=False
        self.other_args={'exp_id': '1', 'seed': 0}
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.target_timestamp = 0
        self.tgnnexplainer = tgnnexplainer
        self.dataset = dataset
        self.subgraph_size=50

    def delta_fidelity(self, exp_events, target_index):
        target_model_y = self.tgnnexplainer.tgnn_reward_wraper.original_scores
        target_explanation_y = self.tgnnexplainer.tgnn_reward_wraper._get_model_prob(target_index, exp_events, num_neighbors=200)

        remaining_events = list(set(self.tgnnexplainer.computation_graph_events) - set(exp_events))
        remaining_explanation_y = self.tgnnexplainer.tgnn_reward_wraper._get_model_prob(target_index, remaining_events, num_neighbors=200)

        delta_fidelity = abs(remaining_explanation_y - target_model_y) - abs(target_explanation_y - target_model_y)

        return delta_fidelity

    def score_func(self, exp_events, mode='fidelity'):
        '''
        Calculate the fidelity of the model for a given subgraph. Fidelity is defined using the
        metric proposed in arXiv:2306.05760.

        Args:
            subgraph: A list of graphNode() objects that are part of the computation graph.

        Returns:
            fidelity: The fidelity of the model for the given subgraph.
        '''
        target_model_y = self.tgnnexplainer.tgnn_reward_wraper.original_scores
        target_explanation_y = self.tgnnexplainer.tgnn_reward_wraper._get_model_prob(self.target_index, exp_events, num_neighbors=200)
        exp_absolute_error = abs(target_model_y - target_explanation_y)
        max_exp_size = self.subgraph_size
        exp_size_percentage = (100*len(exp_events)/max_exp_size)
        exp_percentage_error = 100*abs(exp_absolute_error/target_model_y)

        if mode == 'fidelity':
#            exp_score = exp_percentage_error
#            exp_score = exp_absolute_error
            exp_score = self.delta_fidelity(exp_events, self.target_index)
            return exp_score, exp_absolute_error, target_model_y, target_explanation_y

        elif mode == 'fidelity+size':
            lam = self.lam
            gam = self.gamma
            exp_score = gam*exp_percentage_error + lam*exp_size_percentage
            return exp_score, gam*exp_percentage_error, lam*exp_size_percentage, exp_absolute_error, target_model_y, target_explanation_y

        else:
            RuntimeError('Invalid mode. Choose either fidelity or fidelity+size')

    def __call__(self, event_idxs, num_iter=500, sa_results=None, results_dir=None, results_batch=None):
        testing_gammas = False
        testing_sparsity = True

        rb = [str(results_batch) if results_batch is not None else ''][0]
        print(f'Running results batch {rb} with {len(event_idxs)} events')

        if testing_gammas:
            gammas = [1.0, 0.99, 0.98, 0.97, 0.96]
            sa_results = {g: {'target_event_idxs': [], 'explanations': [], 'explanation_predictions': [], 'model_predictions': [], 'delta_fidelity': []} for g in gammas}
            filename = f'/sa_results_{self.dataset}_{self.model_name}_gammas_{rb}'
            exp_sizes = [self.subgraph_size]

        elif testing_sparsity:
            exp_sizes = [10,20,30,40, 50, 60, 70, 80, 90, 100]
            sa_results = {s: {'target_event_idxs': [], 'explanations': [], 'explanation_predictions': [], 'model_predictions': [], 'delta_fidelity': []} for s in exp_sizes}
            filename = f'/sa_results_{self.dataset}_{self.model_name}_exp_sizes_{rb}'


        for target_index in event_idxs:
            try:
                print(f'---- Explaining event: {event_idxs.index(target_index)} out of {len(event_idxs)} ------')
                self.target_index = target_index
#            self.tgnnexplainer.model.eval()

                self.tgnnexplainer._initialize(self.target_index)
                self.candidate_events = self.tgnnexplainer.computation_graph_events
                print(len(self.candidate_events))
                original_score = self.tgnnexplainer.tgnn_reward_wraper.compute_original_score(self.candidate_events, self.target_index)

                for exp_size in exp_sizes:
                    sa = SimulatedAnnealing(self, self.target_index, self.candidate_events, exp_size, score_func=self.score_func, verbose=False)
#                sa.reinitialize()
                    score, exp, model_pred, exp_pred = sa.run(iterations=num_iter, expmode='fidelity')
                    delta_fidelity = self.delta_fidelity(exp, self.target_index)
                    print('Score: ', score, 'Exp Length: ', len(exp), 'Model Pred: ', model_pred, 'Exp Pred: ', exp_pred, 'Delta Fidelity: ', delta_fidelity)

                    if not testing_gammas:
                        sa_results[exp_size]['target_event_idxs'].append(target_index)
                        sa_results[exp_size]['explanations'].append(exp)
                        sa_results[exp_size]['explanation_predictions'].append(exp_pred)
                        sa_results[exp_size]['model_predictions'].append(model_pred)
                        sa_results[exp_size]['delta_fidelity'].append(delta_fidelity)

                if testing_sparsity:
                    with open(results_dir + '/intermediate_results/'+filename+'.pkl', 'wb') as f:
                        pck.dump(sa_results, f)

                if testing_gammas:
                    for gamma in gammas:

                        print(f'----- Explaining with Gamma: {gamma} -----')
                        self.gamma=gamma
                        self.lam=1-gamma
                        sa.reinitialize()
                        t_score, t_exp, t_model_pred, t_exp_pred = sa.run(iterations=num_iter, expmode='fidelity+size', best_events=exp)
                        delta_fidelity = self.delta_fidelity(t_exp, self.target_index)
                        print('Score: ', t_score, 'Exp Length: ', len(t_exp), 'Model Pred: ', t_model_pred, 'Exp Pred: ', t_exp_pred, 'Delta Fidelity: ', delta_fidelity)
                        sa_results[gamma]['target_event_idxs'].append(target_index)
                        sa_results[gamma]['explanations'].append(t_exp)
                        sa_results[gamma]['explanation_predictions'].append(t_exp_pred)
                        sa_results[gamma]['model_predictions'].append(t_model_pred)
                        sa_results[gamma]['delta_fidelity'].append(delta_fidelity)

                    with open(results_dir + '/intermediate_results/'+filename+f'_{int(time.time())}.pkl', 'wb') as f:
                        pck.dump(sa_results, f)
            except:
                pass

        with open(results_dir + filename+f'_{int(time.time())}.pkl', 'wb') as f:
            pck.dump(sa_results, f)
        return sa_results
