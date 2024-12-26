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

#from visualisation_utils import graph_visualiser

class SimulatedAnnealing:
    def __init__(self, explainer, target_event_idx, candidate_events, exp_size, score_func, expmode='fidelity'):
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
        self.results_dir = f'results/{self.explainer.dataset}/'
        self.verbose = True

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def acceptance_probability(self, delta, temperature, alpha=10):
        return np.exp(-alpha*(delta)/temperature)


    def annealing_iteration(self, *args, mode):
        if mode == 'fidelity':
            current_events, current_score, current_absolute_error = args
        elif mode == 'fidelity+size':
            current_events, current_score, current_percentage_error, current_percentage_size, current_absolute_error = args

        new_events = self.perturb(current_events)
        if self.expmode == 'fidelity':
            new_score, new_absolute_error = self.objective_function(new_events, self.expmode)
        elif self.expmode == 'fidelity+size':
            new_score, new_percentage_error, new_percentage_size, new_absolute_error = self.objective_function(new_events, self.expmode)

        delta = new_score - current_score

        if self.verbose == True:
            print('Exp Size: ', len(new_events))
            print("New score: ", new_score)
            print("Current score: ", current_score)
            print("Delta: ", delta)
            print("Temperature: ", self.temperature)

        if delta < 0:
            current_score = copy.copy(new_score)
            current_events = copy.copy(new_events)
            current_absolute_error = new_absolute_error
            if self.expmode == 'fidelity+size':
                current_percentage_error = new_percentage_error
                current_percentage_size = new_percentage_size
            if new_score < self.best_score:
                self.best_score = copy.copy(new_score)
                self.best_events = copy.copy(new_events)
            self.acceptance_probabilities.append(None)
            self.actions.append(True)
        else:
            acceptance_probability = self.acceptance_probability(delta, self.temperature)
            self.acceptance_probabilities.append(acceptance_probability)
            rand_val = np.random.rand()
            accept = rand_val < acceptance_probability
            if accept:
                current_score = new_score.copy()
                current_events = new_events.copy()
                self.actions.append(True)
            else:
                self.actions.append(False)
        self.exp_sizes.append(len(current_events))
        self.temperature *= self.cooling_rate

        with open(f'{self.results_dir}best_result_{self.target_event_idx}_{self.exp_size}_{self.expmode}.pck.tmp', 'wb') as f:
            pck.dump((self.explainer, self), f)
        if self.expmode == 'fidelity':
            self.scores.append([current_score, new_score, 0, 0, current_absolute_error])
            return current_events, current_score, current_absolute_error
        elif self.expmode == 'fidelity+size':
            self.scores.append([current_score, new_score, current_percentage_error, current_percentage_size, current_absolute_error])
            return current_events, current_score, current_percentage_error, current_percentage_size, current_absolute_error


    def run(self, iterations=10000):
        # Initialize with random events if only optimising fidelity
        if self.expmode == 'fidelity':
            initial_events = list(np.random.choice(self.candidate_events, self.exp_size, replace=False))
            initial_events = [int(e) for e in initial_events]

        elif self.expmode == 'fidelity+size':
            self.explainer, sa = pck.load(open(f'{self.results_dir}best_result_{self.target_event_idx}_{self.exp_size}_fidelity.pck', 'rb'))
            initial_events = sa.best_events

        if self.expmode == 'fidelity':
            initial_score, initial_absolute_error = self.objective_function(initial_events, self.expmode)
        elif self.expmode == 'fidelity+size':
            initial_score, initial_percentage_error, initial_percentage_size, initial_absolute_error = self.objective_function(initial_events, self.expmode)


        current_events = initial_events
        current_score = initial_score
        current_absolute_error = initial_absolute_error
        if self.expmode == 'fidelity+size':
            current_percentage_error = initial_percentage_error
            current_percentage_size = initial_percentage_size


        for i in tqdm(range(iterations)):
            if self.expmode == 'fidelity':
                args = (current_events, current_score, current_absolute_error)
                current_events, current_score, current_absolute_error = self.annealing_iteration(*args, mode=self.expmode)
            elif self.expmode == 'fidelity+size':
                args = (current_events, current_score, current_percentage_error, current_percentage_size, current_absolute_error)
                current_events, current_score, current_percentage_error, current_percentage_size, current_absolute_error = self.annealing_iteration(*args, mode=self.expmode)
            os.rename(f'{self.results_dir}best_result_{self.target_event_idx}_{self.exp_size}_{self.expmode}.pck.tmp', f'{self.results_dir}best_result_{self.target_event_idx}_{self.exp_size}_{self.expmode}.pck')

        return self.best_score, self.best_events

    def add_event(self, current_events, num_events=2):
        new_events = current_events.copy()
        available_events = list(set(self.candidate_events) - set(new_events))
        new_event = np.random.choice(available_events, num_events, replace=False)
        [new_events.append(e) for e in new_event]
        return new_events

    def remove_event(self, current_events, num_events=1):
        new_events = current_events.copy()
        events_to_remove = np.random.choice(new_events, num_events, replace=False)
        [new_events.remove(e) for e in events_to_remove]
        return new_events

    def replace_event(self, current_events, num_events=2):
        new_events = current_events.copy()
        available_events = list(set(self.candidate_events) - set(new_events))
        new_event = np.random.choice(available_events, num_events, replace=False)
        events_to_remove = np.random.choice(new_events, num_events, replace=False)
        for i in range(num_events):
            new_events.remove(events_to_remove[i])
            new_events.append(new_event[i])
#        new_events.remove(events_to_remove)
#        new_events.append(new_event)
        return new_events

    def perturb(self, current_events, num_events=5):
        if self.expmode == 'fidelity':
            return self.replace_event(current_events, num_events)
        elif self.expmode == 'fidelity+size':
            if len(current_events) <= 2:
                move = np.random.choice([self.add_event, self.replace_event])
            else:
                move = np.random.choice([self.add_event, self.remove_event, self.replace_event])
            return move(current_events)


#if __name__ == '__main__':
#    explainer = SA_Explainer(target_node_index=target_node_index)
#    best_exps = []
#    best_scores = []
#
##    with open('results/METR_LA/simulated_annealing/all_results.pck', 'rb') as f:
##        best_exps, best_scores = pck.load(f)
##    best_exps = best_exps[:2]
##    best_scores = best_scores[:2]
#
#    if len(best_exps) == 0:
#        candidate_events = explainer.candidate_events
#    else:
#        candidate_events = best_exps[-1]
#    sa = SimulatedAnnealing(explainer, candidate_events, args.subgraph_size)
#    num_iter=30000
#    score, exp = sa.run(iterations=num_iter)
#    best_scores.append(score)
#    best_exps.append(exp)
##    with open(f'{results_dir}final_result_{num_iter}_iter_{self.explainer.target_index}_{self.exp_size}.pck', 'wb') as f:
##        pck.dump((best_exps, best_scores), f)
#
#    print("Best scores: ", best_scores)

class SA_Explainer:

    def __init__(self, model, target_index=20, tgnnexplainer=None, dataset='wikipedia'):
        self.task='traffic_state_pred'
        self.model = model
        self.model_name='STGCN'
        self.dataset_name='METR_LA'
        self.saved_model=True
        self.train=False
        self.other_args={'exp_id': '1', 'seed': 0}
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.target_index = target_index
        self.target_timestamp = 0
        self.tgnnexplainer = tgnnexplainer
        self.dataset = dataset
        self.subgraph_size=64


#    def generate_graph_nodes(self, x):
#        x = x.cpu()
#        window_size = np.array(x).shape[1]
#        num_nodes = np.array(x).shape[2]
#        nodes = [[] for i in range(window_size)]
#
#        for t in range(window_size):
#            for n in range(num_nodes):
#                nodes[t].append(graphNode(n, t, x[0][t][n][0]))
#        return nodes

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
        target_explanation_y = self.tgnnexplainer.tgnn_reward_wraper._get_model_prob(exp_events, self.target_index, num_neighbors=600)
        print('target_explanation_y:', target_explanation_y, 'target_model_y:', target_model_y)

        exp_absolute_error = abs(target_model_y - target_explanation_y)
        max_exp_size = self.subgraph_size
        exp_size_percentage = (100*len(exp_events)/max_exp_size)
        exp_percentage_error = 100*(exp_absolute_error/target_model_y)

        if mode == 'fidelity':
            exp_score = exp_percentage_error
            return exp_score, exp_absolute_error

        elif mode == 'fidelity+size':
            lam = 0.1
            gam = 0.9
            exp_score = gam*exp_percentage_error + lam*exp_size_percentage
            return exp_score, gam*exp_percentage_error, lam*exp_size_percentage, exp_absolute_error

        else:
            RuntimeError('Invalid mode. Choose either fidelity or fidelity+size')

    def __call__(self, ):

        self.tgnnexplainer._initialize(self.target_index)
        self.candidate_events = self.tgnnexplainer.computation_graph_events
        print(sorted(self.candidate_events), self.target_index)
        original_reward = self.tgnnexplainer.tgnn_reward_wraper.compute_original_score(self.candidate_events, self.target_index)
        print('original_reward:', original_reward)

        print('found candidate events:', len(self.candidate_events))
        sa = SimulatedAnnealing(self, self.target_index, self.candidate_events, self.subgraph_size, score_func=self.score_func)
        num_iter=300
        score, exp = sa.run(iterations=num_iter)
        return score, exp

#def batch_to_cpu(batch):
#    for key in batch.data:
#        batch.data[key] = batch.data[key].to('cpu')
#
#def run_explainer(target_node_index):
#
#    data_feature, train_data, valid_data, test_data = explainer.load_data()
#    explainer.scaler = data_feature['scaler']
#
#    model_path = os.getcwd()+'/libcity/cache/1/model_cache/STGCN_METR_LA.m'
#    explainer.model = explainer.load_model(model_path, data_feature)
#    adj_mx = data_feature['adj_mx']
#    explainer.adj_mx = adj_mx
#
#
#    with torch.no_grad():
#        explainer.model.eval()
#        batch = next(iter(test_data))
##            print(torch.tensor(batch['X']).shape)
##            print(dir(batch))
#        batch['X'] = batch['X'][:1]
#        batch['y'] = batch['y'][:1]
#
#
#        batch.to_tensor(explainer.device)
#        explainer.model_y = explainer.model.predict(batch)
#        explainer.model_y.cpu()
#
#        batch_to_cpu(batch)
#
#        explainer.all_nodes = explainer.generate_graph_nodes(batch['X'])
#
#        explainer.target_node = graphNode(explainer.target_index, explainer.target_timestamp, batch['y'][0][explainer.target_timestamp][explainer.target_index][0])
#
#        if args.mode == 'generate':
#            explainer.candidate_events = explainer.fetch_computation_graph(batch['X'], adj_mx, explainer.target_node)
#            subgraph_sizes = [5,10,25,50,100,int(np.floor(len(explainer.candidate_events)/4)),int(np.floor(len(explainer.candidate_events)/2)),len(explainer.candidate_events)]
#            subgraph_size = 50
##            random.seed(0)
#            subgraph = random.sample(explainer.candidate_events, subgraph_size)
#            explainer.batch = batch
#            return explainer
#
#        elif args.mode == 'visualise':
#
#            masked_input, masked_adj_mx = explainer.create_masked_input(exp_subgraph, batch['X'], explainer.adj_mx)
#            masked_batch = deepcopy(batch)
#            masked_batch['X'] = masked_input # (1, 12, 207, 1)
#            masked_batch.to_tensor(explainer.device)
#            exp_y = explainer.model.predict(masked_batch) # (1, 12, 207, 1)
#            ### Inverse scaling of output to get traffic speed values.
#            y_predicted = explainer.scaler.inverse_transform(exp_y[..., :explainer.output_window]) # (1, 12, 207, 1)
#            graph_visualiser(explainer, batch['X'].cpu() ,masked_input, explainer.model_y.cpu(), exp_y.cpu(), adj_mx)

