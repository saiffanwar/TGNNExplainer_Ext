import numpy as np
import math
from tqdm import tqdm
import pickle as pck
import time
import os
import argparse


class SimulatedAnnealing:
    def __init__(self, args, explainer, candidate_events, exp_size):
        self.args = args
        self.explainer = explainer
        self.candidate_events = candidate_events
        self.best_events = []
        self.best_score = np.inf
        if self.args.expmode == 'fidelity':
            self.starting_temperature = 1
        elif self.args.expmode == 'fidelity+size':
            self.starting_temperature = 1
        self.temperature = self.starting_temperature
        self.cooling_rate = self.args.cooling_rate
        self.exp_size = exp_size
        self.objective_function = explainer.exp_fidelity
        self.best_exp_graph = None
        self.acceptance_probabilities = []
        self.scores = []
        self.actions = []
        self.exp_sizes = []
        self.results_dir = f'results/{self.args.dataset}/'
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def acceptance_probability(self, delta, temperature, alpha=10):
        return np.exp(-alpha*(delta)/temperature)

    def indices_to_events(self, indices):
        f = lambda idx: self.candidate_events[idx]
        return list(map(f, indices))

    def annealing_iteration(self, *args, mode):
        if mode == 'fidelity':
            current_events, current_score, current_absolute_error = args
        elif mode == 'fidelity+size':
            current_events, current_score, current_percentage_error, current_percentage_size, current_absolute_error = args

        new_events = self.perturb(current_events)
        if self.args.expmode == 'fidelity':
            new_score, new_absolute_error = self.objective_function(self.indices_to_events(new_events), self.args.expmode)
        elif self.args.expmode == 'fidelity+size':
            new_score, new_percentage_error, new_percentage_size, new_absolute_error = self.objective_function(self.indices_to_events(new_events), self.args.expmode)

        delta = new_score - current_score

        if self.args.verbose == True:
            print('Exp Size: ', len(new_events))
            print("New score: ", new_score)
            print("Current score: ", current_score)
            print("Delta: ", delta)
            print("Temperature: ", self.temperature)

        if delta < 0:
            current_score = new_score.copy()
            current_events = new_events.copy()
            current_absolute_error = new_absolute_error
            if self.args.expmode == 'fidelity+size':
                current_percentage_error = new_percentage_error
                current_percentage_size = new_percentage_size
            if new_score < self.best_score:
                self.best_score = new_score.copy()
                self.best_events = new_events.copy()
                self.best_exp_graph = self.explainer.create_masked_input(self.indices_to_events(new_events)).cpu().detach().numpy()
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

        with open(f'{self.results_dir}best_result_{self.explainer.target_index}_{self.exp_size}_{self.args.expmode}.pck.tmp', 'wb') as f:
            pck.dump((self.explainer, self), f)
        if self.args.expmode == 'fidelity':
            self.scores.append([current_score, new_score, 0, 0, current_absolute_error])
            return current_events, current_score, current_absolute_error
        elif self.args.expmode == 'fidelity+size':
            self.scores.append([current_score, new_score, current_percentage_error, current_percentage_size, current_absolute_error])
            return current_events, current_score, current_percentage_error, current_percentage_size, current_absolute_error





    def run(self, iterations=10000):
        self.all_event_indices = list(range(len(self.candidate_events)))
        # Initialize with random events if only optimising fidelity
        if self.args.expmode == 'fidelity':
            initial_events = list(np.random.choice(self.all_event_indices, self.exp_size))
        # Initialise with the exp found by optimising fidelity, now ready to optimise fidelity+size
        elif self.args.expmode == 'fidelity+size':
            self.explainer, sa = pck.load(open(f'{self.results_dir}best_result_{self.explainer.target_index}_{self.exp_size}_fidelity.pck', 'rb'))
            initial_events = sa.best_events

        if self.args.expmode == 'fidelity':
            initial_score, initial_absolute_error = self.objective_function(self.indices_to_events(initial_events), self.args.expmode)
        elif self.args.expmode == 'fidelity+size':
            initial_score, initial_percentage_error, initial_percentage_size, initial_absolute_error = self.objective_function(self.indices_to_events(initial_events), self.args.expmode)

        self.best_exp_graph = self.explainer.create_masked_input(self.indices_to_events(initial_events)).cpu().detach().numpy()

        current_events = initial_events
        current_score = initial_score
        current_absolute_error = initial_absolute_error
        if self.args.expmode == 'fidelity+size':
            current_percentage_error = initial_percentage_error
            current_percentage_size = initial_percentage_size


        for i in tqdm(range(iterations)):
            if self.args.expmode == 'fidelity':
                args = (current_events, current_score, current_absolute_error)
                current_events, current_score, current_absolute_error = self.annealing_iteration(*args, mode=self.args.expmode)
            elif self.args.expmode == 'fidelity+size':
                args = (current_events, current_score, current_percentage_error, current_percentage_size, current_absolute_error)
                current_events, current_score, current_percentage_error, current_percentage_size, current_absolute_error = self.annealing_iteration(*args, mode=self.args.expmode)
            os.rename(f'{self.results_dir}best_result_{self.explainer.target_index}_{self.exp_size}_{self.args.expmode}.pck.tmp', f'{self.results_dir}best_result_{self.explainer.target_index}_{self.exp_size}_{self.args.expmode}.pck')

        return self.best_score, self.indices_to_events(self.best_events)

    def add_event(self, current_events, num_events=2):
        new_events = current_events.copy()
        available_events = list(set(self.all_event_indices) - set(new_events))
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
        available_events = list(set(self.all_event_indices) - set(new_events))
        new_event = np.random.choice(available_events, num_events, replace=False)
        events_to_remove = np.random.choice(new_events, num_events, replace=False)
        for i in range(num_events):
            new_events.remove(events_to_remove[i])
            new_events.append(new_event[i])
#        new_events.remove(events_to_remove)
#        new_events.append(new_event)
        return new_events

    def perturb(self, current_events, num_events=5):
        if self.args.expmode == 'fidelity':
            return self.replace_event(current_events, num_events)
        elif self.args.expmode == 'fidelity+size':
            if len(current_events) <= 2:
                move = np.random.choice([self.add_event, self.replace_event])
            else:
                move = np.random.choice([self.add_event, self.remove_event, self.replace_event])
            return move(current_events)


if __name__ == '__main__':
    explainer = run_explainer(target_index=args.target_node)
    best_exps = []
    best_scores = []

#    with open('results/METR_LA/simulated_annealing/all_results.pck', 'rb') as f:
#        best_exps, best_scores = pck.load(f)
#    best_exps = best_exps[:2]
#    best_scores = best_scores[:2]

    if len(best_exps) == 0:
        candidate_events = explainer.candidate_events
    else:
        candidate_events = best_exps[-1]
    sa = SimulatedAnnealing(explainer, candidate_events, args.subgraph_size)
    num_iter=30000
    score, exp = sa.run(iterations=num_iter)
    best_scores.append(score)
    best_exps.append(exp)
#    with open(f'{results_dir}final_result_{num_iter}_iter_{self.explainer.target_index}_{self.exp_size}.pck', 'wb') as f:
#        pck.dump((best_exps, best_scores), f)

    print("Best scores: ", best_scores)



