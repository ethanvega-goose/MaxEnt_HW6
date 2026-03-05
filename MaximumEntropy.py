import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from functions.value_iteration import value_iteration

class MaximumEntropy:
    def __init__(self, env, trajectories, features, lr=0.002, discount=0.9, grid_size=4):
        self.env = env
        self.features = features
        self.trajectories = trajectories
        self.rews = []
        self.grid_size = grid_size

        # Initialize reward weights (theta) randomly
        # These weights are mapped to features to calculate state rewards
        self.theta = np.random.uniform(size=(features.shape[1],))
        self.discount = discount
        self.lr = lr

    def get_rewards(self):
        return self.rews

    def expected_features(self):
        """
        Calculates the empirical feature counts from the expert trajectories.
        This represents the 'target' behavior the AI is trying to match.
        """
        exp_f = np.zeros_like(self.features[0])
        for traj in self.trajectories:
            for s, *_ in traj:
                # Accumulate features for every state visited by the expert
                exp_f += self.features[s]
        # Average the counts across all demonstrated trajectories
        return exp_f / self.trajectories.shape[0]

    def expected_state_visitation_frequency(self, policy):
        """
        Predicts how often an agent following the current policy will visit 
        each state. This is the 'Learner's' behavior.
        """
        # Step 1: Calculate the probability of starting in each state
        prob_initial_state = np.zeros(self.env.n_states)
        for traj in self.trajectories:
            prob_initial_state[traj[0, 0]] += 1.0
        prob_initial_state = prob_initial_state / self.trajectories.shape[0]

        # Step 2: Propagate probabilities through time using the dynamics model
        # mu[t, s] is the probability of being in state s at time t
        mu = np.repeat([prob_initial_state], self.trajectories.shape[1], axis=0)
        for t in range(1, self.trajectories.shape[1]):
            mu[t, :] = 0
            for s in range(self.env.n_states):
                for a in range(self.env.n_actions):
                    for s_p in range(self.env.n_states):
                        # sum of (prob of previous state * prob of action * prob of transition)
                        mu[t, s] += mu[t-1, s_p] * policy[s_p, a] * self.env.dynamics[s_p, a, s]
        
        # Sum across all time steps to get the total frequency for each state
        return mu.sum(axis=0)

    def train(self, n_epochs, save_rewards=True, plot=True):
        self.rews = []
        hist_grads = [] # Array to store the gradient norm at each epoch
        exp_f = self.expected_features()

        for i in tqdm(range(n_epochs)):
            rewards = self.features.dot(self.theta)
            if save_rewards:
                self.rews.append(rewards)
                
            policy = value_iteration(self.discount, self.env, rewards)
            exp_svf = self.expected_state_visitation_frequency(policy)

            grads = exp_f - exp_svf @ self.features
            self.theta += self.lr * grads
            
            # Record the magnitude of the difference (the error)
            hist_grads.append(np.linalg.norm(grads)) 

        # Return both the rewards AND the history of the gradients
        return self.features.dot(self.theta), hist_grads