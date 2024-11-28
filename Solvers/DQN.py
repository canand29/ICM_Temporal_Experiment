# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code base was developed by Guni Sharon (guni@tamu.edu).
# The PyTorch code was developed by Sheelabhadra Dey (sheelabhadra@tamu.edu).

import random
from copy import deepcopy
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import AdamW
from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting

class ICM(nn.Module):
    """
    Intrinsic Curiosity Module consisting of a forward and inverse model
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes
    ):
        super(ICM, self).__init__()
        self.forward_model = QFunction(obs_dim + act_dim, obs_dim, hidden_sizes)
        self.inverse_model = QFunction(obs_dim*2, act_dim, hidden_sizes)

    def forward(self, state, action, next_state):
        # Forward model (predict next state)
        state_action = torch.cat([state, action], dim=-1)
        predicted_next_state = self.forward_model(state_action)

        # Inverse model (predict action)
        state_next_state = torch.cat([state, next_state], dim=-1)
        predicted_action = self.inverse_model(state_next_state)

        return predicted_action, predicted_next_state
class QFunction(nn.Module):
    """
    Q-network definition.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
    ):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, obs):
        x = torch.cat([obs], dim=-1)
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x).squeeze(dim=-1)


class DQN(AbstractSolver):
    def __init__(self, env, eval_env, options):
        assert str(env.action_space).startswith("Discrete") or str(
            env.action_space
        ).startswith("Tuple(Discrete"), (
            str(self) + " cannot handle non-discrete action spaces"
        )
        super().__init__(env, eval_env, options)
        # Create Q-network
        self.model = QFunction(
            env.observation_space.shape[0],
            env.action_space.n,
            self.options.layers,
        )
        # Create target Q-network
        self.target_model = deepcopy(self.model)
        # Set up the optimizer
        self.optimizer = AdamW(
            self.model.parameters(), lr=self.options.alpha, amsgrad=True
        )
        # Define the loss function
        self.loss_fn = nn.SmoothL1Loss()

        # Freeze target network parameters
        for p in self.target_model.parameters():
            p.requires_grad = False

        # Replay buffer
        self.replay_memory = deque(maxlen=options.replay_memory_size)

        # Number of training steps so far
        self.n_steps = 0

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.load_state_dict(self.model.state_dict())

    def epsilon_greedy(self, state):
        """
        Apply an epsilon-greedy policy based on the given Q-function approximator and epsilon.

        Returns:
            The probabilities (as a Numpy array) associated with each action for 'state'.

        Use:
            self.env.action_space.n: Number of avilable actions
            self.torch.as_tensor(state): Convert Numpy array ('state') to a tensor
            self.model(state): Returns the predicted Q values at a 
                'state' as a tensor. One value per action.
            torch.argmax(values): Returns the index corresponding to the highest value in
                'values' (a tensor)
        """
        # Don't forget to convert the states to torch tensors to pass them through the network.
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        state =  torch.as_tensor(state)
        q_vals = self.model(state)
        best_action = torch.argmax(q_vals).item()
        action_probs = np.ones(self.env.action_space.n) * self.options.epsilon / self.env.action_space.n
        action_probs[best_action] = 1 - self.options.epsilon + (self.options.epsilon / self.env.action_space.n)

        return action_probs

    def compute_target_values(self, next_states, rewards, dones):
        """
        Computes the target q values.

        Returns:
            The target q value (as a tensor) of shape [len(next_states)]
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        max_next_q = self.target_model(next_states).max(dim=1)[0]
        target_q_val = rewards + (1.0 - dones) * self.options.gamma * max_next_q
        return target_q_val


    def replay(self):
        """
        TD learning for q values on past transitions.

        Use:
            self.target_model(state): predicted q values as an array with entry
                per action
        """
        if len(self.replay_memory) > self.options.batch_size:
            minibatch = random.sample(self.replay_memory, self.options.batch_size)
            minibatch = [
                np.array(
                    [
                        transition[idx]
                        for transition, idx in zip(minibatch, [i] * len(minibatch))
                    ]
                )
                for i in range(5)
            ]
            states, actions, rewards, next_states, dones = minibatch
            # Convert numpy arrays to torch tensors
            states = torch.as_tensor(states, dtype=torch.float32)
            actions = torch.as_tensor(actions, dtype=torch.float32)
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
            next_states = torch.as_tensor(next_states, dtype=torch.float32)
            dones = torch.as_tensor(dones, dtype=torch.float32)

            # Current Q-values
            current_q = self.model(states)
            # Q-values for actions in the replay memory
            current_q = torch.gather(
                current_q, dim=1, index=actions.unsqueeze(1).long()
            ).squeeze(-1)

            with torch.no_grad():
                target_q = self.compute_target_values(next_states, rewards, dones)

            # Calculate loss
            loss_q = self.loss_fn(current_q, target_q)

            # Optimize the Q-network
            self.optimizer.zero_grad()
            loss_q.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
            self.optimizer.step()

    def memorize(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def train_episode(self):
        """
        Perform a single episode of the Q-Learning algorithm for off-policy TD
        control using a DNN Function Approximation. Finds the optimal greedy policy
        while following an epsilon-greedy policy.

        Use:
            self.epsilon_greedy(state): return probabilities of actions.
            np.random.choice(array, p=prob): sample an element from 'array' based on their corresponding
                probabilites 'prob'.
            self.memorize(state, action, reward, next_state, done): store the transition in the replay buffer
            self.update_target_model(): copy weights from model to target_model
            self.replay(): TD learning for q values on past transitions
            self.options.update_target_estimator_every: Copy parameters from the Q estimator to the
                target estimator every N steps (HINT: to be done across episodes)
        """

        # Reset the environment
        state, _ = self.env.reset()

        for _ in range(self.options.steps):
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
            action_array = self.epsilon_greedy(state)
            action = np.random.choice(len(action_array), p = action_array)

            next_state, reward, done, _ = self.step(action)
            # if self.n_steps < 20:
            #     print("Action array: ", self.epsilon_greedy(state))
            #     print("Step output:", next_state, reward, done)
            #     print("Action output:", action)
            self.memorize(state, action, reward, next_state, done)
            self.replay()

            if self.n_steps % self.options.update_target_estimator_every == 0:
                self.update_target_model()

            self.n_steps += 1
            state = next_state

            if done:
                break



    def __str__(self):
        return "DQN"

    def plot(self, stats, smoothing_window, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on Q values.


        Returns:
            A function that takes an observation as input and returns a greedy
            action
        """

        def policy_fn(state):
            state = torch.as_tensor(state, dtype=torch.float32)
            q_values = self.model(state)
            return torch.argmax(q_values).detach().numpy()

        return policy_fn


class DQNwithICM(DQN):
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)

        # Create ICM module
        self.icm = ICM(env.observation_space.shape[0], env.action_space.n, self.options.layers)
        self.icm_optimizer = AdamW(self.icm.parameters(), lr=self.options.alpha)

    def compute_intrinsic_reward(self, state, action, next_state):
        """
        Computes the intrinsic curiosity reward based on the prediction error of the forward model.
        """
        # Compute the intrinsic reward based on the forward model
        predicted_next_state, _ = self.icm(state, action, next_state)
        prediction_error = F.mse_loss(predicted_next_state, next_state, reduction='none')
        intrinsic_reward = prediction_error.sum(dim=-1)
        return intrinsic_reward

    def replay(self):
        """
        TD learning for q values on past transitions with intrinsic rewards.
        """
        if len(self.replay_memory) > self.options.batch_size:
            minibatch = random.sample(self.replay_memory, self.options.batch_size)
            minibatch = [
                np.array(
                    [
                        transition[idx]
                        for transition, idx in zip(minibatch, [i] * len(minibatch))
                    ]
                )
                for i in range(5)
            ]
            states, actions, rewards, next_states, dones = minibatch

            # Convert numpy arrays to torch tensors
            states = torch.as_tensor(states, dtype=torch.float32)
            actions = torch.as_tensor(actions, dtype=torch.long)
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
            next_states = torch.as_tensor(next_states, dtype=torch.float32)
            dones = torch.as_tensor(dones, dtype=torch.float32)

            # Compute intrinsic rewards
            intrinsic_rewards = self.compute_intrinsic_reward(states, actions, next_states)

            # Combine intrinsic rewards with extrinsic rewards
            total_rewards = rewards + self.options.intrinsic_weight * intrinsic_rewards

            # Current Q-values
            current_q = self.model(states)
            current_q = torch.gather(current_q, dim=1, index=actions.unsqueeze(1).long()).squeeze(-1)

            # Compute target Q-values
            with torch.no_grad():
                target_q = self.compute_target_values(next_states, total_rewards, dones)

            # Calculate loss
            loss_q = self.loss_fn(current_q, target_q)

            # Optimize the Q-network
            self.optimizer.zero_grad()
            loss_q.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
            self.optimizer.step()

            # Optimize the ICM (forward and inverse models)
            self.icm_optimizer.zero_grad()
            predicted_next_state, predicted_action = self.icm(states, actions, next_states)

            # Forward model loss
            forward_loss = F.mse_loss(predicted_next_state, next_states)

            # Inverse model loss
            inverse_loss = F.cross_entropy(predicted_action, actions)

            # Total ICM loss
            icm_loss = forward_loss + inverse_loss

            # Backpropagate ICM loss
            icm_loss.backward()
            self.icm_optimizer.step()

    def train_episode(self):
        """
        Perform a single episode of the Q-Learning algorithm with ICM for curiosity-driven exploration.
        """
        state, _ = self.env.reset()

        for _ in range(self.options.steps):
            action_array = self.epsilon_greedy(state)
            action = np.random.choice(len(action_array), p=action_array)

            next_state, reward, done, _ = self.step(action)

            # Compute intrinsic reward and store in replay memory
            self.memorize(state, action, reward, next_state, done)

            # Replay and train the model with intrinsic and extrinsic rewards
            self.replay()

            if self.n_steps % self.options.update_target_estimator_every == 0:
                self.update_target_model()

            self.n_steps += 1
            state = next_state

            if done:
                break
