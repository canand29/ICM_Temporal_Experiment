# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code base was developed by Guni Sharon (guni@tamu.edu).
# The PyTorch code was developed by Sheelabhadra Dey (sheelabhadra@tamu.edu).
# The logic for the ICM is based on Curiosity-driven Exploration by Self-supervised Prediction. The code is not directly taken from the codebase
# The codebase was based in Python 3.6 and as such was outdated, and we revamped the code to work for Python 3.9
# We also added the temporal feature and normalization as novel features ourselves.
# The timestep addition was also something new as they only had a "done" signal for Mario once he reached the goal or ran out of time but did not have an explicit timestep limit


import csv
import random
from copy import deepcopy
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import AdamW
from Solvers.Abstract_Solver import AbstractSolver, Statistics
from lib import plotting
import cv2


class QFunction(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(QFunction, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        # x should be of shape [batch_size, channels, height, width]
        x = x / 255.0  # Normalize pixel values
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Modified FeatureExtractor to include time
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.fc = nn.Linear(7 * 7 * 32 + 1, feature_dim)  # +1 for time

    def forward(self, x, t):
        x = x / 255.0  # Normalize pixel values
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, t], dim=-1)  # Concatenate time
        x = self.fc(x)  # No activation
        return x


class InverseModel(nn.Module):
    def __init__(self, feature_dim, act_dim):
        super(InverseModel, self).__init__()
        self.fc1 = nn.Linear(2 * feature_dim + 1, 256)  # +1 for time
        self.fc2 = nn.Linear(256, act_dim)

    def forward(self, phi_s, phi_next_s, t):
        x = torch.cat([phi_s, phi_next_s, t], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ForwardModel(nn.Module):
    def __init__(self, feature_dim, act_dim):
        super(ForwardModel, self).__init__()
        self.fc1 = nn.Linear(feature_dim + act_dim + 1, 256)  # +1 for time
        self.fc2 = nn.Linear(256, feature_dim)

    def forward(self, phi_s, a_onehot, t):
        x = torch.cat([phi_s, a_onehot, t], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQN_ICM_Temporal(AbstractSolver):
    def __init__(self, env, eval_env, options):
        # Initialize parent class
        super().__init__(env, eval_env, options)
        self.state_shape = (4, 84, 84)  # For stacking 4 frames
        self.frame_stack = deque(maxlen=4)
        act_dim = env.action_space.n

        in_channels = self.state_shape[0]  # 4 stacked frames
        num_actions = act_dim

        # Initialize Q-network and target network
        self.model = QFunction(
            in_channels,
            num_actions,
        )
        self.target_model = deepcopy(self.model)
        self.target_model.eval()  # Set target model to evaluation mode

        # Initialize ICM components
        self.feature_dim = 256  # Adjust as needed
        self.feature_extractor = FeatureExtractor(
            in_channels, self.feature_dim
        )
        self.inverse_model = InverseModel(
            self.feature_dim, act_dim
        )
        self.forward_model = ForwardModel(
            self.feature_dim, act_dim
        )

        # Initialize optimizer
        self.optimizer = AdamW(
            list(self.model.parameters())
            + list(self.feature_extractor.parameters())
            + list(self.inverse_model.parameters())
            + list(self.forward_model.parameters()),
            lr=self.options.alpha,
            amsgrad=True,
        )

        # Loss functions
        self.loss_fn = nn.SmoothL1Loss()
        self.ce_loss = nn.CrossEntropyLoss()

        # Freeze target network parameters
        for p in self.target_model.parameters():
            p.requires_grad = False

        # Replay buffer
        self.replay_memory = deque(maxlen=options.replay_memory_size)

        # ICM hyperparameters
        self.icm_beta = options.icm_beta
        self.icm_eta = options.icm_eta

        # Steps counter
        self.n_steps = 0

    def preprocess(self, state):
        # Convert to grayscale and resize
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, (84, 84))
        return state

    def stack_frames(self, state, is_new_episode, frame_stack):
        processed_state = self.preprocess(state)
        if is_new_episode:
            frame_stack.clear()
            for _ in range(4):
                frame_stack.append(processed_state)
        else:
            frame_stack.append(processed_state)
        # Stack frames along axis 0 (channels)
        stacked_state = np.stack(frame_stack, axis=0)
        return stacked_state  # Shape: [channels, height, width]

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.load_state_dict(self.model.state_dict())

    def epsilon_greedy(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        q_vals = self.model(state)
        best_action = torch.argmax(q_vals).item()
        action_probs = np.ones(self.env.action_space.n) * self.options.epsilon / self.env.action_space.n
        action_probs[best_action] = 1 - self.options.epsilon + (self.options.epsilon / self.env.action_space.n)
        return action_probs

    def compute_target_values(self, next_states, rewards, dones):
        """
        Computes the target q values.
        """
        with torch.no_grad():
            q_next = self.target_model(next_states)
            max_next_q = q_next.max(dim=1)[0]
            target_q_val = rewards + (1.0 - dones) * self.options.gamma * max_next_q
        return target_q_val

    def memorize(self, state, action, reward, next_state, done, t):
        self.replay_memory.append((state, action, reward, next_state, done, t))

    def replay(self):
        if len(self.replay_memory) > self.options.batch_size:
            minibatch = random.sample(self.replay_memory, self.options.batch_size)
            states, actions, rewards, next_states, dones, ts = zip(*minibatch)

            # Convert lists to numpy arrays
            states = np.stack(states)
            next_states = np.stack(next_states)

            # Convert numpy arrays to torch tensors
            states = torch.as_tensor(states, dtype=torch.float32)
            next_states = torch.as_tensor(next_states, dtype=torch.float32)
            actions = torch.as_tensor(actions, dtype=torch.long)
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
            dones = torch.as_tensor(dones, dtype=torch.float32)
            ts = torch.as_tensor(ts, dtype=torch.float32).unsqueeze(1)  # Shape: [batch_size, 1]

            # Normalize time steps
            max_episode_length = self.options.steps  # Adjust if necessary
            ts_normalized = ts / max_episode_length
            ts_next_normalized = (ts + 1) / max_episode_length

            # Compute features using the feature extractor with time
            phi_s = self.feature_extractor(states, ts_normalized)
            phi_next_s = self.feature_extractor(next_states, ts_next_normalized)

            # One-hot encode actions
            action_onehot = F.one_hot(actions, num_classes=self.env.action_space.n).float()

            # Inverse model prediction
            inv_logits = self.inverse_model(phi_s, phi_next_s, ts_normalized)
            # Inverse loss
            inv_loss = self.ce_loss(inv_logits, actions)

            # Forward model prediction
            pred_phi_next_s = self.forward_model(phi_s, action_onehot, ts_next_normalized)
            # Forward loss
            forward_loss = 0.5 * ((phi_next_s - pred_phi_next_s) ** 2).sum(dim=1)
            forward_loss_mean = forward_loss.mean()

            # Intrinsic reward is the prediction error of the forward model
            intrinsic_reward = forward_loss.detach()

            # Total ICM loss
            icm_loss = self.icm_beta * forward_loss_mean + (1 - self.icm_beta) * inv_loss

            # Scale intrinsic reward and add to extrinsic reward
            total_reward = rewards + self.icm_eta * intrinsic_reward

            # Current Q-values
            q_values = self.model(states)
            current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(-1)

            target_q = self.compute_target_values(next_states, total_reward, dones)

            # Q-learning loss
            loss_q = self.loss_fn(current_q, target_q)

            # Total loss (Q-learning loss + ICM loss)
            total_loss = loss_q + icm_loss

            # Optimize the networks
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()

    def train_episode(self):
        state, _ = self.env.reset()
        state = self.stack_frames(state, is_new_episode=True, frame_stack=self.frame_stack)
        rows = []
        t = 0  # Initialize time step

        for _ in range(self.options.steps):
            action_probs = self.epsilon_greedy(state)
            action = np.random.choice(len(action_probs), p=action_probs)
            next_state, reward, done, _, _ = self.env.step(action)
            next_state = self.stack_frames(next_state, is_new_episode=False, frame_stack=self.frame_stack)
            self.memorize(state, action, reward, next_state, done, t)
            self.replay()
            if self.n_steps % self.options.update_target_estimator_every == 0:
                self.update_target_model()
            self.n_steps += 1
            t += 1  # Increment time step
            state = next_state

            # Append row to memory
            rows.append([self.n_steps, action, reward, done])

            # Write rows to file at the end of the episode
            with open('train_logs.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(["Step", "Action", "Reward", "Done"])  # Write header if empty
                writer.writerows(rows)

            if done:
                break

    def run_greedy(self):
        self.statistics[Statistics.Evaluation.value] += 1
        state, _ = self.eval_env.reset()
        self.eval_frame_stack = deque(maxlen=4)
        state = self.stack_frames(state, is_new_episode=True, frame_stack=self.eval_frame_stack)
        total_reward = 0
        step = 0
        done = False
        while not done and step < self.options.steps:
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.model(state_tensor)
            action = torch.argmax(q_values).item()
            next_state, reward, done, _, _ = self.eval_env.step(action)
            state = self.stack_frames(next_state, is_new_episode=False, frame_stack=self.eval_frame_stack)
            total_reward += reward
            step += 1
        print(f"Total reward in evaluation: {total_reward}")

    def __str__(self):
        return "DQN with ICM Temporal"

    def plot(self, stats, smoothing_window, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)

    def create_greedy_policy(self):
        def policy_fn(state):
            state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.model(state)
            return torch.argmax(q_values).detach().numpy()

        return policy_fn