# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code base was developed by Guni Sharon (guni@tamu.edu).
# The PyTorch code was developed by Sheelabhadra Dey (sheelabhadra@tamu.edu).
# The basis for this code base was from CSCE 642 we added an image input using CNN's for Atari environments
import csv
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
import cv2


class QFunction(nn.Module):
    """
    Q-network definition for image inputs using convolutional layers.
    """

    def __init__(self, in_channels, num_actions):
        super(QFunction, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        # x should be of shape [batch_size, channels, height, width]
        x = x / 255.0  # Normalize pixel values to [0, 1]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN(AbstractSolver):
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)
        self.state_shape = (4, 84, 84)  # For stacking 4 frames
        self.frame_stack = deque(maxlen=4)
        act_dim = env.action_space.n

        in_channels = self.state_shape[0]  # 4 stacked frames

        # Initialize Q-network and target network
        self.model = QFunction(in_channels, act_dim)
        self.target_model = deepcopy(self.model)
        self.target_model.eval()  # Set target model to evaluation mode

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.options.alpha,
            amsgrad=True,
        )

        # Loss function
        self.loss_fn = nn.SmoothL1Loss()

        # Freeze target network parameters
        for p in self.target_model.parameters():
            p.requires_grad = False

        # Replay buffer
        self.replay_memory = deque(maxlen=options.replay_memory_size)

        # Steps counter
        self.n_steps = 0

        # Initialize device (MPS for Mac GPUs)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Move models to device
        self.model.to(self.device)
        self.target_model.to(self.device)

    def preprocess(self, state):
        # Convert to grayscale and resize
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, (84, 84))
        state = state.astype(np.float32)
        return state

    def stack_frames(self, state, is_new_episode):
        processed_state = self.preprocess(state)
        if is_new_episode:
            self.frame_stack.clear()
            for _ in range(4):
                self.frame_stack.append(processed_state)
        else:
            self.frame_stack.append(processed_state)
        # Stack frames along axis 0 (channels)
        stacked_state = np.stack(self.frame_stack, axis=0)
        return stacked_state  # Shape: [channels, height, width]

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.load_state_dict(self.model.state_dict())

    def epsilon_greedy(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)  # Add batch dimension
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

    def memorize(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.replay_memory) > self.options.batch_size:
            minibatch = random.sample(self.replay_memory, self.options.batch_size)
            states, actions, rewards, next_states, dones = zip(*minibatch)

            # Convert lists to numpy arrays
            states = np.stack(states)
            next_states = np.stack(next_states)

            # Convert numpy arrays to torch tensors
            states = torch.as_tensor(states, dtype=torch.float32).to(self.device)
            next_states = torch.as_tensor(next_states, dtype=torch.float32).to(self.device)
            actions = torch.as_tensor(actions, dtype=torch.long).to(self.device)
            rewards = torch.as_tensor(rewards, dtype=torch.float32).to(self.device)
            dones = torch.as_tensor(dones, dtype=torch.float32).to(self.device)

            # Current Q-values
            q_values = self.model(states)
            current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(-1)

            # Compute target Q-values
            target_q = self.compute_target_values(next_states, rewards, dones)

            # Compute loss
            loss = self.loss_fn(current_q, target_q)

            # Optimize the network
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()

    def train_episode(self):
        state, _ = self.env.reset()
        state = self.stack_frames(state, is_new_episode=True)
        total_reward = 0
        total_steps = 0
        rows = []

        while True:
            action_probs = self.epsilon_greedy(state)
            action = np.random.choice(len(action_probs), p=action_probs)
            next_state, reward, done, _ = self.step(action)
            next_state = self.stack_frames(next_state, is_new_episode=False)
            self.memorize(state, action, reward, next_state, done)
            self.replay()
            if self.n_steps % self.options.update_target_estimator_every == 0:
                self.update_target_model()
            self.n_steps += 1
            total_reward += reward
            total_steps += 1
            state = next_state

            rows.append([self.n_steps, action, reward, done])

            # Write rows to file at the end of the episode
            with open('train_logs.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(["Step", "Action", "Reward", "Done"])  # Write header if empty
                writer.writerows(rows)

            if done or total_steps >= self.options.steps:
                break

        # Update statistics
        # self.statistics["Episode Rewards"].append(total_reward)
        # self.statistics["Episode Length"].append(total_steps)

    def run_greedy(self):
        state, _ = self.eval_env.reset()
        eval_frame_stack = deque(maxlen=4)
        state = self.stack_frames(state, is_new_episode=True)
        total_reward = 0
        step = 0
        done = False
        while not done and step < self.options.steps:
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            action = torch.argmax(q_values).item()
            next_state, reward, done, _, _ = self.eval_env.step(action)
            state = self.stack_frames(next_state, is_new_episode=False)
            total_reward += reward
            step += 1
        print(f"Total reward in evaluation: {total_reward}")

    def __str__(self):
        return "DQN Baseline"

    def plot(self, stats, smoothing_window, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)

    def create_greedy_policy(self):
        def policy_fn(state):
            state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.model(state)
            return torch.argmax(q_values).detach().cpu().numpy()

        return policy_fn