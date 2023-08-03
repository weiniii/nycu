# Spring 2023, 535515 Reinforcement Learning
# HW2: DDPG

import sys
import gym
import numpy as np
import os
import time
import random
from collections import namedtuple
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Define a tensorboard writer
writer = SummaryWriter("./tb_record_3/no")

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale    

class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Construct your own actor network
        self.ActorNet = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Tanh(),
        )
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data = nn.init.xavier_normal_(m.weight.data)

        ########## END OF YOUR CODE ##########
        
    def forward(self, inputs):

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your actor network
        outputs = self.ActorNet(inputs)
        return outputs * 2
        ########## END OF YOUR CODE ##########

class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        self.action_dim = action_space.shape[0]

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Construct your own critic network

        self.CriticNet = nn.Sequential(
            nn.Linear(num_inputs + self.action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
   
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data = nn.init.xavier_normal_(m.weight.data)

        ########## END OF YOUR CODE ##########

    def forward(self, inputs, actions):
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your critic network

        outputs = self.CriticNet(torch.cat([inputs, actions], 1))
    
        return outputs
        ########## END OF YOUR CODE ##########        
        

class DDPG(object):
    def __init__(self, num_inputs, action_space, gamma=0.995, tau=0.0005, hidden_size=128, lr_a=1e-4, lr_c=1e-3):

        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space)
        # self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_a)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_c)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor) 
        hard_update(self.critic_target, self.critic)


    def select_action(self, state, action_noise=None):
        self.actor.eval()
        mu = self.actor((Variable(state)))
        mu = mu.data

        ########## YOUR CODE HERE (3~5 lines) ##########
        # Add noise to your action for exploration
        # Clipping might be needed 
        if action_noise is not None:
            mu += torch.Tensor(action_noise.noise())
        return torch.clip(mu, -2, 2)

        ########## END OF YOUR CODE ##########


    def update_parameters(self, batch):
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(batch.mask))
        next_state_batch = Variable(torch.cat(batch.next_state))
        
        ########## YOUR CODE HERE (10~20 lines) ##########
        # Calculate policy loss and value loss
        action_next = self.actor_target(next_state_batch)
        value_next = self.critic_target(next_state_batch, action_next)

        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch.unsqueeze(1)

        y = reward_batch + (self.gamma * mask_batch * value_next)
        Q_current = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(y, Q_current)
        
        # Update the actor and the critic
        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()

        policy_action = self.actor(state_batch)
        policy_loss = -torch.mean(self.critic(state_batch, policy_action))

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        ########## END OF YOUR CODE ########## 

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()


    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        local_time = time.localtime()
        timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
        if not os.path.exists('preTrained/'):
            os.makedirs('preTrained/')

        if actor_path is None:
            actor_path = "preTrained/ddpg_actor_{}_{}_{}".format(env_name, timestamp, suffix) 
        if critic_path is None:
            critic_path = "preTrained/ddpg_critic_{}_{}_{}".format(env_name, timestamp, suffix) 
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None: 
            self.critic.load_state_dict(torch.load(critic_path))

def train():    
    num_episodes = 300
    gamma = 0.995
    tau = 0.02
    hidden_size = 256
    noise_scale = 0.3
    replay_size = 100000
    batch_size = 64
    updates_per_step = 1
    print_freq = 10
    ewma_reward = 0
    rewards = []
    ewma_reward_history = []
    total_numsteps = 0
    updates = 0

    
    agent = DDPG(env.observation_space.shape[0], env.action_space, gamma, tau, hidden_size)
    ounoise = OUNoise(env.action_space.shape[0])
    memory = ReplayMemory(replay_size)
    
    for i_episode in range(num_episodes):
        
        ounoise.scale = noise_scale
        ounoise.reset()
        state, _ = env.reset()
        state = torch.Tensor(state).reshape(1, -1)
        episode_reward = 0
        while True:
            
            ########## YOUR CODE HERE (15~25 lines) ##########
            # 1. Interact with the env to get new (s,a,r,s') samples
            # 2. Push the sample to the replay buffer
            # 3. Update the actor and the critic

            action = agent.select_action(state, ounoise)
            next_state, reward, done, truncated, _ = env.step(action.numpy()[0])
            next_state = torch.Tensor(next_state).reshape(1, -1)
            
            memory.push(state, torch.Tensor(action), torch.Tensor([1 - done]), next_state, torch.Tensor([reward]))
                
            try:
                for i in range(updates_per_step):
                    minibatch = memory.sample(batch_size)
                    batch = Transition(*zip(*minibatch))
                    critic_loss, policy_loss = agent.update_parameters(batch)
            except:
                continue

            state = next_state
            if done or truncated:
                break
            ########## END OF YOUR CODE ########## 
            

        # rewards.append(episode_reward)
        t = 0
        if i_episode % print_freq == 0:
            state, _ = env.reset()
            state = torch.Tensor(state).reshape(1, -1)
            episode_reward = 0
            while True:
                action = agent.select_action(state)
                next_state, reward, done, truncated, _ = env.step(action.numpy()[0])
                
                # env.render()
                episode_reward += reward
                next_state = torch.Tensor(next_state).reshape(1, -1)
                state = next_state
                
                t += 1
                if done or truncated:
                    break

            rewards.append(episode_reward)
            # update EWMA reward and log the results
            ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
            ewma_reward_history.append(ewma_reward)           
            
            print("Episode: {}, length: {}, reward: {:.2f}, ewma reward: {}".format(i_episode, t, rewards[-1], ewma_reward))

        writer.add_scalar('Train/reward', ewma_reward, i_episode)
        writer.add_scalar('Train/policy_loss', policy_loss, i_episode)
        writer.add_scalar('Train/critic_loss', critic_loss, i_episode)

    env_name = 'Pendulum-v1'
    agent.save_model(env_name, '.pth')   



    save_as_gif = False
    from matplotlib import animation
    import matplotlib.pyplot as plt
    def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

        #Mess with this to change frame size
        plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
        anim.save(path + filename, writer='imagemagick', fps=60)
        
    if save_as_gif:
        env_test = gym.make('Pendulum-v1', render_mode='rgb_array')
        frames = []
        state, _ = env_test.reset()
        state = torch.Tensor(state).reshape(1, -1)
        episode_reward = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env_test.step(action.numpy()[0])
            
            frames.append(env_test.render())
            episode_reward += reward
            next_state = torch.Tensor(next_state).reshape(1, -1)
            state = next_state
            
            t += 1
            if done or truncated:
                break
        env.close()
        save_frames_as_gif(frames)

if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10
    # env = gym.make('LunarLanderContinuous-v2')
    env = gym.make('Pendulum-v1')
    
    torch.manual_seed(random_seed)  
    train()