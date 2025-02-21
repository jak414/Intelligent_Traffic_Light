"""
Agent class

The Agent we're gonna use is the 2nd one, which includes MemoryBuffer_2 and a flat NN. 
We kept also the other option initially suggested: improving the NN with the addition of 2 
convolutional layers. For this reason we still have two agents, two buffers and 2 network configs.
As a matter of fact, only the "_2" versions have been used until now.
"""

import torch as T
import numpy as np
from DQN import DQNetwork, DQNetwork_flat
from replay_memory import MemoryBuffer, MemoryBuffer_2

class agent():
    def __init__(self, gamma=0.99, epsilon=0.1, lr=0.001, n_actions=2, batch_size=128, eps_min=0.01,
                  eps_dec=5e-7, replace=500, mem_size=10000, input_dims=(12,6), chkpt_dir="results"):
            
            self.gamma = gamma
            self.epsilon = epsilon
            self.lr = lr
            self.n_actions = n_actions
            #self.input_dims = input_dims
            self.batch_size = batch_size
            self.eps_min = eps_min
            self.eps_dec = eps_dec
            self.replace_target_cnt = replace
            self.env_name = "crossingEnv"
            self.chkpt_dir = chkpt_dir,

            self.learn_step_counter = 0

            self.memory = MemoryBuffer(mem_size, input_dims)

            self.model = DQNetwork(self.lr, self.n_actions, input_dims = (1,24,6), name=self.env_name+'_model', chkpt_dir=self.chkpt_dir[0])
            self.q_next = DQNetwork(self.lr, self.n_actions, input_dims = (1,24,6), name=self.env_name+'_q_next', chkpt_dir=self.chkpt_dir[0])

            #self.model = DQNetwork_flat(self.lr, self.n_actions, name=self.env_name+"_model", chkpt_dir=self.chkpt_dir[0])
            #self.q_next = DQNetwork_flat(self.lr, self.n_actions,  name=self.env_name+"_model", chkpt_dir=self.chkpt_dir[0])

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            action =  np.random.choice(self.n_actions)
        else:
            #position_matrix, speed_matrix, scalar_features = state
            sclr = np.zeros(3)
            spd, pos, sclr[0], sclr[1], sclr[2] = state
            state_m = T.cat((T.tensor(spd), T.tensor(pos))).reshape(1,1,24,6)
            state_s = T.tensor(sclr, dtype=T.float32).unsqueeze(0)
            q_values = self.model.forward(state_m, state_s)
            action = T.argmax(q_values).item()
            
        return action

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.model.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def sample_memory(self):
        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)

        # CURRENT STATE
        sp = T.tensor([states[i][0] for i in range(len(states))]).unsqueeze(1)
        pos = T.tensor([states[i][1] for i in range(len(states))]).unsqueeze(1)
        # output state matrix (BS x 1 x 24 x 6)
        state_matr = T.cat((sp, pos), dim=2) 
        # output scalars (BS x 3)
        sclr = T.tensor([states[i][2:].astype(np.float32) for i in range(len(states))])

        # NEW STATE
        new_sp = T.tensor([new_states[i][0] for i in range(len(new_states))]).unsqueeze(1)
        new_pos = T.tensor([new_states[i][1] for i in range(len(new_states))]).unsqueeze(1)
        # output new_state matrix (BS x 1 x 24 x 6)
        new_state_matr = T.cat((new_sp, new_pos), dim=2) 
        # output scalars (BS x 3)
        new_sclr = T.tensor([new_states[i][2:].astype(np.float32) for i in range(len(new_states))])

        #states = T.tensor(state).to(self.q_eval.device)
        #new_states = T.tensor(new_state).to(self.q_eval.device)
        actions = T.tensor(actions).to(self.model.device)
        rewards = T.tensor(rewards).to(self.model.device)
        dones = T.tensor(dones).to(self.model.device)

        return state_matr, sclr, actions, rewards, new_state_matr, new_sclr, dones
    
    def train(self):
        if self.memory.mem_count < self.batch_size:
            return

        self.model.optimizer.zero_grad()

        self.replace_target_network()
        
        states_m, states_s, actions, rewards, new_states_m, new_states_s, dones = self.sample_memory()

        indices = np.arange(self.batch_size)
        q_pred = self.model.forward(states_m, states_s)[indices, actions.long()]
        q_next = self.q_next.forward(new_states_m, new_states_s).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next
 
        loss = self.model.loss(q_target, q_pred).to(self.model.device)
        loss.backward()
        self.model.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()



class agent_2():
    def __init__(self, gamma=0.99, epsilon=0.001, lr=0.001, n_actions=2, batch_size=128, eps_min=0.001,
                  eps_dec=5e-7, replace=250, mem_size=10000, input_dims=37, chkpt_dir="results"):
            
            self.gamma = gamma
            self.epsilon = epsilon
            self.lr = lr
            self.n_actions = n_actions
            self.batch_size = batch_size
            self.eps_min = eps_min
            self.eps_dec = eps_dec
            self.replace_target_cnt = replace
            self.env_name = "crossingEnv"
            self.chkpt_dir = chkpt_dir,

            self.learn_step_counter = 0
            self.rplcd = 0

            self.memory = MemoryBuffer_2(mem_size, input_dims)

            self.model = DQNetwork_flat(self.lr, self.n_actions, input_dims, name=self.env_name+'_model', chkpt_dir=self.chkpt_dir[0])
            self.q_next = DQNetwork_flat(self.lr, self.n_actions, input_dims, name=self.env_name+'_q_next', chkpt_dir=self.chkpt_dir[0])

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            action =  np.random.choice(self.n_actions)
        else:
            q_values = self.model.forward(state)
            action = T.argmax(q_values).item()
            
        return action

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0: # or (self.learn_step_counter-5) % self.replace_target_cnt == 0 :
            self.q_next.load_state_dict(self.model.state_dict())
            self.rplcd +=1
            #print('...replaced')
            #try with soft updates? like 95/5

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def sample_memory(self):
        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)

        return states, T.tensor(actions), T.tensor(rewards), new_states, T.tensor(dones)
    
    def train(self, st):
        if self.memory.mem_count < self.batch_size:
            return

        self.model.optimizer.zero_grad()

        self.replace_target_network()
        
        states, actions, rewards, new_states, dones = self.sample_memory()

        indices = np.arange(self.batch_size)
        q_pred = self.model.forward(states)[indices, actions.long()]
        q_next = self.q_next.forward(new_states).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next
 
        loss = self.model.loss(q_target, q_pred).to(self.model.device)
        loss.backward()
        self.model.optimizer.step()
        self.learn_step_counter += st

        self.decrement_epsilon()

