"""Buffer"""

import numpy as np

class MemoryBuffer():
    def __init__(self, mem_size, input_shape):
        self.mem_size = mem_size
        self.mem_count = 0
        
        self.state_memory = np.zeros(5, dtype=np.object)
        self.state_memory[0], self.state_memory[1] = np.zeros(input_shape), np.zeros(input_shape)
        self.state_memory= np.array([self.state_memory for _ in range(mem_size)], dtype=object)

        self.action_memory = np.zeros(self.mem_size, dtype=bool)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)

        self.new_state_memory = np.zeros(5, dtype=np.object)
        self.new_state_memory[0], self.new_state_memory[1] = np.zeros(input_shape), np.zeros(input_shape)

        self.new_state_memory= np.array([self.new_state_memory for _ in range(mem_size)], dtype=object)
        
        self.termination_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, obs, action, reward, new_obs, done):
        # when idx gets >mem_size, we start overwriting initial slots
        idx = self.mem_count % self.mem_size
        
        self.state_memory[idx] = obs
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.new_state_memory[idx] = new_obs
        self.termination_memory[idx] = done

        self.mem_count += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_size, self.mem_count)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        dones = self.termination_memory[batch]

        return states, actions, rewards, new_states, dones


class MemoryBuffer_2():
    def __init__(self, mem_size, input_dim=29):
        self.mem_size = mem_size
        self.mem_count = 0
        
        self.state_memory = np.zeros((self.mem_size, input_dim), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dim), dtype=np.float32)
        self.termination_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, obs, action, reward, new_obs, done):
        # when idx gets>mem_size, we start overwriting initial slots
        idx = self.mem_count % self.mem_size
        
        self.state_memory[idx] = obs
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.new_state_memory[idx] = new_obs
        self.termination_memory[idx] = done

        self.mem_count += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_size, self.mem_count)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        dones = self.termination_memory[batch]

        return states, actions, rewards, new_states, dones
