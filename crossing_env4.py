"""Contains the traffic light junction network class."""

import numpy as np
import re
from collections import Counter
import torch as T

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces import Tuple

from flow.core import rewards
from flow.envs.base import Env


ADDITIONAL_ENV_PARAMS = {
    
    "yellow_duration": 5,
    "green_duration":10,

    # whether the traffic lights should be actuated by sumo or RL
    # options are "controlled" and "actuated"
    "tl_type": "controlled",

    "reward_function": "waiting_time",  #or "avg_delay" or "queue_length"

    #factor by which scaling the penalty for changing tl
    #suggested 10 with waiting_time, x with queue_length and x with avg_delay
    "factor" : 0,
    "pedrate_EW" : 300,
    "pedrate_NS" : 500
}

#ADDITIONAL_PO_ENV_PARAMS = {}


class crossingEnv_4(Env):
    """Environment used to train traffic lights.

    Attributes required from env_params:

    * switch_time
    * tl_type
    * discrete/cont ? i think we can keep just discrete
    * inflow prob ? no cause vehs exit the network. capacity we need
    
    Functions required:  
    
    * action_space
    * observation_space
    * apply_rl_action
    * get_state
    * compute_reward

    Notes:

    * States
        An observation could be defined as [poisition, speed].
        Speed is straightforward (get_speed). Position can be defined as:
        - the distance of each vehicle to its intersection +
          a number uniquely identifying which edge the vehicle is on
        - a binary matrix, where rows represents edges>lanes, and 
          columns

    * Actions
        The action space includes whether the traffic light should switch or not.
        In the second case, the current phase is extended by x seconds.

    * Rewards
        The reward fct is split between vehicles and pedestrians:
        Vehicles:
        - Queue based -> minimize queue length 
        - Waiting time based -> minimize w.time (in queue/when v=0)
        - Delay based -> minimize delay ("deviation from max speed")
        Pedestrians:
        - just presence (?)
        Both:
        - Throughput based (# that cleared intersect since last action)
        
        - [additional] penalty for switching traffic lights (?)


    Additional attributes [...]

    """
    
    def __init__(self, env_params, sim_params, network, simulator='traci'):

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        
        self.tl_type = env_params.additional_params.get('tl_type')

        # Saving env variables for plotting
        self.max_steps = env_params.horizon

        # network parameters
        self.len = network.net_params.additional_params["length"]
        self.edges = network.net_params.additional_params["edge_names"]
        self.nlanes = network.net_params.additional_params["nlanes"] 
        self.veh_length = 5.0

        # useful for waiting time (reward fct)
        self.temp_vehs = {}
        self.temp_ped = {}
        self.tot_vehs = {}

        self.last_change = 0. 
        self.green_time = 0.
        self.curr_step = 0

        # when this hits min_switch_time we change from yellow to red
        self.yellow_duration = env_params.additional_params["yellow_duration"]
        self.green_duration = env_params.additional_params["green_duration"]

        # define bounds for the matrix containing the vehicles' info
        self.capacity_row = len(self.edges)*self.nlanes
        #self.capacity_col = 6 #in the paper they used 20
        self.cap = 11 #vehicles in a lane
        #now it depends on the net used, in "real" application it will be established based on capabilities/efficiency

        self.pedrate_NS = env_params.additional_params["pedrate_NS"]
        self.pedrate_EW = env_params.additional_params["pedrate_EW"]

        self.rew_f = env_params.additional_params["reward_function"]
        self.fact = env_params.additional_params["factor"]

        #self.dly = 10
        #self.ped_count = [0, 0, 0, 0]
        #self.removed = 0 #how many pedestrians leave (for the outflow)

        self.greens = ['GGgrrrGGgrrrrrrr', 'rrrGGgrrrGGgrrrr', 'rrrrrrrrrrrrGGGG']
        self.next_phase = self.greens[0]
        
        self.phases = ['rrrGGgrrrGGgrrrr', #VERDE NS
                'rrryyyrrryyyrrrr', 
                'rrrrrrrrrrrrGGGG', #VERDE PEDONI
                'rrrrrrrrrrrryyyy', 
                'GGgrrrGGgrrrrrrr', #VERDE EW
                'yyyrrryyyrrrrrrr' 
                ]

        self.prev_rw = 0
        #weights to balance rw between vehs and ped
        self.beta = 0.5 

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        """
        Defines the dimensions and bounds of the action space.
        [...]
        """
        #2 possible actions: increment phase by x seconds or switch phase (?)
        return Discrete(3) 

    @property
    def observation_space(self):
        """
        Defines the dimensions and bounds of the observation space.
        [...]
        """
        phase = Box(low=0,high=1, shape=(3,1), dtype=np.int32)  #Discrete(4)
        green = Box(low=0,high=1, dtype=np.float32)  #shape=(1,)
        density = Box(low=0,high=1, shape=(self.capacity_row,1), dtype=np.float32)
        queue = Box(low=0,high=1, shape=(self.capacity_row,1), dtype=np.float32)
        ped = Box(low=0,high=1, dtype=np.int32) #shape=(4,1),

        return Tuple((phase, green, density, queue, ped))
    

    #################################
    #      GET INFO ABOUT STATE     #
    #################################
    
    def get_state(self):
        """
        Returns the state of the simulation as perceived by the RL agent.
        [...]
        """
        current_phase = self.k.traffic_light.get_state(self.k.traffic_light.get_ids()[0])
        num_step = self.green_duration if current_phase in self.greens else self.yellow_duration

        if self.rew_f=='queue_length':
            phase = [1 if i==current_phase else 0 for i in self.greens]
        else:
            phase = [1 if i==current_phase else 0 for i in self.phases]

        self.green_time = self.green_time + self.green_duration if current_phase in self.greens else 0
        density, queue = self.get_stats()
        ped = len(self.get_ped(current_phase, num_step))
        
        return T.tensor((*phase, self.green_time, *density, *queue, ped))

    ### MOVE TO UTILS
    def get_stats(self):
        """Returns the density [0,1] and the queue [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        Obs: The queue is computed as the number of vehicles with vel<1 divided by the number of vehicles that could fit in the lane.
        """
        lane_density = []
        lane_queue = []
        vel_limit = 1

        for edge in self.edges:
            veh_byedge = self.k.vehicle.get_ids_by_edge(edge)
            temp_lanes = self.k.vehicle.get_lane(veh_byedge)
            
            #self.cap should be maximum vehs admitted x lane, check if there's a command to avoid manual interv
            lane_density.append(temp_lanes.count(1)/self.cap)
            lane_density.append(temp_lanes.count(2)/self.cap)
            lane_density.append(temp_lanes.count(3)/self.cap)

            speed_byedge = np.array([0 if vel>vel_limit else 1 for vel in self.k.vehicle.get_speed(veh_byedge)])
            speed_mask = Counter(np.where(speed_byedge==1, temp_lanes, -1))
            
            lane_queue.append(speed_mask[1]/self.cap)
            lane_queue.append(speed_mask[2]/self.cap)
            lane_queue.append(speed_mask[3]/self.cap)

        return [min(1, density) for density in lane_density], [min(1, queue) for queue in lane_queue]
    
    def get_ped(self, phase, num_step):

        #ADDING pedestrians
        if len(self.temp_ped)<200:
            if np.random.rand() <= self.pedrate_EW/3600:
                for st in range (num_step):
                    self.temp_ped["ped_"+str(self.curr_step+st)+"_EW"] = 0 #self.time_counter
            if np.random.rand() <= self.pedrate_EW/3600:
                for st in range (num_step):
                    self.temp_ped["ped_"+str(self.curr_step+st)+"_WE"] = 0
            if np.random.rand() <= self.pedrate_NS/3600:
                for st in range (num_step):
                    self.temp_ped["ped_"+str(self.curr_step+st)+"_NS"] = 0
            if np.random.rand() <= self.pedrate_NS/3600:
                for st in range (num_step):
                    self.temp_ped["ped_"+str(self.curr_step+st)+"_SN"] = 0

        #REMOVING pedestrians (when green) [or even yellow?]
        if phase[-4:] == 'GGGG':
            leaving = min(np.ceil(len(self.temp_ped)*0.3), 3)*self.green_duration
            #temp = len(self.temp_ped)
            self.temp_ped = {k: self.temp_ped[k] for k in list(self.temp_ped.keys())[int(leaving):]}
            #self.removed += temp - len(self.temp_ped)  # so as to get the outflow rate of ped
        if phase[-4:] == 'yyyyF':
            leaving = np.ceil(len(self.temp_ped)*0.1)
            self.temp_ped = {k: self.temp_ped[k] for k in list(self.temp_ped.keys())[int(leaving):]}

        #UPDATING TIMER (waiting time)
        #self.temp_ped = {k:v+self.sim_step for k,v in self.temp_ped.items()}
        if phase in self.greens:
            self.temp_ped = {k:v+self.green_duration for k,v in self.temp_ped.items()} 
        else:
            self.temp_ped = {k:v+5 for k,v in self.temp_ped.items()} #yellow phase dura 5sec
        
        return self.temp_ped
    
    #########################
    #    APPLY RL ACTION    #
    #########################

    def find_index(self, phase):
        try:
            return True, self.vehlights.index(phase)
        except ValueError:
            return False, self.pedlights(phase)

    def _apply_rl_actions(self, rl_actions):
        """
        Specifies the actions the rl agent has to perform.
        [...]
        """
        
        #self.green_time += 5*self.sim_step

        current_phase = self.k.traffic_light.get_state(self.k.traffic_light.get_ids()[0])
        current_phase_int = self.phases.index(current_phase)

        if current_phase_int%2!=0: #it's yellow
            self.k.traffic_light.set_state(node_id='0',state= self.next_phase)
            #self.green_time = 0

        
        self.next_phase = self.greens[int(rl_actions)]  # = [1 if rl_actions else 0]
        
        if current_phase in self.greens and current_phase!= self.next_phase or self.green_time>90:
            self.k.traffic_light.set_state(node_id='0',state= self.phases[(current_phase_int+1)%len(self.phases)])
            
    def _set_yellow_phase(self):
        current_phase = self.k.traffic_light.get_state(self.k.traffic_light.get_ids()[0])
        current_phase_int = self.phases.index(current_phase)
        self.k.traffic_light.set_state(node_id='0',state= self.phases[(current_phase_int+1)%len(self.phases)])

    def _set_green_phase(self, rl_action):
        self.k.traffic_light.set_state(node_id='0',state= self.greens[int(rl_action)] )
  

    ############################
    #      COMPUTE REWARD      #
    ############################

    def compute_reward(self, **kwargs): #rl_actions
        """
        Computes the reward generated by the rl_action, 
        given a specific reward function.
        [...]
        """
    
        if self.rew_f =='waiting_time':
            return  (1 - self.beta)*self.waiting_time()+ self.beta*(0.5*self.ped_wait_time()) #- self.action_penalty(rl_actions, self.fact)
        elif self.rew_f =='queue_length':
            return  (1 - self.beta)*self.queue_length() + self.beta*(1.2*self.ped_wait_time()/(len(self.temp_ped)+1)) #- self.action_penalty(rl_actions, self.fact) 
        elif self.rew_f =='avg_delay':
            return  (1 - self.beta)*self.avg_delay() + self.beta*(0.35*self.ped_wait_time()/(len(self.temp_ped)+1))  # - self.action_penalty(rl_actions, self.fact) 
        else:
            print('Function not available')
            return None


    #################################
    #         RUN SIMULATION        #
    #################################
    

    def run_sim(self, num_steps=1):
        if self.curr_step + num_steps > self.max_steps:
            num_steps = self.max_steps - self.curr_step
            
        for _ in range(num_steps):

            ##########
            if len(self.k.vehicle.get_controlled_ids()) > 0:
                accel = []
                for veh_id in self.k.vehicle.get_controlled_ids():
                    action = self.k.vehicle.get_acc_controller(
                        veh_id).get_action(self)
                    accel.append(action)
                self.k.vehicle.apply_acceleration(
                    self.k.vehicle.get_controlled_ids(), accel)

            # perform (optionally) routing actions for all vehicles in the
            # network, including RL and SUMO-controlled vehicles
            routing_ids = []
            routing_actions = []
            for veh_id in self.k.vehicle.get_ids():
                if self.k.vehicle.get_routing_controller(veh_id) \
                        is not None:
                    routing_ids.append(veh_id)
                    route_contr = self.k.vehicle.get_routing_controller(
                        veh_id)
                    routing_actions.append(route_contr.choose_route(self))

            self.k.vehicle.choose_routes(routing_ids, routing_actions)
            ############
            
            self.k.simulation.simulation_step()

            # check base env 
            self.k.update(reset=False)
            if self.sim_params.render:
                self.k.vehicle.update_vehicle_colors()
            crash = self.k.simulation.check_collision()
            if crash:
                break
            #[...]
            self.render()

        
        self.curr_step += num_steps
        curr_state = self.get_state()

        if self.rew_f=='waiting_time':
            curr_reward = self.compute_reward() #self.waiting_time(vehs)[1]
            reward = 0.9*self.prev_rw - curr_reward #self.compute_reward() 
            self.prev_rw = curr_reward
        else:
            reward = -self.compute_reward()
        
        next_obs = np.copy(curr_state)
        is_terminal = self.curr_step >= self.max_steps or crash
        return next_obs, reward, is_terminal
    

    def run_sim2(self, num_steps=1):
        if self.curr_step + num_steps > self.max_steps:
            num_steps = self.max_steps - self.curr_step
            
        for _ in range(num_steps):
            
            self.k.simulation.simulation_step()

            # check base env 
            self.k.update(reset=False)
            if self.sim_params.render:
                self.k.vehicle.update_vehicle_colors()
            crash = self.k.simulation.check_collision()
            if crash:
                break
            #[...]
            self.render()
            
            ##
            self.vehs = []
            for edge in self.edges: #self.k.network.get_edge_list():
                self.vehs.extend(self.veh_set(edge)) #

            self.tot_vehs = {k: v for k, v in self.tot_vehs.items() if k in self.vehs}

            for veh in self.vehs:
                if self.k.vehicle.get_speed(veh)<1:
                    if veh not in self.tot_vehs:
                        self.tot_vehs[veh]= self.sim_step
                    else:
                        self.tot_vehs[veh]+= self.sim_step
                
            ##
        
        self.curr_step += num_steps
        curr_state = self.get_state()

        if self.rew_f=='waiting_time':
            curr_reward = self.compute_reward()  
            reward = 0.9*self.prev_rw - curr_reward  
            self.prev_rw = curr_reward
        #elif self.rew_f=='avg_delay':
        #    curr_reward = self.compute_reward()  
         #   reward = self.prev_rw - curr_reward   
          #  self.prev_rw = curr_reward
        else:
            reward = -self.compute_reward()
        
        next_obs = np.copy(curr_state)
        is_terminal = self.curr_step >= self.max_steps or crash
        return next_obs, reward, is_terminal
    

    #################################
    #             UTILS             #
    #################################

    def veh_set(self, edge):
        """
        Retrieve the closest vehicles to the intersection by edge
        """

        veh_ids = self.k.vehicle.get_ids_by_edge(edge)

        veh_dist = self.get_distance_to_intersection(veh_ids)
        limit = (self.veh_length+2.5) * 10 #*self.capacity_col
        mask = [1 if i<=limit else 0 for i in veh_dist]

        return [veh_ids[i] for i,j in enumerate(mask) if j]

    def edge_mapping(self, edge):
        edge_to_name = {i: self.edges[i] for i in range(len(self.edges))}
        name_to_edge = {name: i for i, name in edge_to_name.items()}

        if isinstance(edge, int):
            return edge_to_name.get(edge, "Invalid edge number")
        elif isinstance(edge, str):
            return name_to_edge.get(edge, "Invalid edge name")
        else:
            return "Invalid input type"
        
    def phase_map(self, phase):
    
        int_2phase = {i: self.phases[i] for i in range(len(self.phases))}
        phase_2int = {name: i for i, name in int_2phase.items()}

        if isinstance(phase, int):
            return int_2phase.get(phase, "Invalid phase number")
        elif isinstance(phase, str):
            return phase_2int.get(phase, "Invalid phase")
        else:
            return "Invalid input type"

    
    def get_distance_to_intersection(self, veh_ids):
        """
        Determine the distance from a vehicle to its next intersection.
        [...]
        """

        if isinstance(veh_ids, list):
            return [self.get_distance_to_intersection(veh_id)
                    for veh_id in veh_ids]
        return self.find_intersection_dist(veh_ids)
        
    def find_intersection_dist(self, veh_id):
        """Return distance from intersection.

        Return the distance from the vehicle's current position to the position
        of the node it is heading toward.
        """

        edge_id = self.k.vehicle.get_edge(veh_id)
        
        if edge_id == "":
            return -10
        if 'center' in edge_id:
            return 0
        edge_len = self.k.network.edge_length(edge_id)
        relative_pos = self.k.vehicle.get_position(veh_id)
        dist = edge_len - relative_pos
        return dist
    
    #######################################
    #      REWARD FUNCTIONS (new file?)   #
    #######################################

    def avg_delay(self):
        """
        Calculate the average [TOTAL now - change labels] delay for a set of vehicles in the system.
        [...]
        """
        
        if len(self.vehs)==0:
            return 0
        
        tot_del = 0
        #v_top = self.network.net_params.additional_params['speed_limit']
        v_top = 12 #15.458 max vel recorded

        for veh_id in self.vehs:
            tot_del += (v_top - self.k.vehicle.get_speed(veh_id))/ v_top

        return tot_del


    def queue_length(self):
        """
        Calculate the queue for a set of vehicles in the system.
        """

        v_stand = 1 #if below vehicle is considered stopped/in queue
        max_distances = {(edge, lane): 0 for edge in self.edges for lane in range(1,4)}
        
        for veh,time in self.tot_vehs.items(): #veh in self.vehs:
            
            if self.k.vehicle.get_speed(veh)<v_stand and time>=6:
                edge = self.k.vehicle.get_edge(veh)
                lane = self.k.vehicle.get_lane(veh)
                distance = self.get_distance_to_intersection(veh)
                
                if distance > max_distances[(edge, lane)]:
                    max_distances[(edge, lane)] = distance
        
        #NS_queue = sum([v for k,v in max_distances.items() if 'NC' in k] + [v for k,v in max_distances.items() if 'SC' in k])
        #EW_queue = sum([v for k,v in max_distances.items() if 'EC' in k] + [v for k,v in max_distances.items() if 'WC' in k])
        #queue_diff = abs(NS_queue - EW_queue)

        queue = sum(max_distances.values()) #+ 0.1*queue_diff #i**2 for i in max_distances.values()
 
        return queue
   

    def waiting_time(self):
        """
        Calculate the average waiting time for a set of vehicles in the system.
        
        - when v<1km/h (?) 
        """

        return sum(self.tot_vehs.values())     #temp_vehs solo per debug rimuovi
    
    def ped_wait_time(self):
        return  sum(self.temp_ped.values())


    def action_penalty(self, rl_action, fct):
        """Penalize when switching phase, to avoid continuous switching"""
        return fct * int(rl_action)



