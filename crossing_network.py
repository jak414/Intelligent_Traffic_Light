"""Contains the traffic light junction network class."""

from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from collections import defaultdict
import numpy as np


ADDITIONAL_NET_PARAMS = {
    # length of lanes
    "length" : 100,
    # radius of intersection
    "radius" : 15,
    # number of lanes for each edge
    "nlanes" : 3,
    # speed limit (# if we want to differentiate among directions)
    "speed_limit": 35, #{"horizontal": 35,"vertical": 35}
	"edge_names" : ['edge-east-EW', 'edge-south-SN', 'edge-west-WE', 'edge-north-NS'] # from x to center
}

ADDITIONAL_NET_PARAMS2 = {
    # length of lanes
    "length" : 100,
    # radius of intersection
    "radius" : 15,
    # number of lanes for each edge
    "nlanes" : 3,
    # speed limit (# if we want to differentiate among directions)
    "speed_limit": 35, #{"horizontal": 35,"vertical": 35}
	"edge_names" : ['EC', 'SC', 'WC', 'NC'], # from x to center
	"traffic_lights": True
}


from flow.core.kernel.network import TraCIKernelNetwork

class crossingNetwork(Network):
	"""
	Add description
	"""
	
	def __init__(self,
		         name,
		         vehicles,
		         net_params,
		         initial_config=InitialConfig(),
		         traffic_lights=TrafficLightParams()):
		"""Initialize a simple junction regulated by a traffic light."""
		optional = ["tl_logic"]
		for p in ADDITIONAL_NET_PARAMS.keys():
		    if p not in net_params.additional_params and p not in optional:
		        raise KeyError('Network parameter "{}" not supplied'.format(p))


	      
		self.speed_limit = net_params.additional_params["speed_limit"]
		# if it's not a dictionary(i.e.: unique), we set the value either for NS and EW lanes
		#if not isinstance(self.speed_limit, dict):
		#    self.speed_limit = {
		#        "horizontal": self.speed_limit,
		#        "vertical": self.speed_limit
		#    }

		
		self.length = net_params.additional_params["length"]
		self.radius = net_params.additional_params["radius"]
		self.num_lanes = net_params.additional_params["nlanes"]


		# specifies whether or not there will be traffic lights at the
		# intersections (True by default)
		self.use_traffic_lights = net_params.additional_params.get(
		    "traffic_lights", True)

		# radius of the inner nodes (ie of the intersections)  (DELETE?)
		self.inner_nodes_radius = 2.9 + 3.3 

		# total number of edges in the network (DELETE?)
		self.num_edges = 4 

		# name of the network (DO NOT CHANGE) (mmmmmm)
		self.name = "BobLoblawsLawBlog"

		super().__init__(name, vehicles, net_params, initial_config,
		                 traffic_lights)
		
	def specify_nodes(self, net_params):
		"""
		[...]
		"""

		nodes = [{"id": "bottom", "x": 0,  "y": -self.length, "type":"priority"},
		         {"id": "right",  "x": self.length,  "y": 0, "type":"priority"},
		         {"id": "top",    "x": 0,  "y": self.length, "type":"priority"},
		         {"id": "left",   "x": -self.length, "y": 0, "type":"priority"},
		         {"id": "center", "x": 0, "y": 0, "type":"traffic_light", "radius":self.radius},
		         #{"id":"testnode", "x":self.length, "y":self.length, "type":"priority"}
		         #{"id":"topleft", "x":-l/4, "y":l/2, "type":"priority"},
		         #{"id":"midleft", "x":-l/4, "y":0, "type":"priority", "radius":r},
		         #{"id":"botleft", "x":-l/4, "y":-l/2, "type":"priority"},
		        ]
		        
		return nodes
		          
	def specify_edges(self, net_params):
		"""[...]"""

		edges = [
		    {
		        "id": "btoc",  "type":"edgeType", #"numLanes": self.num_lanes, "speed": self.speed_limit,#, "type":"edgeType",#
		        "from": "bottom", "to": "center", "length":self.length
		    },
		    {
		        "id": "ctob", "type":"edgeType", #"numLanes": self.num_lanes, "speed": self.speed_limit,#, "type":"edgeType",# 
		        "from": "center", "to": "bottom", "length":self.length
		    },
		    {
		        "id": "ttoc", "type":"edgeType", #"numLanes": self.num_lanes, "speed": self.speed_limit,#, "type":"edgeType",#
		        "from": "top", "to": "center", "length":self.length
		    },
		    {
		        "id": "ctot", "type":"edgeType", #"numLanes": self.num_lanes, "speed": self.speed_limit,#, "type":"edgeType",#
		        "from": "center", "to": "top", "length":self.length
		    },
		    {
		        "id": "rtoc", "type":"edgeType", # "numLanes": self.num_lanes, "speed": self.speed_limit,#, "type":"edgeType",#
		        "from": "right", "to": "center", "length":self.length
		    },
		    {
		        "id": "ctor", "type":"edgeType", #"numLanes": self.num_lanes, "speed": self.speed_limit,#, "type":"edgeType",#
		        "from": "center", "to": "right", "length":self.length
		    },
		    {
		        "id": "ltoc", "type":"edgeType", #"numLanes": self.num_lanes, "speed": self.speed_limit,#, "type":"edgeType",#
		        "from": "left", "to": "center", "length":self.length
		    },
		    {
		        "id": "ctol", "type":"edgeType", #"numLanes": self.num_lanes, "speed": self.speed_limit,#, "type":"edgeType",#
		        "from": "center", "to": "left", "length":self.length
		    },
		    #{
		    #    "id": "test", "numLanes": 1, "speed": speed_limit,#, "type":"edgeType",#
		    #    "from": "right", "to": "testnode", "length":l
		    #},
		    #{
		     #   "id": "test2", "numLanes": 1, "speed": speed_limit,#, "type":"edgeType",#
		      #  "from": "midleft", "to": "botleft", "length":l/2
		    #},
		    
		
		]

		return edges
		
	def specify_types(self, net_params):
		"""
		[...]
		"""
	
		types = [{
		    "id": "edgeType",
			#"length" : self.length,
		    "numLanes": self.num_lanes,
		    "speed": self.speed_limit
		}]
		
		return types
		
	def specify_routes(self, net_params):
		"""
		"""

		rts = {"ttoc": [(["ttoc", "ctor"], 1/3), (["ttoc", "ctol"], 1/3), (["ttoc", "ctob"], 1/3)],
		       "rtoc": [(["rtoc", "ctob"], 1/3), (["rtoc", "ctol"], 1/3), (["rtoc", "ctot"], 1/3)],
		       "ltoc": [(["ltoc", "ctob"], 1/3), (["ltoc", "ctor"], 1/3), (["ltoc", "ctot"], 1/3)],
		       "btoc": [(["btoc", "ctor"], 1/3), (["btoc", "ctol"], 1/3), (["btoc", "ctot"], 1/3)],
		       "ctor": ["ctor"], "ctol": ["ctol"],"ctot": ["ctot"], "ctob": ["ctob"]}

		return rts
	
	
	def specify_connections(self, net_params):
		"""
		"""
		con = {'center': [{'from': 'btoc', 'to': 'ctot','fromLane': '1','toLane': '1'},
		                  {'from': 'btoc', 'to': 'ctor','fromLane': '0','toLane': '0'},
		                  {'from': 'btoc', 'to': 'ctol','fromLane': '2','toLane': '2'},
		                  {'from': 'ltoc', 'to': 'ctot','fromLane': '2','toLane': '2'},
		                  {'from': 'ltoc', 'to': 'ctor','fromLane': '1','toLane': '1'},
		                  {'from': 'ltoc', 'to': 'ctob','fromLane': '0','toLane': '0'},
		                  {'from': 'ttoc', 'to': 'ctob','fromLane': '1','toLane': '1'},
		                  {'from': 'ttoc', 'to': 'ctor','fromLane': '2','toLane': '2'},
		                  {'from': 'ttoc', 'to': 'ctol','fromLane': '0','toLane': '0'},
		                  {'from': 'rtoc', 'to': 'ctot','fromLane': '0','toLane': '0'},
		                  {'from': 'rtoc', 'to': 'ctob','fromLane': '2','toLane': '2'},
		                  {'from': 'rtoc', 'to': 'ctol','fromLane': '1','toLane': '1'}]}
		       #'right':[{'from': 'ctor', 'to': 'test','fromLane': '0','toLane': '0'},
		          #      {'from': 'ctor', 'to': 'test','fromLane': '1','toLane': '0'},
		           #     {'from': 'ctor', 'to': 'test','fromLane': '2','toLane': '0'},]
		      #}
        
		return con
		
	def get_edge_list(self):

		return self._edge_list
	
class crossingNetworkTemplate(Network):

	def specify_routes(self, net_params):
		rts = {"edge-west-WE": [(["edge-west-WE", "edge-north-SN"], 1/3),
                                (["edge-west-WE", "edge-east-WE"], 1/3),
                                (["edge-west-WE", "edge-south-NS"], 1/3)],
               "edge-north-NS": [(["edge-north-NS", "edge-west-EW"], 1/3),
                                 (["edge-north-NS", "edge-east-WE"], 1/3),
                                 (["edge-north-NS", "edge-south-NS"], 1/3)],
               "edge-south-SN": [(["edge-south-SN", "edge-north-SN"], 1/3),
                                 (["edge-south-SN", "edge-west-EW"], 1/3),
                                 (["edge-south-SN", "edge-east-WE"], 1/3)],
               "edge-east-EW": [(["edge-east-EW", "edge-north-SN"], 1/3),
                                (["edge-east-EW", "edge-south-NS"], 1/3),
                                (["edge-east-EW", "edge-west-EW"], 1/3)],
               "edge-east-WE": ["edge-east-WE"], "edge-north-SN": ["edge-north-SN"],
               "edge-west-EW": ["edge-west-EW"], "edge-south-NS": ["edge-south-NS"]}
		
		self.glob = rts
		return rts	  

	def test2(self):
		return self.glob   
					

class crossingNetworkTemplate_2(Network):
    
       def specify_routes(self, net_params):
        rts = {"ttoc": [(["ttoc", "ctor"], 1/3), (["ttoc", "ctol"], 1/3), (["ttoc", "ctob"], 1/3)],
               "rtoc": [(["rtoc", "ctob"], 1/3), (["rtoc", "ctol"], 1/3), (["rtoc", "ctot"], 1/3)],
               "ltoc": [(["ltoc", "ctob"], 1/3), (["ltoc", "ctor"], 1/3), (["ltoc", "ctot"], 1/3)],
               "btoc": [(["btoc", "ctor"], 1/3), (["btoc", "ctol"], 1/3), (["btoc", "ctot"], 1/3)],
               "ctor": ["ctor"], "ctol": ["ctol"],"ctot": ["ctot"], "ctob": ["ctob"]}

        return rts


class crossingNetworkTemplate_3(Network):

	def specify_routes(self, net_params):
		rts = {"WC": [(["WC", "CN"], 1/3),
                                (["WC", "CE"], 1/3),
                                (["WC", "CS"], 1/3)],
               "NC": [(["NC", "CW"], 1/3),
                                 (["NC", "CE"], 1/3),
                                 (["NC", "CS"], 1/3)],
               "SC": [(["SC", "CN"], 1/3),
                                 (["SC", "CW"], 1/3),
                                 (["SC", "CE"], 1/3)],
               "EC": [(["EC", "CN"], 1/3),
                                (["EC", "CS"], 1/3),
                                (["EC", "CW"], 1/3)],
               "CW": ["CW"], "CS": ["CS"],
               "CE": ["CE"], "CN": ["CN"]}
		
		self.glob = rts
		return rts	  

	def test(self):
		return self.routes  
	
	def test2(self):
		return self.glob
						 


class crossingNetworkTemp_test(Network):
	"""
	Add description
	"""
	
	def __init__(self,
		         name,
		         vehicles,
		         net_params,
		         initial_config=InitialConfig(),
		         traffic_lights=TrafficLightParams()):
		"""Initialize a simple junction regulated by a traffic light."""
		optional = ["tl_logic"]
		for p in ADDITIONAL_NET_PARAMS.keys():
		    if p not in net_params.additional_params and p not in optional:
		        raise KeyError('Network parameter "{}" not supplied'.format(p))


	      
		self.speed_limit = net_params.additional_params["speed_limit"]
		# if it's not a dictionary(i.e.: unique), we set the value either for NS and EW lanes
		#if not isinstance(self.speed_limit, dict):
		#    self.speed_limit = {
		#        "horizontal": self.speed_limit,
		#        "vertical": self.speed_limit
		#    }

		
		self.length = net_params.additional_params["length"]
		self.radius = net_params.additional_params["radius"]
		self.num_lanes = net_params.additional_params["nlanes"]


		# specifies whether or not there will be traffic lights at the
		# intersections (True by default)
		self.use_traffic_lights = net_params.additional_params.get(
		    "traffic_lights", True)

		# radius of the inner nodes (ie of the intersections)  (DELETE?)
		self.inner_nodes_radius = 2.9 + 3.3 

		# total number of edges in the network (DELETE?)
		self.num_edges = 4 

		# name of the network (DO NOT CHANGE) (mmmmmm)
		self.name = "BobLoblawsLawBlog"

		self.routes = self.specify_routes(net_params)

		super().__init__(name, vehicles, net_params, initial_config,
		                 traffic_lights)
		
	def specify_routes(self, net_params):
		rts = {"WC": [(["WC", "CN"], 1/3),
                                (["WC", "CE"], 1/3),
                                (["WC", "CS"], 1/3)],
               "NC": [(["NC", "CW"], 1/3),
                                 (["NC", "CE"], 1/3),
                                 (["NC", "CS"], 1/3)],
               "SC": [(["SC", "CN"], 1/3),
                                 (["SC", "CW"], 1/3),
                                 (["SC", "CE"], 1/3)],
               "EC": [(["EC", "CN"], 1/3),
                                (["EC", "CS"], 1/3),
                                (["EC", "CW"], 1/3)],
               "CW": ["CW"], "CS": ["CS"],
               "CE": ["CE"], "CN": ["CN"]}
		
		
		return rts	 
		                  
                         
                         
                         
                         
                         
                         
                         
                         
                         

