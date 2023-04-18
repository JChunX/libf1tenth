import numpy as np
from libf1tenth.util.quick_maths import l2_norm

'''
Graph components
'''

class PlanGraph:
    '''
    PlanGraph represents possible traversals between start and end positions
    
    implemented as 
    - a list of nodes
    - a list of edges, represented as tuples of node indices
    - a dictionary of adjacencies, where the key is a node index and the value is a list of node indices
    
    Args:
    - start_pos: ndarray of shape (2,) representing start position
    - end_pos: ndarray of shape (2,) representing end position
    '''
    def __init__(self, start_pos, end_pos=None):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.nodes = []
        self.edges = []
        self.adjacencies = {}
        self.node_positions = None # shape (n, 2)
        
        self.add_node(PlanNode(start_pos[0], start_pos[1]))
        
    def __repr__(self):
        return f'PlanGraph with {len(self.nodes)} nodes and {len(self.edges)} edges'
        
    def add_node(self, node):
        assert issubclass(type(node), PlanNode), 'node must be a PlanNode'
        self.nodes.append(node)
        node.id = len(self.nodes) - 1
        self.adjacencies[node.id] = []
        if self.node_positions is not None:
            self.node_positions = np.vstack((self.node_positions, node.position))
        else:
            self.node_positions = node.position.reshape(1, 2)
        
    def get_node(self, id):
        return self.nodes[id]
        
    def add_edge(self, a, b, parent_id=None, add_cost=False, cost=None):
        self.edges.append((a, b))
        self.adjacencies[a].append(b)
        self.adjacencies[b].append(a)
        if parent_id is not None:
            if parent_id == a:
                parent_node = self.nodes[a]
                child_node = self.nodes[b]
            else:
                parent_node = self.nodes[b]
                child_node = self.nodes[a]
            child_node.set_parent(parent_node)
            if add_cost:
                if cost:
                    child_node.cost = cost
                else:
                    child_node.cost = (parent_node.cost
                                   + np.linalg.norm(
                                       parent_node.position
                                       - child_node.position
                                       )
                                   )

    def get_node_chain(self, end_node):
        cur_node = end_node
        node_chain = []
        while(cur_node.parent is not None):
            node_chain.append(cur_node)
            cur_node = cur_node.parent
        node_chain.append(cur_node)
        return node_chain
    
    def get_near_node_ids(self, node, radius, max_num=50):
        '''
        Returns a list of the (num) nearest nodes to the given node
        
        Args:
        - node (PlanNode): node to find nearest nodes to
        - num (int): number of nearest nodes to return
        
        Returns:
        - nearest_nodes (list): list of nearest nodes
        '''
        
        diff = self.node_positions - node.position
        distances = l2_norm(diff[:,0], diff[:,1])
        nearest_node_ids = np.argwhere(distances < radius).flatten()
        nearest_node_ids = nearest_node_ids[np.argsort(distances[nearest_node_ids])][:max_num]
        
        return nearest_node_ids
    
    def get_nearest_node_idx(self, node):
        '''
        Returns the nearest node idx to the given node
        
        Args:
        - node (PlanNode): node to find nearest node to
        
        Returns:
        - nearest_node_idx (int): index of nearest node in self.nodes
        '''
        diff = self.node_positions - node.position
        return np.argmin(l2_norm(diff[:,0], diff[:,1]))
    

class FrenetPlanGraph(PlanGraph):
    
    def __init__(self, frenet_frame, start_pos, end_pos=None):
        super().__init__(start_pos, end_pos)
        self.frenet_frame = frenet_frame
        
    def add_edge(self, a, b, parent_id, cost):
        super().add_edge(a, b, parent_id, True, cost)
        
    def get_near_node_ids(self, node, radius, max_num=50):
        distances = self.frenet_frame.frenet_distance(node.position, self.node_positions)
        nearest_node_ids = np.argwhere(distances < radius).flatten()
        nearest_node_ids = nearest_node_ids[np.argsort(distances[nearest_node_ids])][:max_num]
        
        return nearest_node_ids
    
    def get_nearest_node_idx(self, node):
        distances = self.frenet_frame.frenet_distance(node.position, self.node_positions)
        return np.argmin(distances)

class PlanNode:
    '''
    Node representation for planning purposes
    
    Args:
    - x (float): x position
    - y (float): y position
    - parent (PlanNode): parent node
    - aux_states (dict): dictionary of auxiliary states
    '''
    
    def __init__(self, x, y, parent: 'PlanNode'=None, aux_states=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.id = None
        self.cost = 0.0
        self.aux_states = aux_states

    def __repr__(self):
        return f'PlanNode(x={self.x}, y={self.y}, parent={self.parent}, id={self.id}, cost={self.cost}, aux_states={self.aux_states})'
    
    @property
    def position(self):
        return np.array([self.x, self.y])
    
    def set_parent(self, parent):
        assert issubclass(type(parent), PlanNode), 'parent must be a PlanNode'
        self.parent = parent
        