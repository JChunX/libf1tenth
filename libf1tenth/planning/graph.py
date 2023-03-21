import numpy as np
from numba import njit

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
        self.nodes = [PlanNode(start_pos[0], start_pos[1])]
        self.edges = []
        
        self.adjacencies = {0: []}
        self.node_positions = np.array([start_pos]) # shape (n, 2)
        
    def __repr__(self):
        return f'PlanGraph with {len(self.nodes)} nodes and {len(self.edges)} edges'
        
    def add_node(self, node):
        assert issubclass(type(node), PlanNode), 'node must be a PlanNode'
        self.nodes.append(node)
        node.id = len(self.nodes) - 1
        self.adjacencies[node.id] = []
        self.node_positions = np.vstack((self.node_positions, node.position))
        
    def get_node(self, id):
        return self.nodes[id]
        
    def add_edge(self, a, b, parent_id=None):
        self.edges.append((a, b))
        self.adjacencies[a].append(b)
        self.adjacencies[b].append(a)
        if parent_id:
            if parent_id == a:
                parent_node = self.nodes[a]
                child_node = self.nodes[b]
            else:
                parent_node = self.nodes[b]
                child_node = self.nodes[a]
            child_node.set_parent(parent_node)

    def get_node_chain(self, end_node):
        cur_node = end_node
        node_chain = []
        while(cur_node.parent is not None):
            node_chain.append(cur_node)
            cur_node = cur_node.parent
        node_chain.append(cur_node)
        return node_chain
    
    def get_nearest_node_idx(self, node):
        '''
        Returns the nearest node idx to the given node
        
        Args:
        - node: PlanNode object
        
        Returns:
        - nearest_node_idx: index of nearest node in self.nodes
        '''
        return np.argmin(np.linalg.norm(self.node_positions - node.position, axis=1))
    

class PlanNode:
    '''
    Node representation for planning purposes
    '''
    
    def __init__(self, x, y, parent: 'PlanNode'=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.id = None
        self.cost = 0.0

    def __repr__(self):
        return f'PlanNode(x={self.x}, y={self.y}, parent={self.parent})'
    
    @property
    def position(self):
        return np.array([self.x, self.y])
    
    def set_parent(self, parent):
        assert issubclass(type(parent), PlanNode), 'parent must be a PlanNode'
        self.parent = parent
        