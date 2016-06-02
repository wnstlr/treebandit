import numpy as np
import random
from normal import *
from bernoulli import *

"""
Tree structure used for MAB
"""

class Tree:
    """
    The tree will be represented as a dictionary, where the key is the 
    node id, and the value is again a dictionary of traits.
    """
    def __init__(self):
        self.nodes = dict()
        self.leaf_ids = []
        self.node_ids = range(1,100000)   # just used to label the child nodes
        self.root = 0   # tree's root node is always labeled as 0.

    # Create a tree with maximum depth specified
    def create_tree(self, max_depth):
        self.max_depth = max_depth
        self.create_tree_aux(Node(self.root), 0)
        self.num_nodes = 2 ** (self.max_depth+1) - 1

    # Helper function for create_tree
    # Used for recursive calls in create_tree function.
    def create_tree_aux(self, start_node, curr_depth):
        parent_id = start_node.node_id
        parent_node = start_node
        parent_node.depth = curr_depth
        self.nodes[parent_id] = self.node2dict(parent_node)

        # Binary tree
        num_child = 2

        # If we have reached a leaf node, change is_leaf flag to True
        if curr_depth == self.max_depth:
            self.leaf_ids.append(parent_id)
            self.nodes[parent_id]['is_leaf'] = True
            return 

        # Create each child of the current node recursively.
        for c in xrange(num_child):
            child_id = self.node_ids.pop(0)
            child = Node(child_id)
            child.parent_id = parent_node.node_id
            parent_node.add_child(child_id)
            self.nodes[child_id] = self.node2dict(child)
            self.create_tree_aux(child, curr_depth+1)

    # Get the path from the root to the leaf id as a list of node_ids
    def get_path(self, leaf_id):
        path = [leaf_id]
        while True:
            parent = self.nodes[leaf_id]['parent_id']
            if parent != None:
                path = [parent] + path
            else: 
                break
            leaf_id = parent
        return path

    # Get the list of leafs of subtree rooted at specified node
    def get_leafs_of_subtree(self, subroot_id):
        if subroot_id in self.leaf_ids:
            return [subroot_id]
        return self.get_leafs_of_subtree_aux([], subroot_id)

    # Helper function for get_leafs_of_subtree
    def get_leafs_of_subtree_aux(self, leafs, subroot_id):
        for c in self.get_children(subroot_id):
            if self.is_leaf(c):
                leafs.append(c)
            else:
                self.get_leafs_of_subtree_aux(leafs, c)
        return leafs

    # Set up arms for rewards
    # NOTE: this method must be run after creating the Tree class object.
    # Otherwise, the tree will be basically useless as there are no
    # arm defined for each node.
    def setup_smooth_arms(self, val_opt, delta, eta= 0.1, \
                          delta_type="exponential", arm_type="bernoulli"):

        if delta_type == "exponential":
            gamma = 0.5
            self.delta = delta * (gamma * np.ones(self.max_depth)) ** \
                        np.array(range(self.max_depth))
        elif delta_type == "linear":
            self.delta = delta * (self.max_depth - \
                                        np.array(range(self.max_depth)))
        elif delta_type == "polynomial":
            alpha = -0.5
            self.delta = delta * np.array(range(self.max_depth)) ** alpha

        self.eta = eta
        self.val_opt = val_opt

        self.sub_opt = [k for k, v in self.nodes.iteritems() \
                                   if self.val_opt - v['mu'] <= eta]

        # Reset the mu and sigma for each arms to ensure smoothness
        for i in self.sub_opt:
            d = self.get_depth(i)
            leafs = self.get_leafs_of_subtree(i)
            for l in leafs:
                while True:
                    if self.get_mu(i) - self.get_mu(l) <= self.delta[d-1]:
                        break
                    self.reset_mu_sigma(l)

        # Create arms based on the arm_type input and store it in dictionary.
        self.arms = dict()
        if arm_type == "normal":
            for k, v in self.nodes.iteritems():
                self.arms[k] = NormalArm(v['mu'], v['sigma'])
        elif arm_type == "bernoulli":
            for k, v in self.nodes.iteritems():
                self.arms[k] = BernoulliArm(v['mu'])

    ## Accessor and static functions
    def is_leaf(self, node_id):
        return self.nodes[node_id]['is_leaf']

    def get_children(self, node_id):
        return self.nodes[node_id]['children']

    def get_depth(self, node_id):
        return self.nodes[node_id]['depth']

    def get_parent(self, node_id):
        return self.nodes[node_id]['parent']

    def get_mu(self, node_id):
        return self.nodes[node_id]['mu']

    def reset_mu_sigma(self, node_id):
        self.nodes[node_id]['mu'] = random.random()
        self.nodes[node_id]['sigma'] = random.uniform(0, self.get_mu(node_id))

    @staticmethod
    # Convert the node information into dictionary for storing purposes.
    def node2dict(node):
        result = dict()
        # parent id of the node
        result['parent_id'] = node.parent_id
        # list of child nodes of the node
        result['children'] = node.children
        # boolean whether the node is a leaf or not
        result['is_leaf'] = node.is_leaf
        # depth of the node
        result['depth'] = node.depth
        # mu value of the node
        result['mu'] = node.mu
        # sigma value of the node (std) for normal arms
        result['sigma'] = node.sigma
        return result

class Node:
    """
    Single node of the tree
    """
    def __init__(self, node_id):
        self.node_id = node_id
        self.children = []
        self.parent_id = None
        self.is_leaf = False
        self.depth = 0
        self.mu = 0
        self.sigma = random.random()

    def add_child(self, child_id):
        self.children.append(child_id)

