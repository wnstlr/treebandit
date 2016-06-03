# Exploration algorithm using Smoothness of Tree

import numpy as np
import random

class BAST_EXP(object):
    """
    Exploration of smooth tree.
    """
    def __init__(self, tree, delta, beta, eps, conf_method, method_upper, method_lower, delta_type="exponential"):
        self.tree = tree
        self.opt_path = [self.tree.root]
        self.curr_tree = self.tree.root
        self.leafs = self.tree.leaf_ids
        self.n_arms = len(self.tree.nodes)
        self.arms = range(self.n_arms)
        self.conf_method = conf_method
        self.method_upper = method_upper
        self.method_lower = method_lower

        if delta_type == "exponential":
            gamma = 0.5
            self.delta = delta * (gamma * np.ones(self.tree.max_depth)) ** \
                    np.array(range(self.tree.max_depth))
        elif delta_type == "linear":
            self.delta = delta * (self.tree.max_depth - \
                    np.array(range(self.tree.max_depth)))
        elif delta_type == "polynomial":
            alpha = -0.5
            self.delta = delta * np.array(range(self.tree.max_depth)) ** alpha
        
        self.beta = beta
        self.eps = eps  # tolerance for stopping criteria
        self.t = 0
        self.N = 0  # sampling complexity

        # max number of iterations
        self.T = 100000

    def initialize(self):
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.values = np.zeros(self.n_arms)
        self.emp_means = np.zeros(self.n_arms)
        self.uppers = np.ones(self.n_arms)
        self.lowers = np.zeros(self.n_arms)
        self.m_vals = np.zeros(self.n_arms)

    def select_arm(self):
        while True:
            if self.curr_tree in self.tree.leaf_ids:
                break

            # Find the best empirical arm
            children = self.tree.get_children(self.curr_tree)
            opt_branch = None
            max_val = -1
            for c in children:
                if max_val < self.m_vals[c]:
                    max_val = self.m_vals[c]
                    opt_branch = c

            subopt_branch = [x for x in self.tree.get_children(self.curr_tree) \
                    if x != opt_branch][0]

            if self.lowers[opt_branch] >= self.uppers[subopt_branch] - self.eps:
                self.opt_path.append(opt_branch)
                self.curr_tree = opt_branch
            else:
                path1 = self.get_best_LCB(opt_branch)
                path2 = self.get_best_UCB(subopt_branch)
                return path1, path2

        return self.opt_path, self.opt_path

    def update(self, path1, path2, reward1, reward2):
        # Start from the leaf and update the values and emp_means
        rewards = [reward1, reward2]
        paths = [path1, path2]
        for p in xrange(len(paths)):
            path = paths[p]
            self.counts[path] += 1
            reward = rewards[p]
            n = self.counts[path[-1]]

            # We first update the empirical mean from the reward
            for d in xrange(len(path)-1,-1,-1):
                i = path[d]
                if self.tree.is_leaf(i):
                    self.emp_means[i] = \
                            1. / n * reward +  float(n-1) / n * self.emp_means[i]
                    self.values[i] = self.emp_means[i]
                else:
                    self.emp_means[i] = self.empirical_mean_reward(i)
                    leafs = self.tree.get_leafs_of_subtree(i)
                    self.values[i] = max(self.values[leafs])

            # Then we update the m values, upper and lower bounds for 
            # each nodes that lie on the path selected.
            self.update_m_vals(path)
            self.update_uppers(path)
            self.update_lowers(path)

        # Update the sampling complexity.
        self.N += 2

    # Run the algorithm
    def run(self):
        # Continue while there is one arm left
        while True:
            self.t += 1
            if self.t > self.T:
                print "MAX ITER reached."
                break
            path1, path2 = self.select_arm()
            if path1 == path2:
                break
            reward1 = self.tree.arms[path1[-1]].draw()
            reward2 = self.tree.arms[path2[-1]].draw()
            self.update(path1, path2, reward1, reward2)
        self.best_arm = self.opt_path[-1]
        assert(self.best_arm in self.tree.leaf_ids)

    #####
    ## Helper functions for computing the bounds for the algorithm
    #####

    # Get the path that gives the best UCB.
    def get_best_UCB(self, start_node):
        if start_node in self.tree.leaf_ids:
            return [start_node]
        curr_node = start_node
        path_selected = [curr_node]
        all_visited = np.all(self.counts > 0)

        while True:
            # obtain children
            children = self.tree.get_children(curr_node)

            # If there are no children, we have reached a leaf node, so stop.
            if len(children) == 0:
                break

            # Pick the child with bigger empirical mean 
            max_val = -1
            max_node = None
            unvisited = [x for x in children if self.counts[x] == 0]

            if all_visited:
                for child in children:
                    child_val = self.uppers[child]
                    if child_val > max_val:
                        max_val = child_val
                        max_node = child
            elif len(unvisited) == 0:
                max_node = random.choice(children)
            else:
                max_node = random.choice(unvisited)

            # Select child with the maximum bound as the next node
            next_node = max_node

            # Add the selected node to the path
            path_selected.append(next_node)

            # Move to the next node and loop
            curr_node = next_node

        return path_selected

    # Get the path that gives the best LCB.
    def get_best_LCB(self, start_node):
        if start_node in self.tree.leaf_ids:
            return [start_node]
        curr_node = start_node
        path_selected = [curr_node]
        all_visited = np.all(self.counts > 0)

        while True:
            # obtain children
            children = self.tree.get_children(curr_node)

            # If there are no children, we have reached a leaf node, so stop.
            if len(children) == 0:
                break

            # Pick the child with bigger empirical mean 
            max_val = -1
            max_node = None
            unvisited = [x for x in children if self.counts[x] == 0]

            if all_visited:
                for child in children:
                    child_val = self.lowers[child]
                    if child_val > max_val:
                        max_val = child_val
                        max_node = child
            elif len(unvisited) == 0:
                max_node = random.choice(children)
            else:
                max_node = random.choice(unvisited)

            # Select child with the maximum bound as the next node
            next_node = max_node

            # Add the selected node to the path
            path_selected.append(next_node)

            # Move to the next node and loop
            curr_node = next_node

        return path_selected

    # Udpate the m values on the path selected.
    def update_m_vals(self, path):
        for node in reversed(path):
            if node in self.tree.leaf_ids:
                self.m_vals[node] = self.emp_means[node]
            else:
                children = self.tree.get_children(node)
                child_vals = self.emp_means[children]
                self.m_vals[node] = max(max(child_vals), \
                                        self.emp_means[node])

    # Udpate the upper bound on the path selected.
    def update_uppers(self, path):
        for node in reversed(path):
            n = self.counts[node]
            upper = self.method_upper(self.emp_means[node],\
                                        self.conf_method(n, self.beta, self.tree.num_nodes),\
                                        n,\
                                        self.uppers[node])
            if node in self.tree.leaf_ids:
                self.uppers[node] = upper
            else:
                children = self.tree.get_children(node)
                depth = self.tree.get_depth(node)
                child_vals = self.emp_means[children]
                self.uppers[node] =  min(max(child_vals),\
                                    upper + self.delta[depth-1])

    # Udpate the lower bound on the path selected.
    def update_lowers(self, path):
        for node in reversed(path):
            n = self.counts[node]
            lower = self.method_lower(self.emp_means[node],\
                                        self.conf_method(n, self.beta, self.tree.num_nodes),\
                                        n,\
                                        self.lowers[node])
            if node in self.tree.leaf_ids:
                self.lowers[node] = lower
            else:
                children = self.tree.get_children(node)
                child_vals = self.emp_means[children]
                self.lowers[node] = max(max(child_vals), lower)

    # Compute the empirical mean of an arbitrary node 
    def empirical_mean_reward(self, node_id):
        num_visit = self.counts[node_id]
        if self.tree.is_leaf(node_id):
            return self.emp_means[node_id]
        else:
            leafs = self.tree.get_leafs_of_subtree(node_id)
            summed = 0
            for l in leafs:
                summed += self.counts[l] * self.emp_means[l]
            return 1. / self.counts[node_id] * summed
