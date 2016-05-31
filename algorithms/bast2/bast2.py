import numpy as np
import random

class BAST2(object):
    """
    BAST2 Algorithm class 
    Most of it is the same as BAST, but now we keep track of two best arms
    """
    def __init__(self, tree, delta, beta, delta_type="exponential"):
        self.tree = tree
        self.leafs = self.tree.leaf_ids
        self.n_arms = len(self.tree.nodes)
        self.arms = range(self.n_arms)

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
        self.t = 0;

        # max number of iterations
        self.T = 100

    def initialize(self):
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.values = np.zeros(self.n_arms)
        self.emp_means = np.zeros(self.n_arms)

    # Select an arm using the BAST algorithm.
    # Return a whole trajecotry, not just one node
    def select_arm(self, curr_time):
        # for t-th trajectory (iteration)
        curr_node1 = self.tree.root
        curr_node2 = self.tree.root
        path_selected1 = [curr_node1]
        path_selected2 = [curr_node2]
        self.t = curr_time+1
        all_visited = np.all(self.counts > 0)

        while True:
            # list of children of the current node
            children1 = self.tree.get_children(curr_node1)
            children2 = self.tree.get_children(curr_node2)
            children = list(set(children1 + children2))

            # if we reached a leaf node
            if len(children) == 0:
                break

            # Compute bounds for each children of the current node
            unvisited = [x for x in children if self.counts[x] == 0]

            # If all children has been visited at least once, compute
            # the bound values to select the top 2 children
            if all_visited:
                vals = []
                for child in children:
                    child_val = self.compute_bounds(child)
                    vals.append((child, child_val))
                sorted_vals = sorted(vals, key=lambda x: x[1], reverse=True)
                max_node1 = sorted_vals[0][0]
                max_node2 = sorted_vals[1][0]

            # If some children has not been visited yet, arbirarily select
            # a single unvisited node as the next node
            elif len(unvisited) == 0 or len(unvisited) == 1:
                random.shuffle(children)
                max_node1 = children[0]
                max_node2 = children[1]
            else:
                random.shuffle(unvisited)
                max_node1 = unvisited[0]
                max_node2 = unvisited[1]

            # Select child with the top 2 bound as the next nodes
            next_node1 = max_node1
            next_node2 = max_node2

            # Add the selected node to the path
            path_selected1.append(next_node1)
            path_selected2.append(next_node2)

            # Move to the next node and loop
            curr_node1 = next_node1
            curr_node2 = next_node2

        return path_selected1, path_selected2

    def update(self, chosen_path1, chosen_path2, reward1, reward2):
        leaf_id1 = chosen_path1[-1]
        leaf_id2 = chosen_path2[-1]
        self.counts[chosen_path1] += 1
        self.counts[chosen_path2] += 1
        n1 = self.counts[leaf_id1]
        n2 = self.counts[leaf_id2]

        # Start from the leaf and update the values and emp_means for the first path
        for d in xrange(len(chosen_path1)-1,-1,-1):
            i = chosen_path1[d]
            if self.tree.is_leaf(i):
                self.emp_means[i] = \
                        1. / n1 * reward1 +  float(n1-1) / n1 * self.emp_means[i]
                self.values[i] = self.emp_means[i]
            else:
                self.emp_means[i] = self.empirical_mean_reward(i)
                leafs = self.tree.get_leafs_of_subtree(i)
                self.values[i] = max(self.values[leafs])

        # Start from the leaf and update the values and emp_means for the second path
        for d in xrange(len(chosen_path2)-1,-1,-1):
            i = chosen_path2[d]
            if self.tree.is_leaf(i):
                self.emp_means[i] = \
                        1. / n2 * reward2 +  float(n2-1) / n2 * self.emp_means[i]
                self.values[i] = self.emp_means[i]
            else:
                self.emp_means[i] = self.empirical_mean_reward(i)
                leafs = self.tree.get_leafs_of_subtree(i)
                self.values[i] = max(self.values[leafs])


    #####
    ## Helper functions for computing the bounds for the algorithm
    #####

    # Compute the bound as epxressed in the paper
    def compute_bounds(self, node_id):
        cn = self.conf_bound(node_id)
        max_val = -1

        if self.tree.is_leaf(node_id):
            return self.emp_means[node_id] + cn
        else:
            d = self.tree.get_depth(node_id)
            children = self.tree.get_children(node_id)
            for c in children:
                child_val = self.compute_bounds(c)
                if child_val > max_val:
                    max_val = child_val

            val1 = max_val
            val2 = self.emp_means[node_id] + self.delta[d] + cn
            return min(val1, val2)

    # Compute the empirical mean reward of arbitrary node 
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

    # Compute the confidence bound cn for some node
    def conf_bound(self, node_id):
        n = self.counts[node_id]
        return np.sqrt(np.log(2 * self.t * n * (n+1) * 1. / self.beta) / (2*n))
