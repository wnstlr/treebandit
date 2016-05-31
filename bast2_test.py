execfile("core.py")
import matplotlib.pyplot as plt

def has_converged(history, window):
    return len(set(history[-window:])) == 1

beta = 0.95

# Create tree arms
depth = 4
tree = Tree()
tree.create_tree(depth)

results = []
num_sim = 100

tree_trend = []
flat_trend = []

for ns in xrange(num_sim):
    print "-------------------"
    depth = random.randint(4, 7)
    print "Depth : %d"%depth
    tree = Tree()
    tree.create_tree(depth)
    val_opt = random.random()
    delta = random.uniform(1, 10)
    tree.setup_smooth_arms(val_opt, delta)

    tree_arms = tree.arms
    flat_arms = []

    for k, v in tree.nodes.iteritems():
        if v['is_leaf']:
            flat_arms.append({'arm': NormalArm(v['mu'], v['sigma']), 'id': k})

    bast = BAST2(tree, delta, beta)
    bast.initialize()
    ucb1 = UCB1([], [])
    ucb1.initialize(len(flat_arms))

    #print bast.delta
    #print tree.nodes
    #print tree.leaf_ids

    opt_tree_arm = None
    max_mu = -1
    for k in tree.leaf_ids:
        if tree_arms[k].mu > max_mu:
            max_mu = tree_arms[k].mu
            opt_tree_arm = k

    print "Optimal Tree Arm = %d"%opt_tree_arm
    opt_flat_arm = [flat_arms.index(x) for x in flat_arms if x['id'] == opt_tree_arm][0]
    print "Optimal Flat Arm = %d"%opt_flat_arm

    max_iter = 1000

    horizon_tree = []
    horizon_flat = []

    history_tree = []
    history_flat = []

    for t in xrange(max_iter):
        # for BAST
        chosen_path1, chosen_path2 = bast.select_arm(t)
        
        # Take the best empirical arm 
        best_emp_means = list(np.where(bast.emp_means==np.mean(bast.emp_means)))
        best_arm = [x for x in best_emp_means if x in bast.tree.leaf_ids][0]

        ## Reward is different based on the situation
        # If the best arm is in one of the chosen paths
        if best_arm in [chosen_path1[-1], chosen_path2[-1]]:
            reward1 = tree.arms[chosen_path1[-1]].draw()
            reward2 = tree.arms[chosen_path1[-1]].draw()
            bast.update(chosen_path1, chosen_path2, reward1, reward2)
            history_tree.append(chosen_path1[-1])
            if len(history_tree) > 5 and has_converged(history_tree, 5):
                horizon_tree.append(t)
        else:
            raise NotImplementedError

        # for flat UCB1
        chosen_arm = ucb1.select_arm()
        reward = flat_arms[chosen_arm]['arm'].draw()
        ucb1.update(chosen_arm, reward)
        history_flat.append(chosen_arm)
        if len(history_flat) > 5 and has_converged(history_flat, 5):
            horizon_flat.append(t)

    #print bast.counts
    #print bast.values
    best_ones = list(np.where(bast.values == max(bast.values[tree.leaf_ids]))[0])
    best_one = [x for x in best_ones if x in tree.leaf_ids][0]
    print "Found optimal tree arm = %d"%best_one
    if opt_tree_arm == best_one:
        print "++ TREE ARM CORRECT!"

    #print ucb1.counts
    #print ucb1.values
    best_ucb = ucb1.values.index(max(ucb1.values))
    print "Found optimal flat arm =%d"%best_ucb
    if opt_flat_arm == best_ucb:
        print "++ FLAT ARM CORRECT!"

    if len(horizon_tree) != 0 and len(horizon_flat) != 0:
        tree_trend.append(horizon_tree[0])
        flat_trend.append(horizon_flat[0])
    else:
        if len(horizon_tree) != 0:
            tree_trend.append(horizon_tree[0])
        if len(horizon_flat) != 0:
            flat_trend.append(horizon_flat[0])
        if len(horizon_tree) == 0:
            tree_trend.append(max_iter)
            horizon_tree.append(max_iter)
        if len(horizon_flat) == 0:
            flat_trend.append(max_iter)
            horizon_flat.append(max_iter)

    results.append(horizon_tree[0] < horizon_flat[0])

count = 0
for x in results:
    if x:
        count += 1

print "Smaller Horizon for Tree : %0.2f%%"%(float(count) / len(results) * 100)

print tree_trend
print flat_trend

fig = plt.figure()
plt.plot(range(num_sim), tree_trend, 'ro-', label='Tree')
plt.plot(range(num_sim), flat_trend, 'bo-', label='Flat')
plt.legend(loc='best')
plt.title('Tree vs Flat on Horizon Length')
plt.xlabel('Trials')
plt.ylabel('Horizon')
plt.savefig('figure/tree_flat.png', bbox_inches='tight')
plt.close(fig)
