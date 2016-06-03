execfile("core.py")
import matplotlib.pyplot as plt


beta = 0.05
eps = 0.1

# Create tree arms
depth = 6
tree = Tree()
tree.create_tree(depth)

results = []
num_sim = 100

tree_trend = []
flat_trend = []

for ns in xrange(num_sim):
    print '---------------[%d]-------------'%ns
    depth = random.randint(4, 10)
    print depth
    tree = Tree()
    tree.create_tree(depth)

    val_opt = random.uniform(0.5, 0.8)
    delta = random.uniform(1, 10)
    tree.setup_smooth_arms(val_opt, delta)

    tree_arms = tree.arms
    flat_arms = []

    for k, v in tree.nodes.iteritems():
        if v['is_leaf']:
            flat_arms.append({'arm': BernoulliArm(v['mu']), 'id': k})

    bast = BAST_EXP(tree, delta, beta, eps, conf_method, Hoeffding_upper, Hoeffding_lower)
    bast.initialize()
    ucb1 = UCB1([], [])
    ucb1.initialize(len(flat_arms))

    #print bast.delta
    #print tree.nodes
    #print tree.leaf_ids

    opt_tree_arm = None
    max_mu = -1
    for k in tree.leaf_ids:
        if tree_arms[k].p > max_mu:
            max_mu = tree_arms[k].p
            opt_tree_arm = k

    print "Optimal Tree Arm = %d"%opt_tree_arm
    opt_flat_arm = [flat_arms.index(x) for x in flat_arms if x['id'] == opt_tree_arm][0]
    print "Optimal Flat Arm = %d"%opt_flat_arm

    max_iter = 1000
    for t in xrange(max_iter):
        # for flat UCB1
        chosen_arm = ucb1.select_arm()
        reward = flat_arms[chosen_arm]['arm'].draw()
        ucb1.update(chosen_arm, reward)

    bast.run()
    best_tree = bast.best_arm
    print best_tree
    print " Samples drawn: %d"%(bast.N)
    print "Found optimal tree arm(s) = %s"%best_tree
    if opt_tree_arm == best_tree:
        print "++ TREE ARM CORRECT!"

    #print ucb1.counts
    #print ucb1.values
    best_ucb = ucb1.values.index(max(ucb1.values))
    print "Found optimal flat arm = %d"%best_ucb
    if opt_flat_arm == best_ucb:
        print "++ FLAT ARM CORRECT!"

