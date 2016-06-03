# Implementation of the numerical experiment from the paper 
# Coquelin, Munos (2007)

execfile("core.py")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def f(x,a):
    return max(3.6 * x * (1-x), 1 - 1. / a * abs(1 - a - x))

beta = 0.1
depth = 10
eps = 0.1

num_intervals = 2 ** depth
domain = np.linspace(0,1,num_intervals+1)
intervals = [[domain[x], domain[x+1]] for x in xrange(len(domain)-1)]
centers = [(x[0] + x[1]) / 2. for x in intervals]

a = 0.1
L = 1 / a
delta = L / 2.
val_opt = 0.9

tree = Tree()
tree.create_tree(depth)

# Set up bernoulli rewards
for i in xrange(len(tree.leaf_ids)):
    leaf = tree.leaf_ids[i]
    tree.nodes[leaf]['mu'] = f(centers[i], a)
tree.setup_smooth_arms_experiment(arm_type='bernoulli')

# Convert the tree to flat
arms = []
for i in xrange(tree.num_nodes):
    if tree.nodes[i]['is_leaf']:
        arms.append(i)
print len(arms)

# Initialize BAST algorithm
bast = BAST_EXP(tree, delta, beta, eps, conf_method, Hoeffding_upper, Hoeffding_lower)
bast.initialize()
bast.run()

xdomain = np.linspace(0,1,num_intervals)
fx = []
for x in xdomain:
    fx.append(f(x,a))

selected1 = bast.counts[tree.leaf_ids] / float(bast.N)
print selected1
opt_loc = xdomain[arms.index(bast.best_arm)]
print opt_loc

pp = PdfPages('figure/fig_bast.pdf')
fig = plt.figure()
plt.plot(xdomain, fx, 'k', label='target')
plt.plot(xdomain, selected1, 'r--', label='estimate(N=%d)'%bast.N)
plt.plot([opt_loc, opt_loc], [0, 1], 'b--', label='best leaf picked')
plt.xlabel('location of the leaf')
plt.ylabel('value')
plt.legend(loc='best')
plt.show()

pp.savefig(fig)
pp.close()
