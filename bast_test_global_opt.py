# Implementation of the numerical experiment from the paper 
# Coquelin, Munos (2007)

execfile("core.py")
import matplotlib.pyplot as plt

NUM_ITER = 1000000

def f(x,a):
    return max(3.6 * x * (1-x), 1 - 1. / a * abs(1 - a - x))

beta = 0.95
depth = 5

num_intervals = 2 ** depth
domain = np.linspace(0,1,num_intervals+1)
intervals = [[domain[x], domain[x+1]] for x in xrange(len(domain)-1)]
centers = [(x[0] + x[1]) / 2. for x in intervals]

a = 0.1
L = 1 / a
delta = L / 2.
val_opt = 0.5

tree = Tree()
tree.create_tree(depth)

# Set up bernoulli rewards
for i in xrange(len(tree.leaf_ids)):
    leaf = tree.leaf_ids[i]
    tree.nodes[leaf]['mu'] = f(centers[i], a)
tree.setup_smooth_arms(val_opt, delta, arm_type='bernoulli')

# Initialize BAST algorithm
bast = BAST(tree, delta, beta)
bast.initialize()

xdomain = np.linspace(0,1,num_intervals)
fx = []
for x in xdomain:
    fx.append(f(x,a))

selected = []
for t in xrange(NUM_ITER):
    chosen_path = bast.select_arm(t)
    chosen_arm = chosen_path[-1]
    reward = tree.arms[chosen_arm].draw()
    bast.update(chosen_path, reward)

selected = bast.values[tree.leaf_ids]
print selected

fig = plt.figure()
plt.plot(xdomain, fx, 'k', label='target')
plt.plot(xdomain, selected, 'b', label='estimate')
plt.xlabel('location of the leaf')
plt.ylabel('value')
plt.legend(loc='best')
plt.show()
