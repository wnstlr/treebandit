# Implementation of the numerical experiment from the paper 
# Coquelin, Munos (2007)

execfile("core.py")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

NUM_ITER = 10000

def f(x,a):
    return max(3.6 * x * (1-x), 1 - 1. / a * abs(1 - a - x))

beta = 0.01
depth = 5

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
tree.setup_smooth_arms(val_opt, delta, arm_type='bernoulli')

# Initialize BAST algorithm
bast = BAST(tree, delta, beta)
bast.initialize()

xdomain = np.linspace(0,1,num_intervals)
fx = []
for x in xdomain:
    fx.append(f(x,a))

selected1 = []
for t in xrange(NUM_ITER):
    if t % 100 == 0:
        print t
    chosen_path = bast.select_arm(t)
    chosen_arm = chosen_path[-1]
    reward = tree.arms[chosen_arm].draw()
    bast.update(chosen_path, reward)

selected1 = bast.counts[tree.leaf_ids] / float(NUM_ITER)
#selected = bast.values[tree.leaf_ids]
print selected1

NUM_ITER = 1000000
# Initialize 2nd BAST algorithm
bast = BAST(tree, delta, beta)
bast.initialize()

xdomain = np.linspace(0,1,num_intervals)
fx = []
for x in xdomain:
    fx.append(f(x,a))

selected2 = []
for t in xrange(NUM_ITER):
    if t % 100 == 0:
        print t
    chosen_path = bast.select_arm(t)
    chosen_arm = chosen_path[-1]
    reward = tree.arms[chosen_arm].draw()
    bast.update(chosen_path, reward)

selected2 = bast.counts[tree.leaf_ids] / float(NUM_ITER)
#selected = bast.values[tree.leaf_ids]
print selected2
"""
xdomain = np.linspace(0,1,num_intervals)
fx = []
for x in xdomain:
    fx.append(f(x,a))

selected1 = [ 0.0007 , 0.0015 , 0.0011 , 0.0007 , 0.0031 , 0.0039 , 0.0028 , 0.0065 , 0.0084,
  0.0117 , 0.0093  ,0.04   , 0.0673 , 0.0834,  0.0672,  0.0989 , 0.1583 , 0.0748,
  0.071 ,  0.0591  ,0.0235  ,0.0181 , 0.029  , 0.005  , 0.0079 , 0.002  , 0.0051,
  0.0039 , 0.1084 , 0.0234  ,0.0027  ,0.0013]

selected2 = [  1.10000000e-05   ,1.70000000e-05  , 2.90000000e-05   ,1.90000000e-05,
   3.40000000e-05  , 6.10000000e-05 ,  8.50000000e-05 ,  1.33000000e-04,
   3.06000000e-04  , 5.02000000e-04  , 1.22500000e-03  , 1.86900000e-03,
   4.72300000e-03  , 1.62220000e-02  , 4.45930000e-02  , 1.14976000e-01,
   1.21802000e-01  , 4.95370000e-02  , 1.24410000e-02  , 2.57500000e-03,
   1.73800000e-03  , 4.49000000e-04  , 4.90000000e-04  , 3.58000000e-04,
   1.12000000e-04  , 7.90000000e-05  , 3.10000000e-05  , 1.18000000e-04,
   6.24811000e-01  , 5.91000000e-04  , 5.20000000e-05  , 1.10000000e-05]

"""
pp = PdfPages('figure/fig_bast.pdf')
fig = plt.figure()
plt.plot(xdomain, fx, 'k', label='target')
plt.plot(xdomain, selected1, 'r--', label='estimate(T=10e4)')
plt.plot(xdomain, selected2, 'b--', label='estimate(T=10e6)')
plt.xlabel('location of the leaf')
plt.ylabel('value')
plt.legend(loc='best')
plt.show()

pp.savefig(fig)
pp.close()
