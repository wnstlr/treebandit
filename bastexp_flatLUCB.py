execfile("core.py")
import matplotlib.pyplot as plt

def f(x,a):
    return max(3.6 * x * (1-x), 1 - 1. / a * abs(1 - a - x))

num_sims = 10
depth = 7

# global variables for bast
beta_bast = 0.1
eps_bast = 0.1


#global variables for LUCB
m_LUCB = 1
eps_LUCB = 0.1
delta_LUCB = 0.1



Arms_means = []
for i in range(2**depth):
    Arms_means.append(f(i/(2.**depth), 0.1))



# a single LUCB and KL-UCB test, return the averaged number of samplings (horizon) after num_sim simulations.
def LUCB_test(arms_means):
    num_samplings = [0,0]
    #store sampling complexity for LUCB with two types of bounds.
    arms = map(lambda (mu): BernoulliArm(mu), arms_means)
    for i in range(num_sims):
        print "LUCB sim_n =", i
        caseH = LUCB(arms, m_LUCB, eps_LUCB, delta_LUCB, beta_LUCB, Hoeffding)
        caseH.run()
        caseC = LUCB(arms, m_LUCB, eps_LUCB, delta_LUCB, beta_LUCB, Chernoff)
        caseC.run()
        num_samplings[0] += caseH.N/float(num_sims)
        num_samplings[1] += caseC.N/float(num_sims)
    return num_samplings

def BASTEXP_test(delta_bast, arms_means):
    tree = Tree()
    tree.create_tree(depth)
    a = 0.1
    L = 1 / a
    delta_arm = L / 2.
    val_opt = 0.9
    num_samplings = [0,0]
    # Set up bernoulli rewards
    for i in xrange(len(tree.leaf_ids)):
        leaf = tree.leaf_ids[i]
        tree.nodes[leaf]['mu'] = arms_means[i]
    tree.setup_smooth_arms_experiment(arm_type='bernoulli')
    for i in range(num_sims):
        print "sim_n = ", i
        bastH = BAST_EXP(tree, delta_bast, beta_bast, eps_bast, conf_method, Hoeffding_upper, Hoeffding_lower)
        bastH.initialize()
        bastH.run()
        bastC = BAST_EXP(tree, delta_bast, beta_bast, eps_bast, conf_method, Chernoff_upper, Chernoff_lower)
        bastC.initialize()
        bastC.run()
        num_samplings[0] += bastH.N/float(num_sims)
        num_samplings[1] += bastC.N/float(num_sims)
    return num_samplings


## varying delta, compare bastexp and lucb
print "running lucb"
horizons = LUCB_test(Arms_means)
Delta_BAST = 10**np.linspace(-1, 3, 10)
Horizons_bast = np.zeros((len(Delta_BAST), 2))

for i in xrange(len(Delta_BAST)):
    Delta = Delta_BAST[i]
    Horizons_bast[i,:] = np.array(BASTEXP_test(Delta, Arms_means))
"""
for i in range(0, 5):
    for j in range(10):
        Delta = 10**(i-1.)*(1+ j)
        Horizons_bast[idx,:] = BASTEXP_test(Delta, Arms_means)
        print i, j
"""
Horizons_LUCB = np.array([horizons for i in range(len(Delta_BAST))])
Horizons_bast = np.array(Horizons_bast)
Delta_BAST = np.log(Delta_BAST)/np.log(10.)

fig = plt.figure()
plt.plot(Delta_BAST, Horizons_LUCB[:, 0]/100., 'ko-', label='LUCB')
plt.plot(Delta_BAST, Horizons_LUCB[:, 1]/100., 'bo-', label='KL-LUCB')
plt.plot(Delta_BAST, Horizons_bast[:, 0]/100., 'go-', label='BASTEXP')
plt.plot(Delta_BAST, Horizons_bast[:, 1]/100., 'ro-', label='KL-BASTEXP')
plt.legend(loc='best')
plt.title('Expected sample complexity / 100')
plt.xlabel('Ln(delta)')
plt.savefig('figure/bastexp_lucb.png', bbox_inches='tight')
plt.close(fig)




