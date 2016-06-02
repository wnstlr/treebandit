execfile("core.py")
import matplotlib.pyplot as plt

def f(x,a):
    return max(3.6 * x * (1-x), 1 - 1. / a * abs(1 - a - x))

num_sims = 10

# global variables for bast
beta_bast = 0.1
eps_bast = 0.1
depth = 8

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
        print i
        caseH = LUCB(arms, m_LUCB, eps_LUCB, delta_LUCB, beta_LUCB, Hoeffding)
        caseH.run()
        caseC = LUCB(arms, m_LUCB, eps_LUCB, delta_LUCB, beta_LUCB, Chernoff)
        caseC.run()
        num_samplings[0] += caseH.N/num_sims
        num_samplings[1] += caseC.N/num_sims
    return num_samplings

def BASTEXP_test(delta_bast, arms_means):
    tree = Tree()
    tree.create_tree(depth)
    a = 0.1
    L = 1 / a
    delta_arm = L / 2.
    val_opt = 0.9
    num_samplings = 0
    # Set up bernoulli rewards
    for i in xrange(len(tree.leaf_ids)):
        leaf = tree.leaf_ids[i]
        tree.nodes[leaf]['mu'] = f(arms_means[i], a)
    tree.setup_smooth_arms(val_opt, delta_arm, arm_type='bernoulli')
    for i in range(num_sims):
        print i
        bast = BAST_EXP(tree, delta_bast, beta_bast, eps_bast)
        bast.initialize()
        bast.run()
        num_samplings += bast.N/num_sims
    return num_samplings


## varying delta, compare bastexp and lucb
horizons = LUCB_test(Arms_means)
Horizons_bast = []
Delta_BAST = []

for i in range(0, 5):
    for j in range(10):
        Delta = 10**(i-1.)*(1+ j/10.)
        Horizons_bast.append(BASTEXP_test(Delta, Arms_means))
        Delta_BAST.append(Delta)
        print i, j

Horizons_LUCB = np.array([horizons for i in range(len(Delta_BAST))])

fig = plt.figure()
plt.plot(Delta_BAST, Horizons_LUCB[:][0]/100., 'ko-', label='LUCB')
plt.plot(Delta_BAST, Horizons_LUCB[:][1]/100., 'bo-', label='KL-LUCB')
plt.plot(Delta_BAST, Horizons_bast[:]/100., 'go-', label='BAST_EXP')
plt.legend(loc='best')
plt.title('Expected sample complexity / 100')
plt.xlabel('Ln(delta)')
plt.savefig('figure/bastexp_lucb.png', bbox_inches='tight')
plt.close(fig)




Horizons_LUCB = np.array([Horizons_LUCB for i in range(50)])

