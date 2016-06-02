execfile("core.py")
import matplotlib.pyplot as plt
from datetime import datetime
random.seed(datetime.now())


def test1_LUCB(num_sims, Klist):
    horizons_mean = np.zeros((len(Klist), 2))
    for kid, k in enumerate(Klist):
        print kid
        m = k/5
        eps = 0.1
        delta = 0.1
        horizons = np.zeros((num_sims,2), dtype=int)
        for sim in range(num_sims):
            means = np.random.random(k)
            arms = map(lambda (mu): BernoulliArm(mu), means)
            caseH = LUCB(arms, m, eps, delta, beta_LUCB, Hoeffding)
            caseH.run()
            caseC = LUCB(arms, m, eps, delta, beta_LUCB, Chernoff)
            caseC.run()
            horizons[sim,0] = caseH.N
            horizons[sim,1] = caseC.N
            print sim
        horizons_mean[kid] = np.mean(horizons, axis=0)
    return horizons_mean

def test23_LUCB(algo, num_sims, check_points, true_bestarms):
    num_points = len(check_points)
    horizons = np.zeros(num_sims)
    checkpoints = np.zeros((num_sims, num_points), dtype=int)
    checkerrors = np.zeros((num_sims, num_points), dtype=bool)
    for sim in range(num_sims):
        print sim
        algo.initialize()
        algo.set_checkpoints(check_points, true_bestarms)
        algo.run()
        horizons[sim] = algo.N
        checkpoints[sim] = algo.checkpoints
        checkerrors[sim] = algo.checkerrors
    return horizons, checkpoints, checkerrors


## test 1
print "test1_LUCB"
num_sims = 100
Klist = range(10,61,10)
horizons_mean = test1_LUCB(num_sims, Klist)

# plot test 1
fig = plt.figure()
plt.plot(Klist, horizons_mean[:,0]/10000, 'ko-', label='LUCB')
plt.plot(Klist, horizons_mean[:,1]/10000, 'bo-', label='KL-LUCB')
plt.legend(loc='best')
plt.title('Expected sample complexity / 10000')
plt.xlabel('K')
plt.savefig('figure/test1_LUCB.png', bbox_inches='tight')
plt.close(fig)




## test23
print "test 23"
num_sims = 100

# B1
K = 15
means = np.array([0.5] + map(lambda (a): 0.5-a/40., range(2,K+1)))
n_arms = len(means)
arms = map(lambda (mu): BernoulliArm(mu), means)
m = 3
eps = 0.04
delta = 0.1
caseH = LUCB(arms, m, eps, delta, beta_LUCB, Hoeffding)
caseC = LUCB(arms, m, eps, delta, beta_LUCB, Chernoff)
checkpoints = np.arange(1000, 7001, 1000)
true_best_arms = [0,1,2]
horizonsH1, checkpointsH1, checkerrorsH1 = test23_LUCB(caseH, num_sims, checkpoints, true_best_arms)
horizonsC1, checkpointsC1, checkerrorsC1 = test23_LUCB(caseC, num_sims, checkpoints, true_best_arms)
errorrateH1 = np.sum(checkerrorsH1, axis=0)/float(num_sims)
errorrateC1 = np.sum(checkerrorsC1, axis=0)/float(num_sims)

# B2
means /= 2
arms = map(lambda (mu): BernoulliArm(mu), means)
m = 3
eps = 0.02
delta = 0.1
caseH = LUCB(arms, m, eps, delta, beta_LUCB, Hoeffding)
caseC = LUCB(arms, m, eps, delta, beta_LUCB, Chernoff)
horizonsH2, checkpointsH2, checkerrorsH2 = test23_LUCB(caseH, num_sims, checkpoints, true_best_arms)
horizonsC2, checkpointsC2, checkerrorsC2 = test23_LUCB(caseC, num_sims, checkpoints, true_best_arms)
errorrateH2 = np.sum(checkerrorsH2, axis = 0)/float(num_sims)
errorrateC2 = np.sum(checkerrorsC2, axis = 0)/float(num_sims)

# plot test 2
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.hist(horizonsH1/10000, bins=10, range=[2.,22.], normed=True, facecolor='red', align='mid', label='LUCB')
plt.hist(horizonsC1/10000, bins=10, range=[2.,22.], normed=True, facecolor='green', align='mid', label='KL-LUCB')
plt.legend(loc='best')
plt.title('Fraction of runs (in bins of width 100)')
plt.subplot(2, 1, 2)
plt.hist(horizonsH2/10000, bins=10, range=[2.,22.], normed=True, facecolor='red', align='mid', label='LUCB')
plt.hist(horizonsC2/10000, bins=10, range=[2.,22.], normed=True, facecolor='green', align='mid', label='KL-LUCB')
plt.legend(loc='best')
plt.xlabel('Samples / 10000')
plt.savefig('figure/test2_LUCB.png', bbox_inches='tight')
plt.close(fig)

# plot test 3
fig = plt.figure()
plt.plot(checkpoints/1000, errorrateH1, 'ko-', label='LUCB')
plt.plot(checkpoints/1000, errorrateC1, 'bo-', label='KL-LUCB')
plt.legend(loc='best')
plt.title('Empirical mistake probability during run')
plt.xlabel('Samples / 100')
plt.savefig('figure/test3_LUCB.png', bbox_inches='tight')
plt.close(fig)
