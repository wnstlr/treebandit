execfile("core.py")
import matplotlib.pyplot as plt
from datetime import datetime
random.seed(datetime.now())


def test1(num_sims, Klist, algo):
    # algo = Racing or LUCB
    if algo == Racing:
        beta = beta_racing
    elif algo == LUCB:
        beta = beta_lucb
    else:
        print "Wrong algorithm input."
    horizons_mean = np.zeros((len(Klist), 2))
    for kid, k in enumerate(Klist):
        print "K = ", kid
        m = k/5
        eps = 0.1
        delta = 0.1
        horizons = np.zeros((num_sims,2), dtype=int)
        for sim in range(num_sims):
            means = np.random.random(k)
            arms = map(lambda (mu): BernoulliArm(mu), means)
            caseH = algo(arms, m, eps, delta, beta, Hoeffding)
            caseH.run()
            caseC = algo(arms, m, eps, delta, beta, Chernoff)
            caseC.run()
            horizons[sim,0] = caseH.N
            horizons[sim,1] = caseC.N
        horizons_mean[kid] = np.mean(horizons, axis=0)
    return horizons_mean

def test23_LUCB(algo, num_sims, check_points, true_bestarms):
    num_points = len(check_points)
    horizons = np.zeros(num_sims)
    checkpoints = np.zeros((num_sims, num_points), dtype=int)
    checkerrors = np.zeros((num_sims, num_points), dtype=bool)
    for sim in range(num_sims):
        algo.initialize()
        algo.set_checkpoints(check_points, true_bestarms)
        algo.run()
        horizons[sim] = algo.N
        checkpoints[sim] = algo.checkpoints
        checkerrors[sim] = algo.checkerrors
    return horizons, checkpoints, checkerrors





# ## test 1
# print "test 1"
# num_sims = 1000
# Klist = range(10,61,10)
# horizons_mean_racing = test1(num_sims, Klist, Racing)
# horizons_mean_LUCB = test1(num_sims, Klist, LUCB)
#
# # plot test 1
# fig = plt.figure()
# plt.plot(Klist, horizons_mean_racing[:,0]/10000, 'ko-', label='Racing')
# plt.plot(Klist, horizons_mean_racing[:,1]/10000, 'bo-', label='KL-Racing')
# plt.plot(Klist, horizons_mean_LUCB[:,0]/10000, 'g^-', label='LUCB')
# plt.plot(Klist, horizons_mean_LUCB[:,1]/10000, 'y^-', label='KL-LUCB')
# plt.legend(loc='best')
# plt.title('Expected sample complexity / 10000')
# plt.xlabel('K')
# plt.savefig('figure/flat_test1_2.png', bbox_inches='tight')
# plt.close(fig)



## test23
print "test 23"
num_sims = 1000
K = 15
means = np.array([0.5] + map(lambda (a): 0.5-a/40., range(2,K+1)))
n_arms = len(means)
true_best_arms = {0, 1, 2}

# # B1
# print "test 23 for B1"
# arms = map(lambda (mu): BernoulliArm(mu), means)
#
# m = 3
# eps = 0.04
# delta = 0.1
# caseH_racing = Racing(arms, m, eps, delta, beta_racing, Hoeffding)
# caseC_racing = Racing(arms, m, eps, delta, beta_racing, Chernoff)
# caseH_LUCB = LUCB(arms, m, eps, delta, beta_lucb, Hoeffding)
# caseC_LUCB = LUCB(arms, m, eps, delta, beta_lucb, Chernoff)
# checkpoints1 = np.arange(1000, 20001, 1000)
# horizonsH1_racing, checkpointsH1_racing, checkerrorsH1_racing = test23(caseH_racing, num_sims, checkpoints1, true_best_arms)
# horizonsC1_racing, checkpointsC1_racing, checkerrorsC1_racing = test23(caseC_racing, num_sims, checkpoints1, true_best_arms)
# horizonsH1_LUCB, checkpointsH1_LUCB, checkerrorsH1_LUCB = test23_LUCB(caseH_LUCB, num_sims, checkpoints1, true_best_arms)
# horizonsC1_LUCB, checkpointsC1_LUCB, checkerrorsC1_LUCB = test23_LUCB(caseC_LUCB, num_sims, checkpoints1, true_best_arms)
# errorrateH1_racing = np.sum(checkerrorsH1_racing, axis=0)/float(num_sims)
# errorrateC1_racing = np.sum(checkerrorsC1_racing, axis=0)/float(num_sims)
# errorrateH1_LUCB = np.sum(checkerrorsH1_LUCB, axis=0)/float(num_sims)
# errorrateC1_LUCB = np.sum(checkerrorsC1_LUCB, axis=0)/float(num_sims)

# B2
print "test 23 for B2"
means /= 2
arms = map(lambda (mu): BernoulliArm(mu), means)

m = 3
eps = 0.02
delta = 0.1
caseH_racing = Racing(arms, m, eps, delta, beta_racing, Hoeffding)
caseC_racing = Racing(arms, m, eps, delta, beta_racing, Chernoff)
caseH_LUCB = LUCB(arms, m, eps, delta, beta_lucb, Hoeffding)
caseC_LUCB = LUCB(arms, m, eps, delta, beta_lucb, Chernoff)
checkpoints2 = np.arange(1000, 50001, 1000)
horizonsH2_racing, checkpointsH2_racing, checkerrorsH2_racing = test23(caseH_racing, num_sims, checkpoints2, true_best_arms)
horizonsC2_racing, checkpointsC2_racing, checkerrorsC2_racing = test23(caseC_racing, num_sims, checkpoints2, true_best_arms)
horizonsH2_LUCB, checkpointsH2_LUCB, checkerrorsH2_LUCB = test23_LUCB(caseH_LUCB, num_sims, checkpoints2, true_best_arms)
horizonsC2_LUCB, checkpointsC2_LUCB, checkerrorsC2_LUCB = test23_LUCB(caseC_LUCB, num_sims, checkpoints2, true_best_arms)
errorrateH2_racing = np.sum(checkerrorsH2_racing, axis=0)/float(num_sims)
errorrateC2_racing = np.sum(checkerrorsC2_racing, axis=0)/float(num_sims)
errorrateH2_LUCB = np.sum(checkerrorsH2_LUCB, axis=0)/float(num_sims)
errorrateC2_LUCB = np.sum(checkerrorsC2_LUCB, axis=0)/float(num_sims)

# # plot test 2
# fig = plt.figure()
# plt.subplot(4, 1, 1)
# plt.hist(horizonsH1_racing/10000, bins=100, range=[2.,22.], normed=True, facecolor='red', align='mid', label='Racing')
# plt.hist(horizonsC1_racing/10000, bins=100, range=[2.,22.], normed=True, facecolor='green', align='mid', label='KL-Racing')
# plt.legend(loc='best')
# plt.title('Fraction of runs (in bins of width 1000)')
# plt.subplot(4, 1, 2)
# plt.hist(horizonsH2_racing/10000, bins=100, range=[2.,22.], normed=True, facecolor='red', align='mid', label='Racing')
# plt.hist(horizonsC2_racing/10000, bins=100, range=[2.,22.], normed=True, facecolor='green', align='mid', label='KL-Racing')
# plt.legend(loc='best')
# plt.subplot(4, 1, 3)
# plt.hist(horizonsH1_LUCB/10000, bins=100, range=[2.,22.], normed=True, facecolor='yellow', align='mid', label='LUCB')
# plt.hist(horizonsC1_LUCB/10000, bins=100, range=[2.,22.], normed=True, facecolor='blue', align='mid', label='KL-LUCB')
# plt.legend(loc='best')
# plt.subplot(4, 1, 4)
# plt.hist(horizonsH2_LUCB/10000, bins=100, range=[2.,22.], normed=True, facecolor='yellow', align='mid', label='LUCB')
# plt.hist(horizonsC2_LUCB/10000, bins=100, range=[2.,22.], normed=True, facecolor='blue', align='mid', label='KL-LUCB')
# plt.legend(loc='best')
# plt.xlabel('Samples / 10000')
# plt.savefig('figure/flat_test2.png', bbox_inches='tight')
# plt.close(fig)

# # plot test 3 for B_1
# fig = plt.figure()
# plt.plot(checkpoints1/1000, errorrateH1_racing, 'ko-', label='Racing')
# plt.plot(checkpoints1/1000, errorrateC1_racing, 'bo-', label='KL-Racing')
# plt.plot(checkpoints1/1000, errorrateH1_LUCB, 'g^-', label='LUCB')
# plt.plot(checkpoints1/1000, errorrateC1_LUCB, 'y^-', label='KL-LUCB')
# plt.legend(loc='best')
# plt.title('Empirical mistake probability during run')
# plt.xlabel('Samples / 1000')
# plt.savefig('figure/flat_test3_B1.png', bbox_inches='tight')
# plt.close(fig)

# plot test 3 for B_2
fig = plt.figure()
plt.plot(checkpoints2/1000, errorrateH2_racing, 'ko-', label='Racing')
plt.plot(checkpoints2/1000, errorrateC2_racing, 'bo-', label='KL-Racing')
plt.plot(checkpoints2/1000, errorrateH2_LUCB, 'g^-', label='LUCB')
plt.plot(checkpoints2/1000, errorrateC2_LUCB, 'y^-', label='KL-LUCB')
plt.legend(loc='best')
plt.title('Empirical mistake probability during run')
plt.xlabel('Samples / 1000')
plt.savefig('figure/flat_test3_B2.png', bbox_inches='tight')
plt.close(fig)