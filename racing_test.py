execfile("core.py")

import random
from datetime import datetime
random.seed(datetime.now())


# instance B1
K = 15
arm1 = BernoulliArm(0.5)
arms = [arm1]
for a in range(2,K+1):
    arm = BernoulliArm(0.5-float(a)/40)
    arms.append(arm)

n_arms = len(arms)
m = 3
eps = 0.04
delta = 0.1
algo = racing(n_arms, m, eps, delta, Hoeffding)
# algo = racing(n_arms, m, eps, delta, Chernoff)

while len(algo.select) < m and len(algo.discard) < n_arms-m:
    # sample all the remaining arms
    chosen_arms = algo.select_arm()
    # print algo.t
    # print chosen_arms
    rewards = np.zeros(len(chosen_arms))
    for arm_id, arm in enumerate(chosen_arms):
        rewards[arm_id] = arms[arm].draw()
    # update
    algo.update(chosen_arms, rewards)

if len(algo.select) == m:
    best_arms = list(algo.select)
else:
    best_arms = list(algo.select.union(algo.remain))

print "total time used: ", algo.t
print "Arms selected: ", algo.select
print "Arms discarded: ", algo.discard
print "Arms remaining: ", algo.remain
print "Final answer: ", best_arms
