execfile("core.py")

arm1 = BernoulliArm(0.7)
arm1.draw()
arm1.draw()

arm2 = NormalArm(10.0, 1.0)
arm2.draw()
arm2.draw()

arm3 = BernoulliArm(0.2)
arm3.draw()
arm3.draw()

arms = [arm1, arm2, arm3]

n_arms = len(arms)

algo1 = EpsilonGreedy(0.1, [], [])
algo2 = Softmax(1.0, [], [])
algo3 = UCB1([], [])
algo4 = Exp3(0.2, [])

algos = [algo1, algo2, algo3, algo4]

for algo in algos:
  algo.initialize(n_arms)

for t in range(1000):
  for algo in algos:
    chosen_arm = algo.select_arm()
    reward = arms[chosen_arm].draw()
    algo.update(chosen_arm, reward)

print algo1.counts
print algo1.values

print algo2.counts
print algo2.values

print algo3.counts
print algo3.values

print algo4.weights

num_sims = 1000
horizon = 10
results = test_algorithm(algo1, arms, num_sims, horizon)
