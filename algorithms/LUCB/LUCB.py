import numpy as np


class LUCB():
    def __init__(self, arms, m, eps, delta, beta_method, method):
        self.n_arms = len(arms)
        self.arms = arms
        self.m = m
        self.eps = eps
        self.delta = delta
        self.beta = beta_method
        self.method = method

        self.counts = np.zeros(self.n_arms, dtype=int)
        self.values = np.zeros(self.n_arms)
        self.uppers = np.ones(self.n_arms)
        self.lowers = np.zeros(self.n_arms)
        self.sample_arms = range(self.n_arms)
        self.sample_rewards = np.zeros(self.n_arms)

        self.mbest = range(self.m)
        self.t = 1
        self.N = 0
        self.checkpoints = []
        self.checkerrors = []
        return

    def initialize(self):
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.values = np.zeros(self.n_arms)
        self.uppers = np.ones(self.n_arms)
        self.lowers = np.zeros(self.n_arms)
        self.sample_arms = range(self.n_arms)
        self.sample_rewards = np.zeros(self.n_arms)

        self.t = 1
        self.N = 0
        self.mbest = range(self.m)

        self.true_best_arms = []
        self.checkpoints = []
        self.checkerrors = []
        return

    def set_checkpoints(self, checkpoints, true_best_arms):
        self.checkpoints = np.copy(checkpoints)
        self.checkerrors = np.zeros(len(self.checkpoints))
        self.true_best_arms = true_best_arms


    def update(self, sample_arms, sample_rewards):

        self.N += len(self.sample_arms)

        self.sample_rewards = sample_rewards
        self.sample_arms = sample_arms
        for arm_id, arm in enumerate(self.sample_arms):
            self.counts[arm] += 1
            n = self.counts[arm]
            self.values[arm] = ((n - 1) / float(n)) * self.values[arm] + (1 / float(n)) * self.sample_rewards[arm_id]
            beta = self.beta(self.t, self.delta, self.n_arms)
            upper, lower = self.method(self.values[arm], beta, n, self.uppers[arm], self.lowers[arm])
            self.uppers[arm] = upper
            self.lowers[arm] = lower


        sorted_values = np.argsort(self.values)[::-1]
        J = sorted_values[:self.m]
        Jc = sorted_values[self.m:]
        lt = J[np.argmin(self.lowers[J])]
        ut = Jc[np.argmax(self.uppers[Jc])]
        self.sample_arms = [lt, ut]
        Uut = self.uppers[ut]
        Llt = self.lowers[lt]
        self.mbest = np.copy(J)

        self.t += 1

        return Uut - Llt > self.eps

    def run(self):
        if self.checkpoints == []:
            while(self.t < 1.e6):
                for arm_id, arm in enumerate(self.sample_arms):
                    self.sample_rewards[arm_id] = self.arms[arm].draw()
                if self.update(self.sample_arms, self.sample_rewards) == False:
                    break
            if self.t >= 1.e6:
                print "Reach Max Iterations Steps."
        else:
            flag = True
            for id, points in enumerate(self.checkpoints):
                while(self.N < points):
                    for arm_id, arm in enumerate(self.sample_arms):
                        self.sample_rewards[arm_id] = self.arms[arm].draw()
                    flag = self.update(self.sample_arms, self.sample_rewards)
                    if flag == False:
                        break
                self.checkerrors[id] = set(self.mbest) != self.true_best_arms
                if flag == False:
                    break
            while(flag == True):
                for arm_id, arm in enumerate(self.sample_arms):
                    self.sample_rewards[arm_id] = self.arms[arm].draw()
                flag = self.update(self.sample_arms, self.sample_rewards)
        return

