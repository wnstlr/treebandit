import numpy as np
import scipy.optimize as sciopt


# KL divergence function
def klBern(x, y, level):
    return y*np.log(y/x)+(1-y)*np.log((1-y)/(1-x))-level


def klBern_prime(x, y, level):
    return (x-y)/(x*(1-x))


def klBern_prime2(x, y, level):
    return y/x**2 + (1-y)/(1-x)**2


# Hoeffding confidence interval
def Hoeffding(p, beta, n, upper, lower):
    return p + np.sqrt(beta / (2 * n)), p - np.sqrt(beta / (2 * n))


# Chernoff information confidence interval
def Chernoff(p, beta, n, upper, lower):
    dlevel = beta/n
    if p < 1e-14:
        return 1-np.exp(-dlevel), 0.0
    elif p > 1.0 - 1e-14:
        return 1.0, np.exp(-dlevel)
    else:
        # get the upper bound
        upper_curr = sciopt.bisect(f=klBern, a=p, b=1.0, args=(p, dlevel))
        # get the lower bound
        lower_curr = sciopt.bisect(f=klBern, a=0.0, b=p, args=(p, dlevel))

        return upper_curr, lower_curr


def beta_racing(t, delta, n_arms):
    # theoretically, we need alpha > 1, k_1 > 1 + 1/(alpha-1).
    return np.log(11.1*n_arms/delta)+1.1*np.log(t)


class Racing(object):
    def __init__(self, arms, m, eps, delta, exp_rate, lucb_method):
        self.arms = arms
        self.n_arms = len(arms)
        self.m = m
        self.eps = eps
        self.delta = delta
        self.beta = exp_rate
        self.method = lucb_method

        self.counts = np.zeros(self.n_arms, dtype=int)
        self.values = np.zeros(self.n_arms)
        self.uppers = np.ones(self.n_arms)
        self.lowers = np.zeros(self.n_arms)

        self.select = set()
        self.discard = set()
        self.remain = set(range(self.n_arms))

        self.t = 1
        self.N = 0
        self.best_arms = set()

        self.true_best_arms = set()
        self.checkpoints = []
        self.checkerrors = []
        return

    def initialize(self):
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.values = np.zeros(self.n_arms)
        self.uppers = np.ones(self.n_arms)
        self.lowers = np.zeros(self.n_arms)

        self.select = set()
        self.discard = set()
        self.remain = set(range(self.n_arms))

        self.t = 1
        self.N = 0
        self.best_arms = set()

        self.true_best_arms = set()
        self.checkpoints = []
        self.checkerrors = []
        return

    def set_checkpoints(self, checkpoints, best_arms):
        self.checkpoints = np.copy(checkpoints)
        self.true_best_arms = best_arms
        return

    def select_arm(self):
        return list(self.remain)

    def update(self, chosen_arms, rewards):
        for arm_id, arm in enumerate(chosen_arms):
            self.counts[arm] += 1
            n = self.counts[arm]

            self.values[arm] = ((n - 1) / float(n)) * self.values[arm] + (1 / float(n)) * rewards[arm_id]

            beta = self.beta(self.t, self.delta, self.n_arms)
            upper, lower = self.method(self.values[arm], beta, n, self.uppers[arm], self.lowers[arm])
            self.uppers[arm] = upper
            self.lowers[arm] = lower

        # compute J and J^c
        arms_remain = np.array(list(self.remain), dtype=int)
        vals_remain = self.values[arms_remain]
        arms_ind_sort = np.argsort(vals_remain)[::-1]
        J = arms_remain[arms_ind_sort[:(self.m-len(self.select))]]
        Jc = arms_remain[arms_ind_sort[(self.m-len(self.select)):]]
        # compute u_t and l_t
        Uut = np.max(self.uppers[Jc])
        Llt = np.min(self.lowers[J])
        # compute a_B and a_W
        arm_B = arms_remain[np.argmax(vals_remain)]
        arm_W = arms_remain[np.argmin(vals_remain)]
        # try to select or discard
        # we use a = argmin_{a_B, a_W} (Uut-self.lowers[arm_B], self.uppers[arm_W]-Llt) instead of max
        if Uut-self.lowers[arm_B] < self.eps or self.uppers[arm_W]-Llt < self.eps:
            if Uut-self.lowers[arm_B] <= self.uppers[arm_W]-Llt:
                self.remain.remove(arm_B)
                self.select.add(arm_B)
            else:
                self.remain.remove(arm_W)
                self.discard.add(arm_W)
        # increase number of arms drawed and t
        self.N += len(rewards)
        self.t += 1
        return J

    def run(self):
        while len(self.select) < self.m and len(self.discard) < self.n_arms-self.m:
            # sample all the remaining arms
            chosen_arms = self.select_arm()
            rewards = np.zeros(len(chosen_arms))
            for arm_id, arm in enumerate(chosen_arms):
                rewards[arm_id] = self.arms[arm].draw()
            # update
            self.update(chosen_arms, rewards)

        if len(self.select) == self.m:
            self.best_arms = list(self.select)
        else:
            self.best_arms = list(self.select.union(self.remain))
        return

    def run_with_check(self):
        point_to_check = 0

        while len(self.select) < self.m and len(self.discard) < self.n_arms-self.m:
            # sample all the remaining arms
            chosen_arms = self.select_arm()
            rewards = np.zeros(len(chosen_arms))
            for arm_id, arm in enumerate(chosen_arms):
                rewards[arm_id] = self.arms[arm].draw()
            # update
            J = self.update(chosen_arms, rewards)
            # the number of samples is larger than checkpoints[point_to_check], we record the empirical error
            if point_to_check < len(self.checkpoints) and self.N >= self.checkpoints[point_to_check]:
                # current best arms
                curr_best_arms = self.select.union(set(J))
                self.checkerrors.append(curr_best_arms != self.true_best_arms)
                self.checkpoints[point_to_check] = self.N
                point_to_check += 1

        if len(self.select) == self.m:
            self.best_arms = list(self.select)
        else:
            self.best_arms = list(self.select.union(self.remain))
        return
