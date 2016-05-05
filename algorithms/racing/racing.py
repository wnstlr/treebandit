import numpy as np


def Hoeffding(p, beta, n, upper, lower):
    return p + np.sqrt(beta / (2 * n)), p - np.sqrt(beta / (2 * n))


def beta_racing(t, delta, n_arms):
    return np.log(11.1*n_arms/delta)+1.1*np.log(t)


class racing():
    def __init__(self, n_arms, m, eps, delta, method):
        self.n_arms = n_arms
        self.m = m
        self.eps = eps
        self.delta = delta
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms)
        self.uppers = np.ones(n_arms)
        self.lowers = np.zeros(n_arms)
        self.select = set()
        self.discard = set()
        self.remain = set(range(n_arms))
        self.method = method
        self.t = 1
        return

    def set_parameters(self, counts, values, uppers, lowers, select, discard, remain, t):
        self.counts = counts
        self.values = values
        self.uppers = uppers
        self.lowers = lowers
        self.select = select
        self.discard = discard
        self.remain = remain
        self.t = t
        return

    def select_arm(self):
        return list(self.remain)

    def update(self, chosen_arms, rewards):
        for arm_id, arm in enumerate(chosen_arms):
            self.counts[arm] += 1
            n = self.counts[arm]

            self.values[arm] = ((n - 1) / float(n)) * self.values[arm] + (1 / float(n)) * rewards[arm_id]

            beta = beta_racing(self.t, self.delta, self.n_arms)
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
        if Uut-self.lowers[arm_B] < self.eps or self.uppers[arm_W]-Llt < self.eps:
            if Uut-self.lowers[arm_B] <= self.uppers[arm_W]-Llt:
                self.remain.remove(arm_B)
                self.select.add(arm_B)
            else:
                self.remain.remove(arm_W)
                self.discard.add(arm_W)
        # increase t
        self.t += 1
        return