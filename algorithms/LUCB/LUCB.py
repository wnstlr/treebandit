import numpy as np
import scipy.optimize as sciopt

# KL divergence function
def KLdiv(x, y, level):
    return y*np.log(y/x)+(1-y)*np.log((1-y)/(1-x))-level


def KLdiv_prime(x, y, level):
    return (x-y)/(x*(1-x))


def KLdiv_prime2(x, y, level):
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
        upper_curr = sciopt.bisect(f=KLdiv, a=p, b=1.0, args=(p, dlevel))
        # if p < upper < 1.0:
        #     upper_init = upper
        # else:
        #     upper_init = (p+1.0)/2
        # print p, upper_init, dlevel
        # upper_curr = sciopt.newton(func=KLdiv, x0=upper_init, fprime=KLdiv_prime, args=(p, dlevel))
        # get the lower bound
        lower_curr = sciopt.bisect(f=KLdiv, a=0.0, b=p, args=(p, dlevel))
        # if 0.0 < lower < p:
        #     lower_init = lower
        # else:
        #     lower_init = p/2
        # lower_curr = sciopt.newton(func=KLdiv, x0=lower_init, fprime=KLdiv_prime, args=(p, dlevel))

        return upper_curr, lower_curr


def beta_LUCB(t, delta, n_arms):
    # theoretically, we need alpha > 1, k_1 > 1 + 1/(alpha-1).
    a = 405.5*n_arms*np.power(t,1.1)/delta
    return np.log(a)+np.log(np.log(a))


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
        self.true_best_arms = np.copy(true_best_arms)


    def update(self, sample_arms, sample_rewards):

        self.N += len(self.sample_arms)

        self.sample_rewards = sample_rewards
        self.sample_arms = sample_arms
        for arm_id, arm in enumerate(self.sample_arms):
            self.counts[arm] += 1
            n = self.counts[arm]
            self.values[arm] = ((n - 1) / float(n)) * self.values[arm] + (1 / float(n)) * self.sample_rewards[arm_id]

            beta = beta_LUCB(self.t, self.delta, self.n_arms)
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
            flag = 1
            for id, points in enumerate(self.checkpoints):
                while(self.N < points):
                    for arm_id, arm in enumerate(self.sample_arms):
                        self.sample_rewards[arm_id] = self.arms[arm].draw()
                    flag = self.update(self.sample_arms, self.sample_rewards)
                    if flag == False:
                        break
                self.checkerrors[id] = np.array_equal(self.mbest, self.true_best_arms) == False
                if flag == False:
                    break
            while(flag == 1):
                for arm_id, arm in enumerate(self.sample_arms):
                    self.sample_rewards[arm_id] = self.arms[arm].draw()
                flag = self.update(self.sample_arms, self.sample_rewards)
        return

