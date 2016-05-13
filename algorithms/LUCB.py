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
    def __init__(self, n_arms, m, eps, delta, method):
        self.n_arms = n_arms
        self.m = m
        self.eps = eps
        self.delta = delta
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms)
        self.uppers = np.ones(n_arms)
        self.lowers = np.zeros(n_arms)
        self.method = method
        self.mbest = set(range(m))
        self.t = 1
        return

    def set_parameters(self, counts, values, uppers, lowers, select, t):
        self.counts = counts
        self.values = values
        self.uppers = uppers
        self.lowers = lowers
        self.select = select
        self.t = t
        return

    def select_arm(self):
        return list(self.mbest)

    def update(self, chosen_arms, rewards):
        for arm_id, arm in enumerate(chosen_arms):
            self.counts[arm] += 1
            n = self.counts[arm]
            self.values[arm] = ((n - 1) / float(n)) * self.values[arm] + (1 / float(n)) * rewards[arm_id]

            beta = beta_LUCB(self.t, self.delta, self.n_arms)
            upper, lower = self.method(self.values[arm], beta, n, self.uppers[arm], self.lowers[arm])
            self.uppers[arm] = upper
            self.lowers[arm] = lower


        # compute J and J^c
        sorted_values = np.argsort(self.values)[::-1]
        J = sorted_values[:m]
        Jc = sorted_values[m:]
        # compute u_t and l_t
        Uut = np.max(self.uppers[Jc])
        Llt = np.min(self.lowers[J])
        # increase t
        self.t += 1
        # compare to the epsilon
        return Uut - Llt > self.eps

