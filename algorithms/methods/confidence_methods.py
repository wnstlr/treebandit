import numpy as np


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


def Hoeffding_upper(p, beta, n, upper):
    return p + np.sqrt(beta / (2 * n))


def Hoeffding_lower(p, beta, n, lower):
    return p - np.sqrt(beta / (2 * n))


# Chernoff information confidence interval
# Chernoff information confidence interval
def Chernoff(p, beta, n, upper, lower, precision=1e-14, maxIterations=50):
    dlevel = beta/n
    if p < 1e-14:
        return 1-np.exp(-dlevel), 0.0
    elif p > 1.0 - 1e-14:
        return 1.0, np.exp(-dlevel)
    else:
        # get the upper bound
        upper_curr = max(upper, p)
        if upper_curr >= 1.0 - 1e-14:
            upper_curr = (upper_curr+p)/2
        solutionFoundUpper = False
        for i in range(maxIterations):
            # Newton step
            upper_next = upper_curr - klBern(upper_curr, p, dlevel)/klBern_prime(upper_curr, p, dlevel)
            if upper_next >= 1.0:
                # if out of upper bound 1, bisection
                upper_next = (upper_curr+1.0)/2
            if abs(upper_next-upper_curr) <= precision and upper_next <= 1.0:
                solutionFoundUpper = True
                upper_curr = upper_next
                break
            upper_curr = upper_next

        # get the lower bound
        lower_curr = min(lower, p)
        if lower_curr <= 1e-14:
            lower_curr = (lower_curr+p)/2
        solutionFoundLower = False
        for i in range(maxIterations):
            # Newton step
            lower_next = lower_curr - klBern(lower_curr, p, dlevel)/klBern_prime(lower_curr, p, dlevel)
            if lower_next <= 0:
                # if out of lower bound 0, bisection
                lower_next = lower_curr/2
            if abs(lower_next-lower_curr) <= precision and lower_next >= 0:
                solutionFoundLower = True
                lower_curr = lower_next
                break
            lower_curr = lower_next

        return upper_curr, lower_curr


def Chernoff_upper(p, beta, n, upper, precision=1e-14, maxIterations=50):
    dlevel = beta/n
    if p < 1e-14:
        return 1-np.exp(-dlevel)
    elif p > 1.0 - 1e-14:
        return 1.0
    else:
        # get the upper bound
        upper_curr = max(upper, p)
        if upper_curr >= 1.0 - 1e-14:
            upper_curr = (upper_curr+p)/2
        solutionFoundUpper = False
        for i in range(maxIterations):
            # Newton step
            upper_next = upper_curr - klBern(upper_curr, p, dlevel)/klBern_prime(upper_curr, p, dlevel)
            if upper_next >= 1.0:
                # if out of upper bound 1, bisection
                upper_next = (upper_curr+1.0)/2
            if abs(upper_next-upper_curr) <= precision and upper_next <= 1.0:
                solutionFoundUpper = True
                upper_curr = upper_next
                break
            upper_curr = upper_next

        return upper_curr


def Chernoff_lower(p, beta, n, lower, precision=1e-14, maxIterations=50):
    dlevel = beta/n
    if p < 1e-14:
        return 0.0
    elif p > 1.0 - 1e-14:
        return np.exp(-dlevel)
    else:
        # get the lower bound
        lower_curr = min(lower, p)
        if lower_curr <= 1e-14:
            lower_curr = (lower_curr+p)/2
        solutionFoundLower = False
        for i in range(maxIterations):
            # Newton step
            lower_next = lower_curr - klBern(lower_curr, p, dlevel)/klBern_prime(lower_curr, p, dlevel)
            if lower_next <= 0:
                # if out of lower bound 0, bisection
                lower_next = lower_curr/2
            if abs(lower_next-lower_curr) <= precision and lower_next >= 0:
                solutionFoundLower = True
                lower_curr = lower_next
                break
            lower_curr = lower_next

        return lower_curr


def beta_racing(t, delta, n_arms):
    # theoretically, we need alpha > 1, k_1 > 1 + 1/(alpha-1).
    return np.log(11.1*n_arms/delta)+1.1*np.log(t)


def beta_lucb(t, delta, n_arms):
    # theoretically, we need alpha > 1, k_1 > 1 + 1/(alpha-1).
    a = 405.5*n_arms*np.power(t,1.1)/delta
    return np.log(a)+np.log(np.log(a))


def conf_method(n, beta, N):
    # theoretically, we need alpha > 1, k_1 > 1 + 1/(alpha-1
    return np.log(2*N*n*(n+1)/beta)


