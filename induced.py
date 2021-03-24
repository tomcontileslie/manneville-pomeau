import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib as tpl

###############################################################################
##  MANNEVILLE-POMEAU by Tom Conti-Leslie                       MIT License  ##
##  induced.py                                                               ##
##                                                                           ##
## Run this Python3 script to generate a plot of the induced map of T_a on   ##
## the interval [0.5, 1]. This uses non-trivial information about the map,   ##
## so unfortunately requires more work in order to be converted into a       ##
## general application (it was initially general but the level of precision  ##
## needed near the origin to produce a reasonable image is high).            ##
##                                                                           ##
## N.B. running this file saves a tex file in the same directory.            ##
###############################################################################

#########################  USER-CHANGEABLE CONSTANTS  #########################
alpha = 0.75  # value of parameter alpha
N = 40        # number of branches to be computed
S = 10        # number of points to compute per branch
F = 10        # figure size (not sure this changes the output, the tikz needs
              # tweaking)
###############################################################################

def mp(x, a = alpha):
    """
    Mannville-Pomeau map, 1 iteration on x
    """
    if x < 0:
        return 0
    if x < 0.5:
        return x * (1 + (2 ** a) * (x ** a))
    if x < 1:
        return 2 * x - 1
    return 1

def mpn(x, n, a = alpha):
    """
    Applies the Manneville-Pomeau map to x, n times.
    """
    if n == 0:
        return x
    return mp(mpn(x, n - 1, a), a)

def mpinv(x, a = alpha, its = 10):
    """
    Approximates the inverse of x under the Manneville-Pomeau map, looking
    for the inverse lying in the first half of the interval.
    Starts with an approximation then applies Newton-Raphson <its> number
    of times.
    """
    # first guess is an approx inverse
    # (linear interpolation between inverse functions at alpha = 0
    # and alpha = 1)
    y = x/2 + a/4 * ((1 + 8 * x)**0.5 - 1 - 2 * x)
    for i in range(its):
        # Newton Raphson
        y = y - (y + (2**a) * y**(a+1) - x) / (1 + (2**a) * (a + 1) * (y**a))
    return y

# build Markov partition starting with x0. Break points are preimages of 1.
X = [1]
for i in range(1, N + 1):
    X.append(mpinv(X[-1]))
# Translate Markov partition into the induced partition (apply inverse of 2x-1)
for i in range(N + 1):
    X[i] = 0.5 * (X[i] + 1)

# cut each induced Markov interval into S points, and evaluate.
branches_x = []
branches_y = []
for i in range(N):
    branches_x.append(np.linspace(X[i+1], X[i], num = 10))
    yvals = []
    # first value should be 0.5. Last should be 1.0
    yvals.append(0.5)
    for x in branches_x[i][1 : S - 1]:
        yvals.append(mpn(x, i + 1)) # return time is i+1 so apply mp, i+1 times 
    yvals.append(1.0)
    branches_y.append(np.array(yvals))

fig = plt.figure(figsize = [F,F])
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.set_xlim(0.5,1.0)
ax.set_aspect(1)
ax.set_ylim(0.5,1.0)

# plot each branch
for i in range(N):
    ax.plot(branches_x[i], branches_y[i], "k-", linewidth=0.7)

tpl.save("mp_induced_75.tex")
