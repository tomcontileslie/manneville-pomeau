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
F = 10        # figure size

# Constants relating to the creation of the Markov partition and storing
# induced mp values
N = 100       # number of Markov branches to be computed
S = 100       # number of points to compute per branch

# Constants relating to the iteration of the density
D = 200       # number of points in the density function
T = 5        # number of times to iterate the PF operator (TODO 20)
M = 100       # number of Markov branches to be used in the PF operator

# Constants relating to pushing the density
J = 20        # number of Markov sets to push on
L = M         # number of Markov branches of [0,1] to use for the final density
P = 10        # number of points per branch on the support for final density
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

def dmp(x, a = alpha):
    """
    Derivative of the Manneville-Pomeau map
    """
    if x < 0:
        return 0
    if x < 0.5:
        return 1 + (a + 1) * (2 ** a) * (x ** a)
    if x <= 1:
        return 2
    return 1

def mpinv1(x, a = alpha, its = 10):
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

def mpinv2(x, a = alpha):
    # calculated inverse of x in [0,1] on second branch of MP map. This
    # is straightforward and can be done directly
    return 0.5 * (1 + x)

def mpinv(x, b, a = alpha):
    """
    INVERSE OF THE MANNEVILLE-POMEAU MAP BY BRANCH
    
    Calculates inverse of <x> on branch <b> (0 or 1, counting from right).
    """
    if b == 0:
        return mpinv2(x)
    if b == 1:
        return mpinv1(x)
    # if we made it this far there's an issue
    print("WARNING: inverse MP calculated on unknown branch", b)
    return 0

# construct Markov partition of [1/2, 1]
print("Constructing Markov partition...")
K = [1]
for i in range(1, N + 1):
    K.append(mpinv1(K[-1]))
# Translate Markov partition into the induced partition (apply inverse of 2x-1)
X = []
for i in range(N + 1):
    X.append(0.5 * (K[i] + 1))
print("Done")
print("")
    
# create list of values for each branch. Special case for first branch which
# contains both endpoints.
print("Calculating induced map reference values...")
branches_xs = [np.linspace(0.75, 1, S)]
yvals       = []
for x in branches_xs[0]:
    yvals.append(mp(x))
branches_ys = [np.array(yvals)]
# now populate every other branch
for i in range(1, N):
    branches_xs.append(np.linspace(X[i + 1], X[i], S, endpoint = False))
    yvals = []
    for x in branches_xs[-1]:
        # FIXME floating point error may mean first value is not 0.5
        yvals.append(mpn(x, i + 1)) # return time is i+1
    branches_ys.append(np.array(yvals))
print("Done")
print("")

def mpindinv(x, b, a = alpha):
    """
    MANNEVILLE-POMEAU INDUCED INVERSE
    
    Enter x in [0.5, 1] and branch number b (starting at rightmost, b = 0)
    and this returns an estimate of the inverse of x under the induced map
    on the b^th branch.
    
    To do this, use linear interpolation swapping x and y lists. On each
    branch, the function is increasing (and bijective) so this will work.
    
    FIXME strange behaviour for x = 1.0
    """
    return np.interp(x, branches_ys[b], branches_xs[b])

def findbranch(x, branches = X):
    """
    Takes x and returns number of the branch x lies on.
    Branch numbering should start at rightmost branch (= 0).
    """
    for i in range(len(branches) - 1):
        if branches[i + 1] <= x:
            return i
    # if we got this far x is smaller than all Markov points computed,
    # return one greater than the last interval we have
    # FIXME is that a reasonable last resort output? What are we using this
    # for?
    print("WARNING: couldn't find branch for x =", x)
    return len(branches) - 1

def dmpind(x):
    """
    DERIVATIVE OF THE MANNEVILLE-POMEAU INDUCED MAP
    
    Enter x in [0.5, 1]. Returns derivative of T_A at that point.
    
    Uses iterated chain rule.
    """
    n = findbranch(x)
    # so return time is n + 1
    z = x # we will repeatedly apply mp to z
    prod = dmp(z)
    for i in range(n):
        z = mp(z)
        prod *= dmp(z)
    return float(prod)
    
def PF_op_pw(x, u, df, inverse, nmax):
    """
    PERRON-FROBENIUS OPERATOR POINTWISE
    
    Arguments:
    - <x> a float
    - <u> a function with one argument
    - <df> a function: the derivative of the dynamical system function f
      (should take one arg)
    - <inverses> a parametric function f(x, n) where x is the point that needs
      an inverse, and n is the branch number that the inverse should be
      returned on
    - <nmax> the maximal branch number (will consider inverses on branches 0
      to nmax - 1)
    
    Returns:
    - a float, which is the value of PF(u) at the point x -- where PF is the
      PF-operator associated to the system f.
     
    NOTES:
    - Uses a formula for the PF-operator that only works if f is piecewise
      monotonic.
    """
    y = 0
    for i in range(nmax):
        z = inverse(x, i)
        y += u(z) / abs(df(z))
    return y

def PF_op(xvals, u, df, inverse, nmax):
    """
    PERRON-FROBENIUS OPERATOR
    
    Arguments:
    - <xvals> a numpy array with points at which to evaluate PF(u)
    - <u> a function with one argument
    - <df> a function with one argument (derivative of the dynamics)
    - <inverse> a parametric inverse function f(x, b) (b branch number)
    - <nmax> maximal branch number
    
    Returns:
    - <yvals> a numpy array with same size as <xvals>, with values of PF(u)
    """
    yvals = []
    for x in xvals:
        yvals.append(PF_op_pw(x, u, df, inverse, nmax))
    return np.array(yvals)

# Now start with the constant density 1 and iterate the PF operator as many
# times as specified, hoping it will eventually stabilise
print("Iterating the Perron-Frobenius operator on the induced system...")
xvals = np.linspace(0.5, 1, D)
tot = np.full(D, 1.0)
current = np.full(D, 1.0)

def current_u(x):
    """
    returns the value of the current PF iterate of u at x.
    Uses linear interpolation since current is defined discreetly
    """
    return np.interp(x, xvals, current)

for i in range(T):
    # get new PF iterate
    current = PF_op(xvals, current_u, dmpind, mpindinv, M)
    print(i + 1, "/", T)
print("Done")
print("")

# Create a support of x-coords for the full system. Go from last Markov
# interval so the support is in increasing order
print("Switching to full interval...")
support = np.array([])
for i in range(L - 1, 0, -1):
    support = np.append(support, np.linspace(K[i + 1], K[i], P, endpoint = False))
# special case for rightmost interval which should have the same coords as
# where we were evaluating the prev density
support_size_on_left = len(support)
for i in range(L):
    support = np.append(support, branches_xs[L - i - 1])
# Now convert density into something supported on this x axis
rho = []
for x in support:
    if x <= 0.5:
        rho.append(0.0)
    else:
        rho.append(np.interp(x, xvals, current))
print("Done")
print("")

# So now we have x coordinates for [0,1] and a density function rho on [1/2, 1]
# Use the pushing formula; this involves characteristic functions so let's
# organise that first
def char(x, j):
    """
    CHARACTERISTIC FUNCTION FOR JTH *INDUCED* MARKOV INTERVAL
    """
    if X[j + 1] <= x < X[j]:
        return 1
    else:
        return 0
    
# Now push onto original system
print("Pushing onto [0,1]...")
# empty density
psi = np.full(len(support), 0.0)
# function acting as density
j_density = []
def density_func(x):
    return np.interp(x, support, j_density)
# go through double sum formula
for j in range(J):
    # create rho * X_A_j
    j_density = []
    for x in support:
        # FIXME maybe inefficient
        j_density.append(np.interp(x, support, rho) * char(x, j))
    j_density = np.array(j_density)
    # first step, no application of PF
    psi += j_density
    # every subsequent step, apply PF operator
    for k in range(j):
        j_density = PF_op(support, density_func, dmp, mpinv, 2)
        psi += j_density
    print(j + 1, "/", J)
print("Done")
print("")

# normalise psi
psi = psi / np.trapz(psi, x = support)
        
fig = plt.figure(figsize = [F,F])
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.set_xlim(0.0,1.0)
#ax.set_aspect(1)
ax.set_ylim(0.0,10.0)  

ax.plot(support, psi, 'k-')


















