import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib as tpl

###############################################################################
##  MANNEVILLE-POMEAU by Tom Conti-Leslie                       MIT License  ##
##  invariant_density.py                                                     ##
##                                                                           ##
## This Python3 script generates an approximation of the invariant density   ##
## of the Manneville-Pomeau map. It does this by first inducing the map      ##
## on the second half of the interval and iterating the Perron-Frobenius     ##
## operator there in order to find an invariant density for the induced      ##
## map. It then lifts this measure to the original interval [0,1] using a    ##
## standard formula, optimised for this map.                                 ##
##                                                                           ##
## Note that iterating the PF operator is time-consuming and reducing the    ##
## constants D, T, M will speed this up. This will have little effect on     ##
## the result, especially for small alpha.                                   ##
##                                                                           ##
## N.B. running this file saves a tex file in the same directory.            ##
###############################################################################

#########################  USER-CHANGEABLE CONSTANTS  #########################
alpha = 0.75  # value of parameter alpha
F = 10        # figure size

# Constants relating to the creation of the Markov partition and storing
# induced mp values
N = 200       # number of Markov branches to be computed
S = 10        # number of points to compute per branch

# Constants relating to the iteration of the density on the induced system
D = 100       # number of points in the density function
T = 20        # number of times to iterate the PF operator
M = 150       # number of Markov branches to be used in the PF operator

# Constants relating to pushing the density
L = 400       # number of points in the density function for [0,1]
P = 100       # number of pre-images to evaluate at each point
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

def rho(x):
    return np.interp(x, xvals, current)

# We now have an invariant density rho on [1/2, 1].
# Push down using the standard formula tailored to this map.
# Start by creating an x-coordinate support.
print("Pushing density onto [0,1]...")
support = np.linspace(0.0, 1.0, L)
# store density points in psi
psi = []
for x in support:
    # to calculate psi(x) we need to evaluate rho at pre-images, and we need
    # the derivative of T^k there too. Save the pre-images of x to avoid
    # re-computing
    
    # initialise running product for derivatives, and running preimage for rho
    running_prod = 2 # two because the points are on right branch, so there is
                     # a default derivative of 2
    running_preim = x
    y = 0 # running sum
    for i in range(P):
        y += rho(mpinv2(running_preim)) / running_prod
        running_preim = mpinv1(running_preim)
        running_prod *= dmp(running_preim)
    psi.append(y)
print("Done")
print("")

print("Integral is", np.trapz(psi, x = support), "; normalising...")
psi = psi / np.trapz(psi, x = support)

fig = plt.figure(figsize = [F,F])
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.set_xlim(0.0,1.0)
#ax.set_aspect(1)
ax.set_ylim(0.0,10.0)  

ax.plot(support, psi, 'k-')

diff = abs(PF_op(support, lambda x : np.interp(x, support, psi), dmp, mpinv, 2) - psi)

print("C^infty difference is", max(diff))
print("C^1     difference is", np.trapz(diff, x = support))
    
tpl.save("invariant_density_mp.tex")
