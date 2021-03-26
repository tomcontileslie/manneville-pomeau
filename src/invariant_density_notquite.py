import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib as tpl

###############################################################################
##  MANNEVILLE-POMEAU by Tom Conti-Leslie                       MIT License  ##
##  invariant_density_notquite.py                                            ##
##                                                                           ##
## The goal of this script is to find an invariant density for the MP map.   ##
## However, it doesn't work for now because I've used a formula (apply       ##
## the transfer operator N times to the constant density 1 and take the      ##
## average) which is not guaranteed to work for non-expanding maps. This     ##
## method doesn't seem to converge here, but the general shape of the result ##
## is still about right so it does illustrate the build-up of mass near the  ##
## indifferent fixed point. However, the next step will be to do this        ##
## properly: use this method for the INDUCED map first, then push it down to ##
## the full interval.                                                        ##
##                                                                           ##
## N.B. running this file saves a tex file in the same directory.            ##
###############################################################################

#########################  USER-CHANGEABLE CONSTANTS  #########################
alpha = 0.75  # value of parameter alpha
N = 500       # number of iterates taken
P = 1000      # number of linspace points in the interval [0,1]
F = 10        # size of output figure
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

def mpinv1(x, a = alpha):
    # calculates inverse of x in [0,1] on first branch of MP map
    # first guess is an approx inverse
    y = x/2 + a/4 * ((1 + 8 * x)**0.5 - 1 - 2 * x)
    for i in range(10):
        # Newton Raphson
        y = y - (y + (2**a) * y**(a+1) - x) / (1 + (2**a) * (a + 1) * (y**a))
    return y

def mpinv2(x, a = alpha):
    # calculated inverse of x in [0,1] on second branch of MP map. This
    # is straightforward and can be done directly
    return 0.5 * (1 + x)

def PF_op_pw(u, df, inverses, x):
    """
    PERRON-FROBENIUS OPERATOR POINTWISE
    
    Arguments:
    - <u> a function with one argument
    - <df> a function: the derivative of the dynamical system function f
      (should take one arg)
    - <inverses> a list of functions, each taking one argument, that find the
      inverse of x under each branch of f
    - <x> a float
    
    Returns:
    - a float, which is the value of PF(u) at the point x -- where PF is the
      PF-operator associated to the system f.
     
    NOTES:
    - Uses a formula for the PF-operator that only works if f is piecewise
      monotonic.
    """
    y = 0
    for inv in inverses:
        z = inv(x)
        y += u(z) / abs(df(z))
    return y

def PF_op(u, df, inverses, xvals):
    """
    PERRON-FROBENIUS OPERATOR
    
    Arguments:
    - <u> a function with one argument
    - <df> a function with one argument (derivative of the dynamics)
    - <inverses> a list of functions with one argument (inverses on branches)
    - <xvals> a numpy array with points at which to evaluate PF(u)
    
    Returns:
    - <yvals> a numpy array with same size as <xvals>, with values of PF(u)
    """
    yvals = []
    for x in xvals:
        yvals.append(PF_op_pw(u, df, inverses, x))
    return np.array(yvals)

# now to define an n-average.
# iterate number 1 here
xvals = np.linspace(0, 1, P)
tot = np.full(P, 1.0)
current = np.full(P, 1.0)

def current_u(x):
    """
    returns the value of the current PF iterate of u at x.
    Uses linear interpolation since current is defined discreetly
    """
    return np.interp(x, xvals, current)

for i in range(N):
    # get new PF iterate
    current = PF_op(current_u, dmp, [mpinv1, mpinv2], xvals)
    tot += current

avg = tot / N

fig = plt.figure(figsize = [F,F])
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.set_xlim(0.0,1.0)
#ax.set_aspect(1)
ax.set_ylim(0.0,10.0)

ax.plot(xvals, avg, 'k-')

tpl.save("invariant_density_notquite.tex")
