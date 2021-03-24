# manneville-pomeau
A collection of Python scripts and tex files relative to the family of Manneville-Pomeau interval maps.

Some or all of these are used in my undergraduate final year MMath project on [intermittency in dynamical systems](https://en.wikipedia.org/wiki/Intermittency), 
under supervision by [Dr Mike Todd](http://www.mcs.st-and.ac.uk/~miket/).

I'm using the definition of the Manneville-Pomeau map from
\[[LSV99](https://www.cambridge.org/core/journals/ergodic-theory-and-dynamical-systems/article/abs/probabilistic-approach-to-intermittency/08553B0B2F623A2946507D8A18860D86)\],
where it is defined as a map from \[0,1\] to itself given by:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;T_\alpha(x)=\begin{cases}x(1+2^\alpha{x^\alpha})&x\in[0,1/2)\\2x-1&x\in[1/2,1]\end{cases}" title="Definition of the LSV Manneville-Pomeau map"/>

for a parameter alpha strictly greater than 0.

A number of standard computations are of interest:
- Finding this map's Markov partition
- Inducing this map on the interval \[1/2, 1\] (in order to prove it is an expanding map, for example)
- Calculating the size of the return time tails

My final year project provides and illustrates these computations analytically and numerically in a broad overview of the variety of ergodic results that
can be proven for a canonical map like this one.
