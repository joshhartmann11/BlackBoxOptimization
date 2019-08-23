# Black Box Optimization
Implementations of some common black box optimization techniques used in model calibration.

## Dynamically Dimensioned Search
Written from "https://www.cs.ubc.ca/~hutter/EARG.shtml/earg/papers09/2005WR004723.pdf"

This algorithm performs random normal perturbations on a random ever-decreasing
set of input parameters.  If the random perturbation performs better than the previous
parameters then the perturbation is kept.

## Gradient Descent
Just a classic gradient descent method, using derivatives found through function evaluation and implemented with Adam to speed things up.

## Shuffled Complex Evolution
Written from "https://link.springer.com/article/10.1007/BF00939380"

Starts by generating random input parameters.  The sample is then split into complexes where the parameters are evolved. The samples are then brought together and re-split into new complexes where the process continues.
