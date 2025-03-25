# 2D_EulerCode

This code solves the 2D Euler Equations using the finite volume formulation. Two schemes are implemented to calculate the fluxes in the interior: Lax-Friedrich and Steger Warming. Both schemes are first order and are dissipative in nature. For an example of this code for a supersoning engine inlet please locate the presentation pdf and view it.

## Boundary Conditions

Boundary conditions as implemented apply a supersonic inlet on the left-most boundary, supersonic outlet on the right-most boundary and solid wall boundary conditions on the upper and lower boundary. Future implementations would alter these functions to such that they can be generalized to any setup. 

## Grid Generation

This code also utilizes differential grid generation which solves the elliptical grid generation equation, utilizing an initial grid (more often than not an algebraic grid is best).



## Future Work

* Higher order schemes for calculating interior points
* generalized implementation of boudnary conditions
* Updated implementation of CFL condition
* Updated implementation of Steger-Warming and Lax-Friedrich to run quicker

