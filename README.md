Implementation of neural-corrector method of Marlantes and Maki (2021, 2022, 2023, 2024a, 2024b) for 1-DOF Duffing equation.

(c) 2024 The Regents of the University of Michigan

This code is licensed under the GNU GPL v3 license, available at https://opensource.org/license/gpl-3-0

Duffing model:
   m*xdd + b1*xd + b2*xd^2 + c1*x + c3*x^3 + c5*x^5 = SUM_i (zeta_i - alpha*x)*cos(omega_i*t + phi_i)

Neural-Corrector model:
   m*xdd = f0 + delta
    
Time integration is optionally 1st or 2nd order implicit, BDF1 or BDF2. ML update for force correction always remains explicit.

The Duffing equation is discretized as follows:

v = state vector v[0] = x, v[1] = xd
vd = derivative of state vector vd[0] = xd, vd[1] = xdd

A = coefficient matrix = [ 0,      1]
                        [-c1/m, -b1/m]  

q = forcing vector (includes excitation and nonlinear terms) q[0] = 0, q[1] = f/m
   ...where f = SUM_i (zeta_i - alpha*v[0])*cos(omega_i*t + phi_i) + v[0]*cos(omega*t) - b2*v[1]^2 - c3*v[0]^3 - c5*v[0]^5
                        
Written in standard form:

vd = A*v + q

Discretization using implicit finite-differencing:

BDF1:

( v_{n+1} - v_{n} ) / dt = A*v_{n+1} + q_{n+1}

BDF2:

( 3*v_{n+1} - 4*v_{n} + v_{n-1} ) / 2*dt = A*v_{n+1} + q_{n+1}

...where n is the discrete step at the current time, and n+1 is the next step in the solution.

All terms can be collected to the left-hand-side to obtain a discrete equation equal to zero.
This form can be solved using a root finding method, in this case we use a discrete Newton's method.
