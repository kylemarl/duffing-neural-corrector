"""
duffingnc.py: Class implementation of neural-corrector method of Marlantes and Maki (2021,
2022, 2023a, 2023b, 2024) for 1-DOF Duffing equation.

(c) 2024 The Regents of the University of Michigan

This code is licensed under the GNU GPL v3 license, available at https://opensource.org/license/gpl-3-0
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

import keras
from keras.models import Sequential, save_model, load_model
from keras.layers import InputLayer, Dense
import tensorflow as tf

__author__ = "Kyle E. Marlantes"
__license__ = "GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Kyle Marlantes"
__email__ = "kylemarl@umich.edu"
__status__ = "Research"

class Duffing_NC:
    """Implementation of neural-corrector method of Marlantes and Maki (2021,
       2022, 2023a, 2023b, 2024) for 1-DOF Duffing equation.
    
       Duffing model:
           m*xdd + b1*xd + b2*xd^2 + c1*x + c3*x^3 + c5*x^5 = SUM_i (zeta_i - alpha*x)*cos(omega_i*t + phi_i)

       Neural-Corrector model:
           m*xdd = f0 + delta
            
       Time integration is optionally 1st or 2nd order implicit, BDF1 or
       BDF2. ML update for force correction always remains explicit.

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
       """
        
    def __init__(self, m=0., b1=0.,b2=0., c1=0.,c3=0.,c5=0.,alpha=0.):
        # these can be set directly by the caller as a global
        self.m = m # physical mass
        self.b1 = b1 # linear damping coefficient
        self.b2 = b2 # quadratic damping coefficient
        self.c1 = c1 # linear stiffness coefficient
        self.c3 = c3 # cubic stiffness coefficient
        self.c5 = c5 # quintic stiffness coefficient
        self.alpha = alpha # coefficient on nonlinear excitation force
        self.one_way_couple=False # if TRUE couples ML update with approximate nonlinear state, else ML update uses linear state update
        self.ndof=1 # must always be 1 for Duffing equation

    def model_properties (self):
        print('physical mass: m',self.m)   
        print('damping coefficients: b1',self.b1,'b2',self.b2)
        print('stiffness coefficients: c1',self.c1,'c3',self.c3,'c5',self.c5)            
        print('natural frequency', np.sqrt(self.c1/self.m))


    def wave_spectrum (self,Hs0,omega0,ws):
        E = (Hs0**2)/3. # spectrum variance        
        B = (5./4.)*omega0**4
        A = 4*E*B
        return A / ws**5 * np.exp(-B / ws**4 ) # energy spectrum

    def make_waves (self,time0,Hs0,omega0,phase_seed,regular_waves=False):
        """Return time series of wave elevation."""
        if regular_waves:
            ampl = np.array([0.5*Hs0])
            ws = np.array([omega0])
            phase = np.array([0.])
            Sw = np.array([0.]) # just a dummy for return
            
        else:
            # compute sampling bandwidth to ensure repeat period is >= max(time0)
            dw = 2.*np.pi/max(time0) # wave sample bandwidth
            ws = np.arange(omega0/4.,4.*omega0,dw) # wave sample frequencies
            nwaves = len(ws) # number of waves samples        

            np.random.seed(phase_seed)
            phase = np.random.uniform(-np.pi, np.pi, nwaves)
            Sw = self.wave_spectrum(Hs0,omega0,ws)
            ampl = np.sqrt(2.*Sw*dw) # component amplitudes      

        eta=np.zeros(len(time0))
        for i in range(len(ampl)):
            eta+=ampl[i]*np.cos(ws[i]*time0+phase[i])
        return eta,Sw

    def l_excitation (self,wave0):
        """Return linear wave excitation force."""
        eta = wave0        
        return eta
         
    def nl_excitation (self,u_new0,wave0):
        """Return nonlinear wave excitation force."""        
        eta=self.l_excitation(wave0)
        return -self.alpha*u_new0[0]*eta # returns zero if alpha=0 (default)
    
    def solve (self,mode=0,tol=0.0001,forcing_case=0,order=2,trace=True,zero_delta=False,p0=0,v0=0):
        """Main solver routine. Used to make training/testing data, and to invoke neural-corrector method.
        
           mode = 
           0: training data solve, creates linear and nonlinear data
           1: testing data solve, creates linear and nonlinear data
           2: neural-corrector solve, respects coupling setting from train_model call

           order = 
           1: first-order in time BDF1
           2: second-order in time BDF2

           tol = controls tolerance of Newton update, compares to 2-norm of update vector

           forcing_case = can be 0 up to the number of forcing cases - 1. Used to control the case to be solved.

           trace = if TRUE will print trace output to terminal window

           zero_delta = if TRUE will skip ML prediction of force correction delta (for debugging and testing)

           p0,v0 = position, velocity initial conditions
           """        
        Nu=2
        delta=np.zeros([self.ndof])
        
        if mode==0: # make training data
            if trace: print('Solving in training data mode.')
            wave0 = self.train_wave[forcing_case]

            b10=self.b1
            c10=self.c1
                
            max_time0 = self.train_max_time
            dt0 = self.train_dt
            time0 = self.train_time
            
        elif mode==1: # make testing data
            if trace: print('Solving in test data mode.')
            wave0 = self.test_wave[forcing_case]

            b10=self.b1
            c10=self.c1
                
            max_time0 = self.test_max_time
            dt0 = self.test_dt
            time0 = self.test_time
            
        elif mode==2: # NC solve  
            if trace: 
                print('Solving in neural-corrector mode.')
                print('...one way coupling: ',self.one_way_couple) # could call this 'linear update'
            wave0 = self.test_wave[forcing_case]
                
            max_time0 = self.nc_max_time
            dt0 = self.nc_dt
            time0 = self.nc_time   

            # select low-order forcing model
            if self.fmethod=='A':
                b10=self.b1
                c10=self.c1
                include_fwl=True
                 
            elif self.fmethod=='B':
                b10=self.b1
                c10=self.c1
                include_fwl=False
                
            elif self.fmethod=='C':
                b10=0.
                c10=self.c1
                include_fwl=False

            elif self.fmethod=='D':
                b10=0.
                c10=self.c1
                include_fwl=True
                
            elif self.fmethod=='E':
                b10=0.
                c10=0.
                include_fwl=False
                                    
            self.regdatX=np.zeros([1,self.k*self.nfeatures])
            self.regdat=np.zeros([1,self.k,self.ndof])
            self.regdatd=np.zeros([1,self.k,self.ndof])
            self.regdatdd=np.zeros([1,self.k,self.ndof])
            self.regdatw=np.zeros([1,self.k])

        Nt = len(time0)        
            
        def excitation (u_new0,t,n,linear=False):     
            """Return wave excitation forcing on right-hand-side."""
            if (linear) and (mode!=2):
                return self.l_excitation(wave0[n])
            elif mode==2:
                if include_fwl: return self.l_excitation(wave0[n])
                else: return 0.
            else: # mode 0, mode 1, nonlinear
                return self.l_excitation(wave0[n]) + self.nl_excitation(u_new0,wave0[n])

        def predict_delta (t,n,linear=False):
            """Predict new force correction delta using trained ML model. Result is stored."""
            if (zero_delta) or (linear):
                self.nc_delta[forcing_case,n,:] = np.zeros([self.ndof]) 
            else:
                if t>self.k*dt0: 
                    for jj in range(self.k):
                        if self.one_way_couple:
                            self.regdat[:,jj,:]=self.nc_pos_l[forcing_case,n-(self.k-jj)*self.s+(self.s-1)]
                            self.regdatd[:,jj,:]=self.nc_vel_l[forcing_case,n-(self.k-jj)*self.s+(self.s-1)]
                            self.regdatdd[:,jj,:]=self.nc_acc_l[forcing_case,n-(self.k-jj)*self.s+(self.s-1)]            
                            self.regdatw[:,jj]=self.nc_wave[forcing_case,n-(self.k-jj)*self.s+(self.s-1)] 
                        else: # full coupling
                            self.regdat[:,jj,:]=self.nc_pos[forcing_case,n-(self.k-jj)*self.s+(self.s-1)]
                            self.regdatd[:,jj,:]=self.nc_vel[forcing_case,n-(self.k-jj)*self.s+(self.s-1)]
                            self.regdatdd[:,jj,:]=self.nc_acc[forcing_case,n-(self.k-jj)*self.s+(self.s-1)]               
                            self.regdatw[:,jj]=self.nc_wave[forcing_case,n-(self.k-jj)*self.s+(self.s-1)] 

                    # scale 
                    self.regdat = self.transform(self.regdat,self.meanp,self.stdp) # these statistics match the training data
                    self.regdatd = self.transform(self.regdatd,self.meanv,self.stdv)
                    self.regdatdd = self.transform(self.regdatdd,self.meana,self.stda)
                    self.regdatw = (self.regdatw - self.meanw)/self.stdw
                    # many to one
                    self.regdatX[0,0:self.k] = self.regdat[:,:,0]        
                    self.regdatX[0,self.k:2*self.k] = self.regdatd[:,:,0]  
                    self.regdatX[0,2*self.k:3*self.k] = self.regdatdd[:,:,0]      
                    self.regdatX[0,3*self.k:4*self.k] = self.regdatw # note that we scale each feature before stacking!
                    # predict
                    delta0 = self.model_1_graph(self.regdatX, training=False)                
                    # unscale
                    self.nc_delta[forcing_case,n,0] = delta0*self.stdL[0] + self.meanL[0]                

                else:
                    self.nc_delta[forcing_case,n,:] = np.zeros([self.ndof])    

        def get_delta (n):
            """Simple switching routine."""
            if (mode==0) or (mode==1):
                return np.zeros([self.ndof])
            elif mode==2:                
                return self.nc_delta[forcing_case,n,:]

        def compute_q(u_new0,t,n,linear=False):
            """Compute the forcing vector on the right-hand-side which may include ML component."""            
            delta = get_delta(n) # this isn't fully implicit!

            # nonlinear forcing terms (not excitation)
            if (linear) or (mode==2): # do not include nonlinear terms for nc-solve
                fnl=0.
            else:
                fnl = self.b2*u_new0[1]*abs(u_new0[1]) + self.c3*u_new0[0]*abs(u_new0[0]**2.) + self.c5*u_new0[0]*abs(u_new0[0]**4.)
         
            f = excitation(u_new0,t,n,linear=linear) - fnl + delta[0]
            return np.array([0.,f/self.m])

        def compute_G_BDF2 (u_new0,u_now0,u_old0,A0,t,n,linear=False):            
            q_k = compute_q(u_new0,t,n,linear=linear)   
            return (3./(2.*dt0))*u_new0 - (4./(2.*dt0))*u_now0 + (1./(2.*dt0))*u_old0 - np.dot(A0,u_new0) - q_k

        def compute_G_BDF1 (u_new0,u_now0,A0,t,n,linear=False):
            q_k = compute_q(u_new0,t,n,linear=linear)    
            return (1./dt0)*u_new0 - np.dot(A0,u_new0) - (1./dt0)*u_now0 - q_k

        def compute_Gk (u_new0,u_now0,u_old0,A0,t,n,linear=False):
            if order==1: # BDF1
                return compute_G_BDF1 (u_new0,u_now0,A0,t,n,linear=linear)
            elif order==2:
                if n==1: # BDF1 for first step
                    return compute_G_BDF1 (u_new0,u_now0,A0,t,n,linear=linear) 
                else: # BDF2
                    return compute_G_BDF2 (u_new0,u_now0,u_old0,A0,t,n,linear=linear)
                                
        def compute_accel (u_new0,A0,t,n,linear=False):
            q_k = compute_q(u_new0,t,n,linear=linear)
            return np.dot(A0,u_new0) + q_k 
        
        def compute_total_force (pos0,vel0,acc0,t,n,linear=False):
            """Returns the total force on the right-hand-side of the equation of motion, where m*xdd = f_total."""
            fw=excitation(np.array([pos0,vel0]),t,n,linear=linear) # excitation force
            delta = get_delta(n) # ML force correction
            l_terms = -b10*vel0 - c10*pos0 # linear force terms
            if linear: # nonlinear force terms
                nl_terms = 0.
            else:
                nl_terms = -self.b2*vel0*abs(vel0) - self.c3*pos0*abs(pos0**2.) - self.c5*pos0*abs(pos0**4.)
            return l_terms + nl_terms + fw + delta[0]
            
        def NewtonStep (u_new0,u_now0,u_old0,A0,t,n,linear=False):
            """Implicit state update to solve system of nonlinear equations using Newton root-finding technique."""            
            J=np.zeros([Nu,Nu])

            u_new1=u_new0.copy()
            u_now1=u_now0.copy()
            u_old1=u_old0.copy()

            if mode==2: predict_delta(t,n,linear=linear) # explicit update for ML force correction

            err=100.
            inner_loop=0 
            while err>tol: # inner implicit loop
                inner_loop+=1
                G_k0 = compute_Gk(u_new1,u_now1,u_old1,A0,t,n,linear=linear) # unperturbed right-hand-side of system 
                
                # build Jacobian
                u_new00=u_new1.copy()
                du=0.01 # state perturbation for finite difference                
                for i in range(Nu): # for each state variable
                    u_new1[i]+=du # perturb state variable                    
                    G_k = compute_Gk(u_new1,u_now1,u_old1,A0,t,n,linear=linear) # update perturbed right-hand-side of system                   
                    J[:,i] = (G_k - G_k0)/du # finite difference to estimate derivatives
                    u_new1=u_new00.copy() # restore unperturbed state vector

                u_new1=u_new00.copy() # restore unperturbed state vector                
                del_k = np.linalg.solve(-J,G_k0) # solve system            
                u_new1 = u_new1 + del_k # Newton update
                err=np.linalg.norm(del_k,ord=None) # defaults to 2-norm for vectors
                # print('inner_loop',inner_loop,'err',err)

            udot = compute_accel(u_new1,A0,t,n,linear=linear)

            store_data(u_new1,udot,t,n,linear=linear)

            # update state vectors
            # print('u_new1',u_new1,'u_now1',u_now1,'u_old1',u_old1)
            u_old1 = u_now1.copy()
            u_now1 = u_new1.copy()

            return u_new1,u_now1,u_old1

        def store_data (u_new1,udot1,t,n,linear=False):
            """Store advancing solution."""
            if linear: # store linear states
                if mode==0:
                    self.train_pos_l[forcing_case,n]=u_new1[0]
                    self.train_vel_l[forcing_case,n]=u_new1[1]
                    self.train_acc_l[forcing_case,n]=udot1[1] # udot[0] should equal u_new[1]
                    self.train_fl[forcing_case,n]=compute_total_force(self.train_pos_l[forcing_case,n],self.train_vel_l[forcing_case,n],self.train_acc_l[forcing_case,n],t,n,linear=linear)
                elif mode==1:
                    self.test_pos_l[forcing_case,n]=u_new1[0]
                    self.test_vel_l[forcing_case,n]=u_new1[1]
                    self.test_acc_l[forcing_case,n]=udot1[1] # udot[0] should equal u_new[1]
                    self.test_fl[forcing_case,n]=compute_total_force(self.test_pos_l[forcing_case,n],self.test_vel_l[forcing_case,n],self.test_acc_l[forcing_case,n],t,n,linear=linear)
                elif mode==2:
                    self.nc_pos_l[forcing_case,n]=u_new1[0]
                    self.nc_vel_l[forcing_case,n]=u_new1[1]
                    self.nc_acc_l[forcing_case,n]=udot1[1] # udot[0] should equal u_new[1]
                    self.nc_fl[forcing_case,n]=compute_total_force(self.nc_pos_l[forcing_case,n],self.nc_vel_l[forcing_case,n],self.nc_acc_l[forcing_case,n],t,n,linear=linear)                                        
            else: # store nonlinear states
                if mode==0:
                    self.train_pos[forcing_case,n]=u_new1[0]
                    self.train_vel[forcing_case,n]=u_new1[1]
                    self.train_acc[forcing_case,n]=udot1[1] # udot[0] should equal u_new[1]
                    self.train_fh[forcing_case,n]=compute_total_force(self.train_pos[forcing_case,n],self.train_vel[forcing_case,n],self.train_acc[forcing_case,n],t,n)
                elif mode==1:
                    self.test_pos[forcing_case,n]=u_new1[0]
                    self.test_vel[forcing_case,n]=u_new1[1]
                    self.test_acc[forcing_case,n]=udot1[1] # udot[0] should equal u_new[1]
                    self.test_fh[forcing_case,n]=compute_total_force(self.test_pos[forcing_case,n],self.test_vel[forcing_case,n],self.test_acc[forcing_case,n],t,n)
                elif mode==2:
                    self.nc_pos[forcing_case,n]=u_new1[0]
                    self.nc_vel[forcing_case,n]=u_new1[1]
                    self.nc_acc[forcing_case,n]=udot1[1] # udot[0] should equal u_new[1]
                    self.nc_fh[forcing_case,n]=compute_total_force(self.nc_pos[forcing_case,n],self.nc_vel[forcing_case,n],self.nc_acc[forcing_case,n],t,n)                                                
            
        A = np.array([[0,1],[-c10/self.m,-b10/self.m]]) # this matrix will reduce with choices in low-order forcing, see train_model.
        
        u_new,u_now,u_old=np.zeros([Nu]),np.zeros([Nu]),np.zeros([Nu]) # initialize state vectors (initial conditions are zero)
        u_new_l,u_now_l,u_old_l=np.zeros([Nu]),np.zeros([Nu]),np.zeros([Nu])

        u_now[0]=p0
        u_now[1]=v0
        u_now_l[0]=p0
        u_now_l[1]=v0

        if trace:             
            print('...for forcing case ',forcing_case)
            
        for n in range(1,Nt):
            t = time0[n]

            # linear update first (for one way coupling)
            if (mode==0) or (mode==2): # no linear solve for testing data
                u_new_l,u_now_l,u_old_l=NewtonStep(u_new_l,u_now_l,u_old_l,A,t,n,linear=True)
            
            # nonlinear update (incl ML delta prediction, if mode==2)
            u_new,u_now,u_old=NewtonStep(u_new,u_now,u_old,A,t,n)
            if (trace) and (t%(0.1*max_time0)==0):             
                print('...time ',t,' of ',max_time0)

    
    def make_training_data (self,Hs,omegan,max_time,dt,phase_seed=2020,p0=0.,v0=0.,regular_waves=False):
        """Solves the full nonlinear Duffing model for all forcing cases.
           Also computes corresponding linear responses for one-way coupling.
           The number of forcing cases in the training dataset is determined by the 
           length of the gamma and omega arrays.
           
           gamma = array of excitation amplitudes for training data
           
           omega = array of excitation frequencies for training data (must be same length as gamma)
           
           max_time = maximum amount of time in solution
           
           dt = time step for training data solve
           """
        self.n_train_cases = len(Hs)
        self.train_Hs = Hs # sig wave height
        self.train_omega = omegan # peak frequency of distribution
        self.train_phase_seed = phase_seed
        self.train_regular_waves = regular_waves
        
        self.train_max_time = max_time
        self.train_dt = dt
        self.train_time = np.arange(0.,max_time+dt,dt) 

        Nt = len(self.train_time)
        # storage for high-fidelity training responses
        self.train_pos=np.zeros([self.n_train_cases,Nt])
        self.train_vel=np.zeros([self.n_train_cases,Nt])
        self.train_acc=np.zeros([self.n_train_cases,Nt])
        self.train_fh=np.zeros([self.n_train_cases,Nt])
        self.train_wave=np.zeros([self.n_train_cases,Nt])
        self.train_Sw=[]
        # storage for low-fidelity training responses
        self.train_pos_l=np.zeros([self.n_train_cases,Nt])
        self.train_vel_l=np.zeros([self.n_train_cases,Nt])
        self.train_acc_l=np.zeros([self.n_train_cases,Nt])
        self.train_fl=np.zeros([self.n_train_cases,Nt])
        self.train_wave_l=np.zeros([self.n_train_cases,Nt])
        # storage for low-fidelity forcing model (used later for training)
        self.train_f0=np.zeros([self.n_train_cases,Nt])
        
        for n in range(self.n_train_cases):

            self.train_wave[n,:],Sw = self.make_waves(self.train_time,self.train_Hs[n],self.train_omega[n],self.train_phase_seed,regular_waves=regular_waves)
            self.train_Sw.append(Sw)
            self.solve(mode=0,forcing_case=n,p0=p0,v0=v0)
        
    def make_testing_data (self,Hs,omegan,max_time,dt,phase_seed=2020,p0=0.,v0=0.,regular_waves=False):
        """Solves the full nonlinear Duffing model for all forcing cases.
           Testing data is the dataset which is not shown to the ML model during training, 
           but is used to test the performance of the trained model. 
           Transferability of the model can be tested by using gamma and/or omega that
           differ from the training data set.
        
           gamma = array of excitation amplitudes for testing data
           
           omega = array of excitation frequencies for testing data (must be same length as gamma)
           
           max_time = maximum amount of time in solution
           
           dt = time step for testing data solve

           p0,v0 = position, velocity initial conditions
           """
        self.n_test_cases = len(Hs)
        self.test_Hs = Hs
        self.test_omega = omegan
        self.test_phase_seed = phase_seed
        self.test_regular_waves = regular_waves 
                
        self.test_max_time = max_time
        self.test_dt = dt
        self.test_time = np.arange(0.,max_time+dt,dt)

        Nt = len(self.test_time)
        # storage for high-fidelity test responses
        self.test_pos=np.zeros([self.n_test_cases,Nt])
        self.test_vel=np.zeros([self.n_test_cases,Nt])
        self.test_acc=np.zeros([self.n_test_cases,Nt])
        self.test_fh=np.zeros([self.n_test_cases,Nt])
        self.test_wave=np.zeros([self.n_test_cases,Nt])
        self.test_Sw=[]
        
        for n in range(self.n_test_cases):

            self.test_wave[n,:],Sw = self.make_waves(self.test_time,self.test_Hs[n],self.test_omega[n],self.test_phase_seed,regular_waves=regular_waves)
            self.test_Sw.append(Sw)
            
            self.solve(mode=1,forcing_case=n,p0=p0,v0=v0)
            
    def transform(self, dataset, mean, std):
        """Normalize data by mean and standard deviation."""
        v=np.zeros([dataset.shape[0],dataset.shape[1],dataset.shape[2]])
        for i in range(len(mean)):
            if std[i]!=0:
                v[:,:,i]=(dataset[:,:,i] - mean[i])/std[i]
            else:
                v[:,:,i]=dataset[:,:,i] - mean[i]            
        return v

    def train_model (self,dt=0.,method='A',k=5,s=1,layers=2,cells=30,couple=False,verbose=0,epochs=5000,transient_time=0.,load_model_from=''):
        """Trains the ML model for use in neural-corrector method using the training data generated by make_training_data.

           This method offers four different low-fidelity force model options using the 'method' parameter. By changing the 
           low-fidelity force model, it is possible to change what physics the ML model learns, and what physics are retained 
           as analytical terms.

           A low-order forcing model, f0, can be considered as follows:

           m*xdd = delta + f0

           ...where 'delta' is the ML-based force correction term. f0 does not necessarily have to be linear, but all four methods
           available in this method are linear. Additional methods could be added.

           method = 
           'A': f0 will contain all linear physics, including linear stiffness, linear damping, and linear excitation.
                f0_A = -b1*xd - c1*x + gamma*cos(omega*t)
           'B': f0 will contain only linear stiffness and linear damping.
                f0_B = -b1*xd - c1*x
           'C': f0 will contain only linear stiffness.
                f0_C = -c1*x
           'D': f0 will contain linear stiffness and linear excitation.
                f0_D = -c1*x + gamma*cos(omega*t)
           'E': f0 will be zero! This means that the ML model must learn all forces that are not inertial.
                f0_E = 0 (for all t)        

           k = the number of previous timesteps of state (x, xd, xdd) and wave timeseries to use as input into the ML model to predict delta.

           s = the number of timesteps to skip between the k samples used above. Used to space out the stencil over a longer time period 
               but not increase the amount of input data.

           layers = the number of hidden layers in the ML model.

           cells = the number of cells in the hidden layers in the ML model.

           couple =
           True: the state used to update the ML model is the high-fidelity state, i.e. previous solution steps of the neural-ODE.
           False: the state used to update the ML model is the low-fidelity state (which uncouples the solution update).

           verbose = 
           0,1,2: see Keras documentation on model.fit options. If >0 will provide training output on the screen.

           epochs = the maximum number of training epochs. See Keras documentation.

           transient_time = time after which to use for training
           """
        self.one_way_couple=not couple
        self.k=k
        self.s=s
        self.ndof=1
        self.nfeatures=self.ndof*3+1
        self.fmethod=method
        
        if method=='A':
            for n in range(self.n_train_cases):
                fwl = self.l_excitation(self.train_wave[n]) # linear excitation model
                self.train_f0[n] = -self.b1*self.train_vel[n] - self.c1*self.train_pos[n] + fwl
             
        elif method=='B':
            self.train_f0 = -self.b1*self.train_vel - self.c1*self.train_pos
            
        elif method=='C':
            self.train_f0 = -self.c1*self.train_pos
            
        elif method=='D':
            for n in range(self.n_train_cases):
                fwl = self.l_excitation(self.train_wave[n]) # linear excitation model
                self.train_f0[n] = -self.c1*self.train_pos[n] + fwl

        elif method=='E':
            self.train_f0 = self.train_f0*0.
                    
        # compute delta for training data set
        self.train_delta = self.train_fh - self.train_f0

        # check time step
        if dt==0.:
            self.nc_dt=self.train_dt # default to training dt
        else:
            self.nc_dt = dt 
        
        time0=np.arange(0.,self.train_max_time+self.nc_dt,self.nc_dt)
        Nt = len(time0)
        
        train_pos_r=np.zeros([self.n_train_cases,Nt])
        train_vel_r=np.zeros([self.n_train_cases,Nt])
        train_acc_r=np.zeros([self.n_train_cases,Nt])
        train_delta_r = np.zeros([self.n_train_cases,Nt])
        train_wave_r = np.zeros([self.n_train_cases,Nt])
        
        if self.nc_dt!=self.train_dt: # resample data on nc_dt
            # resampling here, store resampled data local to this function
            for n in range(self.n_train_cases):
                if self.one_way_couple: # train on linear responses
                    interp_=interp1d(self.train_time, self.train_pos_l[n], kind='cubic') 
                    train_pos_r[n]=interp_(time0)
                    interp_=interp1d(self.train_time, self.train_vel_l[n], kind='cubic') 
                    train_vel_r[n]=interp_(time0)
                    interp_=interp1d(self.train_time, self.train_acc_l[n], kind='cubic') 
                    train_acc_r[n]=interp_(time0)
                else: # fully-coupled, train on nonlinear responses
                    interp_=interp1d(self.train_time, self.train_pos[n], kind='cubic') 
                    train_pos_r[n]=interp_(time0)
                    interp_=interp1d(self.train_time, self.train_vel[n], kind='cubic') 
                    train_vel_r[n]=interp_(time0)
                    interp_=interp1d(self.train_time, self.train_acc[n], kind='cubic') 
                    train_acc_r[n]=interp_(time0)
                interp_=interp1d(self.train_time, self.train_wave[n], kind='cubic') 
                train_wave_r[n]=interp_(time0)
                interp_=interp1d(self.train_time, self.train_delta[n], kind='cubic') 
                train_delta_r[n]=interp_(time0)
            
        else: # no resampling required, timesteps match
            if self.one_way_couple: # train on linear responses
                train_pos_r = self.train_pos_l
                train_vel_r = self.train_vel_l
                train_acc_r = self.train_acc_l
                train_delta_r = self.train_delta
            else:
                train_pos_r = self.train_pos
                train_vel_r = self.train_vel
                train_acc_r = self.train_acc   
                train_delta_r = self.train_delta
    
            train_wave_r = self.train_wave

        val_split=0.
        fileloc=''
        
        for n in range(self.n_train_cases):
            Xp, Xv, Xa = train_pos_r[n,time0>transient_time], train_vel_r[n,time0>transient_time], train_acc_r[n,time0>transient_time]
            Xw = train_wave_r[n,time0>transient_time]
            y = train_delta_r[n,time0>transient_time]

            NN=Xp.shape[0] 

            Xp_00=np.zeros((NN-k*s-1,k,self.ndof)) # cant use last one because no output data for that
            Xv_00=np.zeros((NN-k*s-1,k,self.ndof))
            Xa_00=np.zeros((NN-k*s-1,k,self.ndof))
            Xw_00=np.zeros((NN-k*s-1,k))
            # many to one
            y_00=np.zeros((NN-k*s-1,1,self.ndof))

            for ii in range(Xp_00.shape[0]): 
                for jj in range(k):
                    Xp_00[ii,jj]=Xp[ii+jj*s]#,:]            
                    Xv_00[ii,jj]=Xv[ii+jj*s]#,:]
                    Xa_00[ii,jj]=Xa[ii+jj*s]#,:]
                    Xw_00[ii,jj]=Xw[ii+jj*s]
                y_00[ii,:]=y[ii+k*s-(s-1)]#,:] # many to one y_00[ii,:]=y[ii+k*s,:]

            if n==0:
                Xp_0=Xp_00; Xv_0=Xv_00; Xa_0=Xa_00; Xw_0=Xw_00; y_0=y_00
            else:
                Xp_0=np.append(Xp_0,Xp_00,axis=0)
                Xv_0=np.append(Xv_0,Xv_00,axis=0)
                Xa_0=np.append(Xa_0,Xa_00,axis=0)
                Xw_0=np.append(Xw_0,Xw_00,axis=0)
                y_0=np.append(y_0,y_00,axis=0)
                        
        self.meanp,self.stdp=np.zeros([self.ndof]),np.zeros([self.ndof])
        self.meanv,self.stdv=np.zeros([self.ndof]),np.zeros([self.ndof])
        self.meana,self.stda=np.zeros([self.ndof]),np.zeros([self.ndof])
        self.meanL,self.stdL=np.zeros([self.ndof]),np.zeros([self.ndof])
        for i in range(self.ndof):
            self.meanp[i], self.stdp[i] = np.mean(Xp_0[:,:,i]), np.std(Xp_0[:,:,i])
            self.meanv[i], self.stdv[i] = np.mean(Xv_0[:,:,i]), np.std(Xv_0[:,:,i])
            self.meana[i], self.stda[i] = np.mean(Xa_0[:,:,i]), np.std(Xa_0[:,:,i])    
            self.meanL[i], self.stdL[i] = np.mean(y_0[:,:,i]), np.std(y_0[:,:,i])
        self.meanw, self.stdw = np.mean(Xw_0), np.std(Xw_0)


        if load_model_from=='': # train new models

            Xp_0 = self.transform(Xp_0, self.meanp, self.stdp) 
            Xv_0 = self.transform(Xv_0, self.meanv, self.stdv) 
            Xa_0 = self.transform(Xa_0, self.meana, self.stda) 
            Xw_0=(Xw_0 - self.meanw)/self.stdw

            y_0 = self.transform(y_0, self.meanL, self.stdL) 

            y_1 = y_0[:,:,0]

            Xp_train, Xv_train, Xa_train, Xw_train = Xp_0, Xv_0, Xa_0, Xw_0
            y_train = y_1

            X_train = np.zeros((Xp_train.shape[0],k*self.nfeatures)) # stack features into a single line, for Dense model
            for idx in range(Xp_train.shape[0]):
                X_train[idx,0:k] = Xp_train[idx,:,0] # position
                X_train[idx,k:2*k] = Xv_train[idx,:,0] # velocity
                X_train[idx,2*k:3*k] = Xa_train[idx,:,0] # acceleration
                X_train[idx,3*k:4*k] = Xw_train[idx,:] # wave (excitation)
                                  
            self.model_1 = Sequential()
            self.model_1.add(InputLayer(input_shape=(k*self.nfeatures,))) # for many to one MLP, stack features
            self.model_1.add(Dense(cells, activation='relu'))#, input_shape=(k, nfeatures)))

            for n in range(layers):
                self.model_1.add(Dense(cells, activation='relu'))

            self.model_1.add(Dense(1, activation='linear'))
            self.model_1.compile(optimizer='adam', loss='mean_squared_error')
            self.model_1._name="Neural-Corrector"
            print(self.model_1.summary())

    ##        callbacks = [keras.callbacks.EarlyStopping(monitor="loss", patience=50, min_delta=0.0001, restore_best_weights=True)]

            # train force model
    ##        self.history_1 = self.model_1.fit(X_train, y_train, epochs=epochs, validation_split=val_split, batch_size=X_train.shape[0], verbose=verbose, callbacks=callbacks)
            self.history_1 = self.model_1.fit(X_train, y_train, epochs=epochs, validation_split=val_split, batch_size=X_train.shape[0], verbose=verbose)

            # training predictions (for verification)
            self.train_pred = self.model_1.predict(X_train)
            self.train_pred = self.train_pred*self.stdL[0] + self.meanL[0]
            self.tmp_train = y_1*self.stdL[0] + self.meanL[0]

        else: # load model
            self.model_1 = load_model(load_model_from)

        # prepare model for use in loop
        self.model_1_graph = tf.function(self.model_1.call, experimental_relax_shapes=True) # could move this to solve
        
    def nc_solve (self,trace=True,zero_delta=False,p0=0.,v0=0.):
        """Solves neural-corrector method over test forcing conditions.
           trace =
           True/False: if True will write trace output to the screen.
           
           zero_delta = 
           True/False: if True will skip ML model update for delta and set delta = 0.

           p0,v0 = position, velocity initial conditions
           """        
        self.nc_max_time = self.test_max_time
        self.nc_time = np.arange(0.,self.nc_max_time+self.nc_dt,self.nc_dt)
        Nt = len(self.nc_time)

        # storage for nc responses
        self.nc_pos=np.zeros([self.n_test_cases,Nt])
        self.nc_vel=np.zeros([self.n_test_cases,Nt])
        self.nc_acc=np.zeros([self.n_test_cases,Nt])
        self.nc_fh=np.zeros([self.n_test_cases,Nt])
        self.nc_wave=self.test_wave #np.zeros([self.n_test_cases,Nt])
        self.nc_delta=np.zeros([self.n_test_cases,Nt,self.ndof])

        # storage for low-fidelity responses computed during update (for one-way coupling)
        self.nc_pos_l=np.zeros([self.n_test_cases,Nt])
        self.nc_vel_l=np.zeros([self.n_test_cases,Nt])
        self.nc_acc_l=np.zeros([self.n_test_cases,Nt])
        self.nc_fl=np.zeros([self.n_test_cases,Nt])
        
        # need to call solve for each testing forcing case
        for n in range(self.n_test_cases):
            self.solve(mode=2,forcing_case=n,zero_delta=zero_delta,p0=p0,v0=v0)

    def compute_norms (self,transient_time=0.):
        """Compute L2, Linf, and JSD on predictions vs test data. Store result.
        
           transient_time = transient time to be excluded from norm calculation.
           """
        self.nc_pos_L2,self.nc_pos_Linf,self.nc_pos_JSD=np.zeros([self.n_test_cases]),np.zeros([self.n_test_cases]),np.zeros([self.n_test_cases])
        self.nc_vel_L2,self.nc_vel_Linf,self.nc_vel_JSD=np.zeros([self.n_test_cases]),np.zeros([self.n_test_cases]),np.zeros([self.n_test_cases])
        self.nc_acc_L2,self.nc_acc_Linf,self.nc_acc_JSD=np.zeros([self.n_test_cases]),np.zeros([self.n_test_cases]),np.zeros([self.n_test_cases])
        self.nc_fh_L2,self.nc_fh_Linf,self.nc_fh_JSD=np.zeros([self.n_test_cases]),np.zeros([self.n_test_cases]),np.zeros([self.n_test_cases])

        # linear model
        self.nc_pos_l_L2,self.nc_pos_l_Linf,self.nc_pos_l_JSD=np.zeros([self.n_test_cases]),np.zeros([self.n_test_cases]),np.zeros([self.n_test_cases])
        self.nc_vel_l_L2,self.nc_vel_l_Linf,self.nc_vel_l_JSD=np.zeros([self.n_test_cases]),np.zeros([self.n_test_cases]),np.zeros([self.n_test_cases])
        self.nc_acc_l_L2,self.nc_acc_l_Linf,self.nc_acc_l_JSD=np.zeros([self.n_test_cases]),np.zeros([self.n_test_cases]),np.zeros([self.n_test_cases])
        self.nc_fl_L2,self.nc_fl_Linf,self.nc_fl_JSD=np.zeros([self.n_test_cases]),np.zeros([self.n_test_cases]),np.zeros([self.n_test_cases])

        def getL2 (pred,truth,time):
            """Calculate and return L2 norm."""
            return np.sqrt(sum((pred[time>transient_time]-truth[time>transient_time])**2)/len(time[time>transient_time]))

        def getLinf (pred,truth,time):
            """Calculate and return Linf norm."""
            return max(abs(pred[time>transient_time]-truth[time>transient_time]))

        def compute_all_norms (pred,truth,time):
            """Compute and return L2, Linf, and JSD for given time series."""            
            L2=getL2(pred,truth,time)
            Linf=getLinf(pred,truth,time)
    
            nbins=50
            hist_pred,binEdges=np.histogram(pred[time>transient_time],bins=nbins, density=True, range=(min(truth),max(truth))) # was pred
            hist_pred_binc = 0.5*(binEdges[1:]+binEdges[:-1])
            hist_truth,binEdges=np.histogram(truth[time>transient_time],bins=nbins, density=True, range=(min(truth),max(truth)))
            hist_truth_binc = 0.5*(binEdges[1:]+binEdges[:-1])            
            JSD = jensenshannon(hist_truth,hist_pred)**2
            
            return L2,Linf,JSD

        time0=self.nc_time
        Nt = len(time0)
        
        test_pos_r=np.zeros([self.n_test_cases,Nt])
        test_vel_r=np.zeros([self.n_test_cases,Nt])
        test_acc_r=np.zeros([self.n_test_cases,Nt])
        test_fh_r = np.zeros([self.n_test_cases,Nt])

        # predictions vs test data
        if self.nc_dt!=self.test_dt: # resample test data on nc_dt
            # resampling here, store resampled data local to this function
            for n in range(self.n_test_cases):
                interp_=interp1d(self.test_time, self.test_pos[n], kind='cubic') 
                test_pos_r[n]=interp_(time0)
                interp_=interp1d(self.test_time, self.test_vel[n], kind='cubic') 
                test_vel_r[n]=interp_(time0)
                interp_=interp1d(self.test_time, self.test_acc[n], kind='cubic') 
                test_acc_r[n]=interp_(time0)
                interp_=interp1d(self.test_time, self.test_fh[n], kind='cubic') 
                test_fh_r[n]=interp_(time0)
                      
        else: # no resampling required, timesteps match
            test_pos_r = self.test_pos
            test_vel_r = self.test_vel
            test_acc_r = self.test_acc   
            test_fh_r = self.test_fh

        for n in range(self.n_test_cases): 
            # neural-corrector predictions
            self.nc_pos_L2[n],self.nc_pos_Linf[n],self.nc_pos_JSD[n]=compute_all_norms(self.nc_pos[n],test_pos_r[n],time0)
            self.nc_vel_L2[n],self.nc_vel_Linf[n],self.nc_vel_JSD[n]=compute_all_norms(self.nc_vel[n],test_vel_r[n],time0)
            self.nc_acc_L2[n],self.nc_acc_Linf[n],self.nc_acc_JSD[n]=compute_all_norms(self.nc_acc[n],test_acc_r[n],time0)
            self.nc_fh_L2[n],self.nc_fh_Linf[n],self.nc_fh_JSD[n]=compute_all_norms(self.nc_fh[n],test_fh_r[n],time0)
            
            # linear model
            self.nc_pos_l_L2[n],self.nc_pos_l_Linf[n],self.nc_pos_l_JSD[n]=compute_all_norms(self.nc_pos_l[n],test_pos_r[n],time0)
            self.nc_vel_l_L2[n],self.nc_vel_l_Linf[n],self.nc_vel_l_JSD[n]=compute_all_norms(self.nc_vel_l[n],test_vel_r[n],time0)
            self.nc_acc_l_L2[n],self.nc_acc_l_Linf[n],self.nc_acc_l_JSD[n]=compute_all_norms(self.nc_acc_l[n],test_acc_r[n],time0)
            self.nc_fl_L2[n],self.nc_fl_Linf[n],self.nc_fl_JSD[n]=compute_all_norms(self.nc_fl[n],test_fh_r[n],time0)
        
