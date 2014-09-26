# -*- coding: utf-8 -*-
import steadystate
import calibrations.calibrate_idioinvest_ramsey as Para
import approximate_aggstate_noshock as approximate
import numpy as np
import simulate_MPI as simulate
import utilities
from mpi4py import MPI
import cPickle

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

var_a =  np.log(1+1/1.67)
Para.mu_a = -var_a/2
corr_pers = 0.99

def get_stdev(rho):
    frac = rho/corr_pers
    std_pers = np.sqrt(var_a*frac*(1-corr_pers**2))
    std_iid = np.sqrt(var_a*(1-frac))
    return [std_pers,std_iid]
    
T = 202
N = 5000
Para.k = 200

#simulate persistence
data = {}
state = np.random.get_state()
for rho in np.linspace(0.6,0.9,10):
    np.random.set_state(state)
    Para.sigma[:,2] = get_stdev(rho)
    approximate.calibrate(Para)
    
    Gamma,Z,Y,Shocks,y = {},{},{},{},{}
    Gamma[0] = np.zeros((N,4))

    ss = steadystate.steadystate(zip(np.zeros((1,4)),np.ones(1)))
    Z[0] = ss.get_Y()[:1]
    
    simulate.simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,T)    
    data[rho] = (np.vstack(Y.values()),[y[t] for t in range(0,T,50)])
    utilities.sendMessage('Finished persistance: ' + str(0.9))
    
    fout = open('persistance.dat','rw')
    cPickle.dump(data,fout)
    fout.close()
    
    
    
