# -*- coding: utf-8 -*-
import steadystate
import calibrations.calibrate_idioinvest_ramsey_shock as Para
import approximate_aggstate_test as approximate
import numpy as np
import simulate_MPI as simulate
import utilities
from mpi4py import MPI
import cPickle

simulate.approximate = approximate
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

var_a =  0.10622#np.log(1+1/2.67)
Para.mu_a = -var_a/2
corr_pers = 0.99

def get_stdev(rho):
    frac = rho/corr_pers
    std_pers = np.sqrt(var_a*frac*(1-corr_pers**2))
    std_iid = np.sqrt(var_a*(1-frac))
    return [std_pers,std_iid]
    
T = 202
N = 10000
Para.k = 304

#simulate persistence
data = {}
state = np.random.get_state()
if rank == 0:
    #utilities.sendMessage('Starting Persistence')
    fout = open('persistence.dat','wr')
    fout.close()

for rho in np.linspace(0.,0.5,6):
    if rank ==0:
        print rho
    np.random.set_state(state)
    Para.sigma_e[:2] = get_stdev(rho)
    #Para.phat[2] = frict    
    
    Gamma,Z,Y,Shocks,y = {},{},{},{},{}
    Gamma[0] = np.zeros((N,4))
    
    steadystate.calibrate(Para)
    ss = steadystate.steadystate(zip(np.zeros((1,4)),np.ones(1)))
    Z[0] = ss.get_Y()[:2]
    
    simulate.simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,T)
    if rank == 0:    
        data[rho] = (np.vstack(Y.values()),[y[t] for t in range(0,T,50)])
        message = 'Finished persistance: ' + str(rho)
        #utilities.sendMessage(message)
        fout = open('persistence.dat','wr')
        cPickle.dump((state,data),fout)
        fout.close()
        
#if rank == 0:
#    utilities.sendMessage('Finished Persistence')
    
    
    
