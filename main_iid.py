# -*- coding: utf-8 -*-
import steadystate
import calibrations.calibrate_PR as Para
import approximate_aggstate_test as approximate
import numpy as np
import simulate_MPI as simulate
import utilities
from mpi4py import MPI
import cPickle
import warnings
warnings.filterwarnings('ignore')

simulate.approximate = approximate
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

var_a =  0.10622#np.log(1+1/2.67)
#Para.mu_a = -var_a/2
corr_pers = 0.99

def get_stdev(rho):
    frac = rho/corr_pers
    std_pers = np.sqrt(var_a*frac*(1-corr_pers**2))
    std_iid = np.sqrt(var_a*(1-frac))
    return [std_pers,std_iid]
    
T = 202
N = 15000
Para.k = 48*10

#simulate persistence
data = {}
state = np.random.get_state()
    
def run_sigma_e_experiment():
    if rank == 0:
        utilities.sendMessage('Starting sigma_e')
    Para.beta = 0.99
    for sigma_e in np.linspace(0.01,0.07,6):
        if rank ==0:
            utilities.sendMessage(str(sigma_e))
            print sigma_e
        np.random.set_state(state)
        Para.sigma_e[:2] = [0.,sigma_e]
        
        Gamma,Z,Y,Shocks,y = {},{},{},{},{}
        Gamma[0] = np.zeros((N,3))
        
        steadystate.calibrate(Para)
        ss = steadystate.steadystate(zip(np.zeros((1,3)),np.ones(1)))
        Z[0] = ss.get_Y()[:2]
        
        simulate.simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,T)
        if rank == 0:    
            data[sigma_e] = (np.vstack(Y.values()),[y[t][:5000] for t in range(0,T,50)])
            fout = open('iid_PR_sigma_e.dat','wr')
            cPickle.dump((state,data),fout)
            fout.close()
            
    if rank == 0:
        utilities.sendMessage('Finished sigma_e')

def run_beta_experiment():
    if rank == 0:
        utilities.sendMessage('Starting beta experiment')
    Para.sigma_e[:2] = [0.,0.058]
    for beta in np.linspace(0.97,0.99,6):
        if rank ==0:
            utilities.sendMessage(str(beta))
            print beta
        np.random.set_state(state)
        Para.beta = beta
        
        Gamma,Z,Y,Shocks,y = {},{},{},{},{}
        Gamma[0] = np.zeros((N,3))
        
        steadystate.calibrate(Para)
        ss = steadystate.steadystate(zip(np.zeros((1,3)),np.ones(1)))
        Z[0] = ss.get_Y()[:2]
        
        simulate.simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,T)
        if rank == 0:    
            data[beta] = (np.vstack(Y.values()),[y[t][:5000] for t in range(0,T,50)])
            fout = open('iid_PR_beta.dat','wr')
            cPickle.dump((state,data),fout)
            fout.close()
            
    if rank == 0:
        utilities.sendMessage('Finished beta experiment')

run_sigma_e_experiment()
run_beta_experiment()
