# -*- coding: utf-8 -*-
import steadystate
import calibrations.calibrate_idioinvest_ramsey_frict_shock as Para
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
Para.k = 48*7

#simulate persistence
data = {}
state = np.random.get_state()
    
def run_rho_experiment():
    if rank == 0:
        utilities.sendMessage('Starting Persistence')
    for rho in np.linspace(0.6,0.6,1):
        if rank ==0:
            utilities.sendMessage(str(rho))
            print rho
        np.random.set_state(state)
        Para.sigma_e[:2] = get_stdev(rho)
        Para.phat[2] = 0.
        
        Gamma,Z,Y,Shocks,y = {},{},{},{},{}
        Gamma[0] = np.zeros((N,4))
        
        steadystate.calibrate(Para)
        ss = steadystate.steadystate(zip(np.zeros((1,4)),np.ones(1)))
        Z[0] = ss.get_Y()[:2]
        
        simulate.simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,T)
        if rank == 0:    
            data[rho] = (np.vstack(Y.values()),[y[t][:5000] for t in range(0,T,50)])
            fout = open('pers15.dat','wr')
            cPickle.dump((state,data),fout)
            fout.close()
            
    if rank == 0:
        utilities.sendMessage('Finished Persistence')
        utilities.sendMessage('Starting Financial Frictions')
        
    for rho in np.linspace(0.0,0.6,6):
        if rank ==0:
            #utilities.sendMessage(str(rho))
            print rho
        np.random.set_state(state)
        Para.sigma_e[:2] = get_stdev(rho)
        Para.phat[2] = 0.001
        
        Gamma,Z,Y,Shocks,y = {},{},{},{},{}
        Gamma[0] = np.zeros((N,4))
        
        steadystate.calibrate(Para)
        ss = steadystate.steadystate(zip(np.zeros((1,4)),np.ones(1)))
        Z[0] = ss.get_Y()[:2]
        
        simulate.simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,T)
        if rank == 0:    
            data[rho] = (np.vstack(Y.values()),[y[t][:5000] for t in range(0,T,50)])
            fout = open('pers_frict15.dat','wr')
            cPickle.dump((state,data),fout)
            fout.close()
            
    if rank == 0:
        utilities.sendMessage('Finished Frictions')
        
        
def run_friction_experiment():
    if rank == 0:
        utilities.sendMessage('Starting Friction Experiment')
    Para.sigma_e[:2] = get_stdev(0.6)
    for frict in [0.,0.001,0.002,0.004]:
        if rank ==0:
            utilities.sendMessage(str(frict))
            print frict
        np.random.set_state(state)
        Para.phat[2] = frict
        Para.sigma_E = 0.
        
        Gamma,Z,Y,Shocks,y = {},{},{},{},{}
        Gamma[0] = np.zeros((N,4))
        
        steadystate.calibrate(Para)
        ss = steadystate.steadystate(zip(np.zeros((1,4)),np.ones(1)))
        Z[0] = ss.get_Y()[:2]
        
        simulate.simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,T)
        if rank == 0:    
            data[frict] = (np.vstack(Y.values()),[y[t][:5000] for t in range(0,T,50)])
            fout = open('frict15.dat','wr')
            cPickle.dump((state,data),fout)
            fout.close()
            
def run_long_simulation():
    if rank == 0:
        utilities.sendMessage('Starting long simulation')
    N = 50000
    T = 300
    Para.k = 5*16*12
    Para.sigma_e[:2] = get_stdev(0.6)
    Para.phat[2] = 0.
    Para.sigma_E = 0.
    Gamma,Z,Y,Shocks,y = {},{},{},{},{}
    Gamma[0] = np.zeros((N,4))
    
    steadystate.calibrate(Para)
    ss = steadystate.steadystate(zip(np.zeros((1,4)),np.ones(1)))
    Z[0] = ss.get_Y()[:2]
    
    simulate.simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,T)
    if rank == 0:    
        data = (np.vstack(Y.values()),[y[t][:5000] for t in range(0,T,50)])
        fout = open('long_sim.dat','wr')
        cPickle.dump((state,data),fout)
        fout.close()
        utilities.sendMessage('Finished long simulation')
        
def run_span_of_control():
    if rank == 0:
        utilities.sendMessage('Starting span of control')
    N = 15000
    Para.k = 48*10
    Para.sigma_e[:2] = get_stdev(0.6)
    Para.phat[2] = 0.
    Para.sigma_E = 0.
    for nu in np.linspace(0.6,0.88,6):
        np.random.set_state(state)
        if rank ==0:
            utilities.sendMessage(str(nu))
            print nu
        Para.xi_l = 0.66*nu
        Para.xi_k = 0.34*nu
        approximate.calibrate(Para)
        Gamma,Z,Y,Shocks,y = {},{},{},{},{}
        Gamma[0] = np.zeros((N,4))
        
        steadystate.calibrate(Para)
        ss = steadystate.steadystate(zip(np.zeros((1,4)),np.ones(1)))
        Z[0] = ss.get_Y()[:2]
        
        simulate.simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,T)
        if rank == 0:    
            data[nu] = (np.vstack(Y.values()),[y[t][:5000] for t in range(0,T,50)])
            fout = open('span_sim.dat','wr')
            cPickle.dump((state,data),fout)
            fout.close()
            utilities.sendMessage('Finished span of control')
            
def run_rho_experiment_ce():
    #if rank == 0:
    #    utilities.sendMessage('Starting Persistence')
    import calibrations.calibrate_competitive as Para
    Para.k = 48*7
    Para.tau_k = 0.
    Para.tau_l = 0.
    for rho in np.linspace(0.0,0.6,6):
        if rank ==0:
            #utilities.sendMessage(str(rho))
            print rho
        np.random.set_state(state)
        Para.sigma_e[:2] = get_stdev(rho)
        Para.phat[2] = 0.
        
        Gamma,Z,Y,Shocks,y = {},{},{},{},{}
        Gamma[0] = np.zeros((N,3))
        
        steadystate.calibrate(Para)
        ss = steadystate.steadystate(zip(np.zeros((1,3)),np.ones(1)))
        Z[0] = ss.get_Y()[:1]
        
        simulate.simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,T)
        if rank == 0:    
            data[rho] = (np.vstack(Y.values()),[y[t][:5000] for t in range(0,T,50)])
            fout = open('pers15_ce.dat','wr')
            cPickle.dump((state,data),fout)
            fout.close()
            
    if rank == 0:
        utilities.sendMessage('Finished Persistence')
            
run_rho_experiment_ce()
    
