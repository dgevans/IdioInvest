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
#Para.mu_a = -var_a/2
corr_pers = 0.99

def get_stdev(rho):
    frac = rho/corr_pers
    std_pers = np.sqrt(var_a*frac*(1-corr_pers**2))
    std_iid = np.sqrt(var_a*(1-frac))
    return [std_pers,std_iid]
    
T = 50
Para.k = 304
Para.sigma_E = 0.03
Para.phat[:] = 0.
approximate.calibrate(Para)
Gamma0,weights = cPickle.load(file('Gamma0.dat','r'))
Gamma,Z,Y,Shocks,y = {},{},{},{},{}
Gamma[0] = Gamma0
ss = steadystate.steadystate(zip(Gamma0,weights))
Z[0] = ss.get_Y()[:2]


simulate.simulate_aggstate_ConditionalMean(Para,Gamma,Z,Y,Shocks,y,20)
Gamma[0] = Gamma[19]
Z[0] = Z[19]

simulate.simulate_aggstate_ConditionalMean(Para,Gamma,Z,Y,Shocks,y,T)

Y1 = np.vstack(Y.values())

simulate.simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,2,T0=0,agg_shocks=np.ones(2))
simulate.simulate_aggstate_ConditionalMean(Para,Gamma,Z,Y,Shocks,y,15,1)

Y2 = np.vstack(Y.values())


simulate.simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,500)

Y3 = np.vstack(Y.values())

if rank == 0:
    fout = file('Agg_simulations_shocks2.dat','wr')
    cPickle.dump((Y1,Y2,Y3),fout)
    fout.close()
    
    
    
