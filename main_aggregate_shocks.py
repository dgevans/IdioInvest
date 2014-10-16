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
    
T = 50
Para.k = 304
Para.sigma_E = 0.03
Para.phat[:] = 0.
Para.sigma_e[:] = 0.
approximate.calibrate(Para)
Gamma0,weights = cPickle.load(file('Gamma0_2.dat','r'))
Gamma,Z,Y,Shocks,y = {},{},{},{},{}
Gamma[0] = Gamma0
ss = steadystate.steadystate(zip(Gamma0,weights))
Z[0] = ss.get_Y()[:2]
print Z[0]

simulate.simulate_aggstate_ConditionalMean(Para,Gamma,Z,Y,Shocks,y,20,weights=weights)
Gamma[0] = Gamma[19]
Z[0] = Z[19]

simulate.simulate_aggstate_ConditionalMean(Para,Gamma,Z,Y,Shocks,y,20,weights=weights)

Y1 = np.vstack(Y.values())
if rank ==0:
    y1 = np.vstack([y_i[np.newaxis,:,:] for y_i in y.values()])

simulate.simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,2,T0=0,agg_shocks=-np.array([-1,-1.]),weights=weights)
simulate.simulate_aggstate_ConditionalMean(Para,Gamma,Z,Y,Shocks,y,20,1,weights=weights)

Y2 = np.vstack(Y.values())
if rank ==0:
    y2 = np.vstack([y_i[np.newaxis,:,:] for y_i in y.values()])

for t,t0 in [(10,0),(100,9),(500,99),(1000,499),(1500,999),(2000,1499)]:
    simulate.simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,t,T0=t0,weights=weights)
    
    Y3 = np.vstack(Y.values())
    if rank ==0:
        y3 = np.vstack([y_i[np.newaxis,:,:] for y_i in y.values()])
    
    if rank == 0:
        fout = file('Agg_simulations_shock_labor.dat','wr')
        cPickle.dump((Y1,Y2,Y3),fout)
        fout.close()
        fout = file('Agg_simulations_shock__labor_y.dat','wr')
        cPickle.dump((y1,y2,y3),fout)
        fout.close()
    
    
    
