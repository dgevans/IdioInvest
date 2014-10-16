import steadystate
import calibrations.calibrate_idioinvest_ramsey_frict_shock as Para
from scipy.optimize import root
import numpy as np
import cPickle
from IPython.parallel import Reference
from itertools import imap
import simulate as simulate
from mpi4py import MPI
import approximate_aggstate_test as approximate

rank = MPI.COMM_WORLD.Get_rank()

var_a =  0.10622#np.log(1+1/2.67)np.log(1+1/2.67)/3
Para.mu_a = 0.#-var_a/2
corr_pers = 0.99

def get_stdev(rho):
    frac = rho/corr_pers
    std_pers = np.sqrt(var_a*frac*(1-corr_pers**2))
    std_iid = np.sqrt(var_a*(1-frac))
    return [std_pers,std_iid]

N = 5000
Para.k = 400
steadystate.calibrate(Para)

v = simulate.v
v.block =True
v.execute('import calibrations.calibrate_idioinvest_ramsey_frict_shock as Para')
v.execute('Para.k = 100')
v.execute('import approximate_aggstate_test as approximate')
v['sig'] = get_stdev(0.6)
v.execute('Para.sigma_e[:2] = sig')
v.execute('Para.phat[2] = 0.00')
v.execute('approximate.calibrate(Para)')
v.execute('approximate.shock = 0.')
v.execute('Para.sigma_E = 0.')

v.execute('import numpy as np')
v.execute('state = np.random.get_state()')
data = {}

Gamma,Z,Y,Shocks,y = {},{},{},{},{}
Gamma[0] = np.zeros((N,4))

ss = steadystate.steadystate(zip(np.zeros((1,4)),np.ones(1)))
Z[0] = ss.get_Y()[:1]
simulate.simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,200) #simulate 150 period with no aggregate shocks
    
    
    
