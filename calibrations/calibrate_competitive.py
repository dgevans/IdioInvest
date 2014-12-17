# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 12:18:27 2014

@author: dgevans
"""
import numpy as np
import pycppad as ad

beta = 0.96
gamma = 2.
sigma = 1.5
#sigma_e = np.array([0.04,0.05,0.1,0.05])
sigma_e = np.array([0.02,0.0,0.,0.])
sigma_E = 0.0*np.eye(1)
mu_e = 0.
mu_a = 0.
chi = 1.
delta = 0.06
xi_l = 0.66*(1-0.21) #*.75
xi_k = 0.34*(1-0.21) #*.75

Gov = 0.17

tau_l = 0.3
tau_k = 0.2



n = 2 # number of measurability constraints
nG = 2 # number of aggregate measurability constraints.
ny = 15 # number of individual controls (m_{t},mu_{t},c_{t},l_{t},rho1_,rho2,phi,x_{t-1},kappa_{t-1}) Note that the forward looking terms are at the end
ne = 4 # number of Expectation Terms (E_t u_{c,t+1}, E_t u_{c,t+1}mu_{t+1} E_{t}x_{t-1 E_t rho_{1,t-1}} [This makes the control x_{t-1},rho_{t-1} indeed time t-1 measuable])
nY = 5 # Number of aggregates (alpha_1,alpha_2,tau,eta,lambda)
nz = 3 # Number of individual states (m_{t-1},mu_{t-1})
nv = 2 # number of forward looking terms (x_t,rho1_t)
n_p = 3 #number of parameters
nEps = 1
nZ = 1 # number of aggregate states
neps = len(sigma_e)

phat = np.array([-0.01,-0.005,0.002])

temp = 1.



def F(w):
    '''
    Individual first order conditions
    '''
    logm,nu_a,nu_e,logc,logl,lognl,logk_,w_e,r,pi,f,b_,labor_res,x_,alpha_ = w[:ny] #y
    EUc,EUc_r,Ex_,Ek_= w[ny:ny+ne] #e
    logK,logAlpha_,R_,W,T = w[ny+ne:ny+ne+nY] #Y
    logm_,nu_a_,nu_e_= w[ny+ne+nY:ny+ne+nY+nz] #z
    x,alpha = w[ny+ne+nY+nz:ny+ne+nY+nz+nv] #v
    nu,nu_l,chi_psi = w[ny+ne+nY+nz+nv:ny+ne+nY+nz+nv+n_p] #v
    logK_ = w[ny+ne+nY+nz+nv+n_p] #Z
    eps_p,eps_t,eps_l_p,eps_l_t = w[ny+ne+nY+nz+nv+n_p+nZ:ny+ne+nY+nz+nv+n_p+nZ+neps] #shock
    Eps = w[ny+ne+nY+nz+nv+n_p+nZ+neps] #aggregate shock
    
    psi_hat,psi = 0.,0.
    try:
        if (ad.value(b_) < 0):
            psi_hat = -2*0.001*(chi_psi-1.) * b_
            psi = -0.001*(chi_psi-1.) * b_**2
    except:
        if (b_ < 0):
            psi_hat = -2*0.001*(chi_psi-1.) * b_
            psi = -0.001*(chi_psi-1.) * b_**2
            
    Alpha_ = np.exp(logAlpha_)
    Ri_ = R_ + psi_hat * W
    m_,m = np.exp(logm_),np.exp(logm)
    c,l,k_,nl = np.exp(logc),np.exp(logl),np.exp(logk_),np.exp(lognl)
    
    
    
    Uc = c**(-sigma)
    Ul = -chi*l**(gamma)
    A = np.exp((1-temp)*Eps*xi_l + nu_a_+eps_p+eps_t)
    fn = A * xi_l * k_**(xi_k) * nl**(xi_l-1)
    fk = A * xi_k * k_**(xi_k-1) * nl**xi_l + (1-delta)
    
    ret = np.empty(ny+n,dtype=w.dtype)
    ret[0] = x_ - Ex_ # x is independent of eps
    ret[1] = k_ - Ek_
    
    ret[2] = x_*Uc/(beta*EUc) + Uc*W*(psi-psi_hat*b_) + Uc*((1-tau_k)*(pi-k_) - (Ri_-1) *k_)  +(1-tau_l)*W*w_e*l*Uc - Uc*(c-T) - x #x
    ret[3] = alpha - m*Uc #rho1
    ret[4] = (1-tau_l)*W*Uc*w_e + Ul #phi1
    ret[5] = r -  (1+(1-tau_k)*(fk-1)) #r
    ret[6] = W - fn #phi2
    ret[7] = pi - ( A*(1-xi_l)*k_**(xi_k)*nl**(xi_l) + (1-delta)*k_   ) #pi
    ret[8] = nu_a - (nu*(nu_a_+eps_p) + (1-nu)*mu_a ) #a
    ret[9] = f - A * k_**(xi_k) * nl**(xi_l) - (1-delta)*k_ #f  
    ret[10] = nu_e - nu_l*(nu_e_ + eps_l_p) - (1-nu_l)*mu_e
    ret[11] = w_e - np.exp(temp*Eps+nu_e_+ eps_l_p + eps_l_t)
    ret[12] = b_ - (x_*m_/Alpha_ - k_)
    ret[13] = labor_res - (l*w_e - nl + psi)
    ret[14] = alpha_ - Alpha_
    
    ret[15] = Alpha_ - beta*Ri_*m_*EUc #rho2_
    ret[16] = Alpha_ - beta*m_*EUc_r #rho3
    
    return ret
    
def G(w):
    '''
    Aggregate equations
    '''
    logm,nu_a,nu_e,logc,logl,lognl,logk_,w_e,r,pi,f,b_,labor_res,x_,alpha_ = w[:ny] #y
    logK,logAlpha_,R_,W,T = w[ny+ne:ny+ne+nY] #Y
    logm_,nu_a_,nu_e_= w[ny+ne+nY:ny+ne+nY+nz] #z
    logK_ = w[ny+ne+nY+nz+nv+n_p] #Z
    Eps = w[ny+ne+nY+nz+nv+n_p+nZ+neps] #aggregate shock
    
    c,l,k_ = np.exp(logc),np.exp(logl),np.exp(logk_)
    K_ = np.exp(logK_)
    K = np.exp(logK)
    
    
    ret = np.empty(nY+nG,dtype=w.dtype)
    ret[0] = R_
    ret[1] = logAlpha_
    
    ret[2] = labor_res
    ret[3] = logm
    ret[4] = T + Gov
    
    ret[5] = f -c -K-Gov # no government debt
    ret[6] = k_ - K_ #capital stock
    
    
    return ret
    
def f(y):
    '''
    Expectational equations that define e=Ef(y)
    '''
    logm,nu_a,nu_e,logc,logl,lognl,logk_,w_e,r,pi,f,b_,labor_res,x_,alpha_ = y
    
    c,k_ = np.exp(logc),np.exp(logk_)
    Uc = c**(-sigma)
    
            
    #EUc,EUc_r,Ea_,Ek_,Emu,Erho2_,Erho3_,Efoc_k
    ret = np.empty(ne,dtype=y.dtype)
    ret[0] = Uc
    ret[1] = Uc * r
    ret[2] = x_
    ret[3] = k_
        
    
    return ret
    
def Finv(YSS,z):
    '''
    Given steady state YSS solves for y_i
    '''
    logm,nu_a,nu_e = z
    logK,logAlpha_,R_,W,T  = YSS
    
    Alpha_ = np.exp(logAlpha_)
        
    
    m = np.exp(logm)
    w_e = np.exp(nu_e)
    
    Uc =Alpha_/m
    c = (Uc)**(-1/sigma)
    
    Ul = -(1-tau_l)*Uc*W*w_e
    l = ( -Ul/chi )**(1./gamma)
    
    r = 1./beta * np.ones(c.shape)
    A = np.exp(nu_a)
    
    T1 = A*xi_l*(((1/beta-1)/(1-tau_k)+delta)/(A*xi_k))**(xi_k/(xi_k-1))
    T2 = xi_l - 1 + xi_k*xi_l/(1-xi_k)
    nl = (W/T1)**(1/T2)

    k_ = (((1/beta-1)/(1-tau_k)+delta)/(A*xi_k))**(1/(xi_k-1)) * nl**(-xi_l/(xi_k-1))
    
    
    f = A*k_**(xi_k)*nl**(xi_l) + (1-delta) * k_
    
    pi = f - W*nl 
    
    
    x_ = (Uc*((1-tau_k)*(pi-k_) - (R_-1) *k_)  +(1-tau_l)*W*w_e*l*Uc - Uc*(c-T))/(1-1/beta)
    b_ = x_/Uc-k_
    
    labor_res = w_e*l-nl
    alpha_ = Alpha_*np.ones(c.shape)
    
    #logm,nu_a,nu_e,logc,logl,lognl,w_e,r,pi,f,b_,labor_res,res,x_,logk_    
    return np.vstack((
    logm,nu_a,nu_e,np.log(c),np.log(l),np.log(nl),np.log(k_),w_e,r,pi,f,b_,labor_res,x_,alpha_   
    ))
    
    

    
def GSS(YSS,y_i,weights):
    '''
    Aggregate conditions for the steady state
    '''
    logm,nu_a,nu_e,logc,logl,lognl,logk_,w_e,r,pi,f,b_,labor_res,x_,alpha_  = y_i
    logK,logAlpha_,R_,W,T  = YSS
  
    l,k_ = np.exp(logl),np.exp(logk_)
    K = np.exp(logK)      
    
    return np.hstack([
    weights.dot(K - k_), weights.dot(labor_res),R_-1/beta,T+Gov,
    weights.dot(f-np.exp(logc)-K-Gov)
     ])
    
    
    
def nomalize(Gamma,weights =None):
    '''
    Normalizes the distriubtion of states if need be
    '''
    if weights == None:            
        Gamma[:,0] -= np.mean(Gamma[:,0])
    else:
        Gamma[:,0] -= weights.dot(Gamma[:,0])
    return Gamma
    
def check_extreme(z_i):
    '''
    Checks for extreme positions in the state space
    '''
    return False
    extreme = False
    if z_i[0] < -3. or z_i[0] > 6.:
        extreme = True
    if z_i[1] > 8. or z_i[1] < -5.:
        extreme = True
    return extreme
    
def check_SS(YSS):
    if YSS[1] < -1.5:
        return False
    return True

def check(y_i):
    if y_i[11] < -5.:
        return False
    return False
        
    