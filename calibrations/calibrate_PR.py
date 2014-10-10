# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 12:18:27 2014

@author: dgevans
"""
import numpy as np
import pycppad as ad

beta = 0.99
sigma = 1.
#sigma_e = np.array([0.04,0.05,0.1,0.05])
sigma_e = np.array([0.02,0.0])
sigma_E = 0.0
mu_a = 0.
Delta = 0.0
xi_l = 0.66*(1-0.2) #*.75
xi_k = 0.34*(1-0.2) #*.75

Gov = 0.



n = 4 # number of measurability constraints
nG = 2 # number of aggregate measurability constraints.
ny = 15 # number of individual controls (m_{t},mu_{t},c_{t},l_{t},rho1_,rho2,phi,x_{t-1},kappa_{t-1}) Note that the forward looking terms are at the end
ne = 8 # number of Expectation Terms (E_t u_{c,t+1}, E_t u_{c,t+1}mu_{t+1} E_{t}x_{t-1 E_t rho_{1,t-1}} [This makes the control x_{t-1},rho_{t-1} indeed time t-1 measuable])
nY = 7 # Number of aggregates (alpha_1,alpha_2,tau,eta,lambda)
nz = 3 # Number of individual states (m_{t-1},mu_{t-1})
nv = 3 # number of forward looking terms (x_t,rho1_t)
n_p = 1 #number of parameters
nZ = 2 # number of aggregate states

neps = len(sigma_e)

phat = np.array([-0.01])



def F(w):
    '''
    Individual first order conditions
    '''
    logm,muhat,nu_a,logc,logl,phi,foc_k,r,rho2_,rho3_,res,foc_R_,x_,logk_,rho1_ = w[:ny] #y
    EUc,EUc_r,Ex_,Ek_,EUc_mu,Erho2_,Erho3_,Efoc_k= w[ny:ny+ne] #e
    logXi,logAlpha,logK_,R_,W,tau,Eta = w[ny+ne:ny+ne+nY] #Y
    logm_,muhat_,nu_a_= w[ny+ne+nY:ny+ne+nY+nz] #z
    x,logk,rho1 = w[ny+ne+nY+nz:ny+ne+nY+nz+nv] #v
    nu = w[ny+ne+nY+nz+nv] #v
    logXi_,logAlpha_ = w[ny+ne+nY+nz+nv+n_p:ny+ne+nY+nz+nv+n_p+nZ] #Z
    eps_p,eps_t= w[ny+ne+nY+nz+nv+n_p+nZ:ny+ne+nY+nz+nv+n_p+nZ+neps] #shock
    Eps = w[ny+ne+nY+nz+nv+n_p+nZ+neps] #aggregate shock
    
            
    Alpha_,Alpha = np.exp(logAlpha_),np.exp(logAlpha)
    Xi_,Xi = np.exp(logXi_),np.exp(logXi)  
    m_,m = np.exp(logm_),np.exp(logm)
    c,l,k_,k = np.exp(logc),np.exp(logl),np.exp(logk_),np.exp(logk)
    mu,mu_ = muhat*m,muhat_*m_    
    
    Uc = c**(-sigma)
    Ucc = -sigma*c**(-sigma-1)
    delta_i = nu_a_+eps_p+eps_t + Delta   
    
    fkk = xi_k*(xi_k-1) * k_**(xi_k-2) * l**xi_l
    fll = xi_l*(xi_l-1) * k_**(xi_k) * l**(xi_l-2)
    fl = xi_l * k_**(xi_k) * l**(xi_l-1)
    fk = xi_k * k_**(xi_k-1) * l**xi_l
    flk = xi_k * xi_l * k_**(xi_k-1) * l**(xi_l-1)
    f =  k_**(xi_k) * l**xi_l
     
    
    ret = np.empty(ny+n,dtype=w.dtype)
    ret[0] = x_ - Ex_ # x is independent of eps
    ret[1] = k_ - Ek_
    ret[2] = rho2_ - Erho2_
    ret[3] = rho3_ - Erho3_
    
    
    
    ret[4] = x_*Uc/(beta*EUc) + Uc*((1-tau)*(1-xi_l)*f + (1-delta_i)*k_ - R_ *k_) + Uc*W - Uc*c - x #x
    ret[5] = Alpha - m*Uc #rho1
    ret[6] = r -  ( (1-tau)*fk + 1 - delta_i )#r
    ret[7] = W - fl#phi2
    ret[8] = nu_a - (nu*(nu_a_+eps_p) + (1-nu)*mu_a ) #a
    
    #ret[9] = (Uc + Ucc*mu*(x - Uc/Ucc) - x_*Ucc*mu_/(beta*EUc) 
    #            - rho1*m*Ucc - rho2_*m_*R_*Ucc - rho3_*m_*r*Ucc - Xi )
    ret[9] = (Uc + x_*Ucc/(beta*EUc) *(mu-mu_) +Ucc*mu*((1-tau)*(1-xi_l)*f + (1-delta_i)*k_ - R_ *k_ 
                +W-c-Uc/Ucc) - rho1*m*Ucc - rho2_*m_*R_*Ucc - rho3_*m_*r*Ucc - Xi )
                
    ret[10] = (Uc*mu*(1-tau)*(1-xi_l)*fl  - rho3_*m_*(1-tau)*flk*Uc  - phi*fll - Eta + fl*Xi)
    ret[11] = foc_k - ( mu*Uc*( (1-tau)*(1-xi_l)*fk + (1-delta_i) - R_) - rho3_*m_*(1-tau)*fkk*Uc
                        -phi*flk - Xi_/beta + (fk + 1 - Delta)*Xi ) #fock
    ret[12] = rho1_ + rho2_ + rho3_
    ret[13] = res - (f +(1-Delta)*k_ - c - Gov - k)
    ret[14] = foc_R_ - (k_*EUc_mu - rho2_*m_*EUc)
    
    
    
    ret[15] = Efoc_k #k_
    ret[16] = mu_  - EUc_mu/EUc 
    ret[17] = Alpha_ - beta*R_*m_*EUc #rho2_
    ret[18] = Alpha_ - beta*m_*EUc_r #rho3
    
    return ret
    
def G(w):
    '''
    Aggregate equations
    '''
    logm,muhat,nu_a,logc,logl,phi,foc_k,r,rho2_,rho3_,res,foc_R_,x_,logk_,rho1_ = w[:ny] #y
    logXi,logAlpha,logK_,R_,W,tau,Eta = w[ny+ne:ny+ne+nY] #Y
    logm_,muhat_,nu_a_= w[ny+ne+nY:ny+ne+nY+nz] #z
    logXi,logAlpha = w[ny+ne+nY+nz+nv+n_p:ny+ne+nY+nz+nv+n_p+nZ] #Z
    Eps = w[ny+ne+nY+nz+nv+n_p+nZ+neps] #aggregate shock
    
    c,l,k_,m,m_ = np.exp(logc),np.exp(logl),np.exp(logk_),np.exp(logm),np.exp(logm_)
    K_ = np.exp(logK_)
    Uc = c**(-sigma)
    mu,mu_ = muhat*m,muhat_*m_    
    
    fk = xi_k * k_**(xi_k-1) * l**xi_l
    f =  k_**(xi_k) * l**xi_l
    
    ret = np.empty(nY+nG,dtype=w.dtype)
    ret[0] = R_
    ret[1] = logK_
    
    ret[2] = res
    ret[3] = 1-l
    ret[4] = m_*rho3_*fk*Uc - mu*Uc*(1-xi_l)*f
    ret[5] = logm-0.
    ret[6] = mu*Uc + phi
    
    ret[7] = K_ - k_
    ret[8] = foc_R_
    
    return ret
    
def f(y):
    '''
    Expectational equations that define e=Ef(y)
    '''
    logm,muhat,nu_a,logc,logl,phi,foc_k,r,rho2_,rho3_,res,foc_R_,x_,logk_,rho1_ = y
    
    c,k_ = np.exp(logc),np.exp(logk_)
    Uc = c**(-sigma)
    m = np.exp(logm)
    mu = m*muhat
            
    #EUc,EUc_r,Ea_,Ek_,Emu,Erho2_,Erho3_,Efoc_k
    ret = np.empty(ne,dtype=y.dtype)
    ret[0] = Uc
    ret[1] = Uc * r
    ret[2] = x_
    ret[3] = k_
    ret[4] = Uc*mu
    ret[5] = rho2_
    ret[6] = rho3_
    ret[7] = foc_k   
        
    
    return ret
    
def Finv(YSS,z):
    '''
    Given steady state YSS solves for y_i
    '''
    logm,muhat,nu_a = z
    logXi,logAlpha,logK_,R_,W,tau,Eta = YSS
    
    Xi = np.exp(logXi)
    Alpha_ = np.exp(logAlpha)
        
    
    m = np.exp(logm)
    mu = muhat*m
    Uc =Alpha_/m
    c = (Uc)**(-1/sigma)
    Ucc = -sigma * c**(-sigma-1)
    
    r = 1/beta * np.ones(c.shape)
    
    delta_i = Delta + nu_a
    
    T1 = xi_l*((1/beta-1+delta_i)/(xi_k*(1-tau)))**(xi_k/(xi_k-1))
    T2 = xi_l - 1 + xi_k*xi_l/(1-xi_k)
    l = (W/T1)**(1/T2)

    k_ = ((1/beta-1+delta_i)/(xi_k*(1-tau)))**(1./(xi_k-1)) * l**(xi_l/(1-xi_k))

    
    
    fkk = xi_k*(xi_k-1) * k_**(xi_k-2) * l**xi_l
    fll = xi_l*(xi_l-1) * k_**(xi_k) * l**(xi_l-2)
    fl = xi_l * k_**(xi_k) * l**(xi_l-1)
    fk = xi_k * k_**(xi_k-1) * l**xi_l
    flk = xi_k * xi_l * k_**(xi_k-1) * l**(xi_l-1)
    f =  k_**(xi_k) * l**xi_l
    
    
    x_ = -( Uc*((1-tau)*(1-xi_l)*f + (1-delta_i)*k_- 1/beta*k_) +Uc*W - Uc*c )/(1/beta-1)
    
    
    
    rho3 = (Uc*mu*((1-tau)*(1-xi_l)*(fk-fl*flk/fll) + 1 - delta_i - 1/beta) + Eta*flk/fll - Xi/beta + ((fk-fl*flk/fll)+1-Delta)*Xi  ) / (m*(1-tau)*Uc*(fkk-flk*flk/fll))
    phi = (Uc*mu*(1-tau)*(1-xi_l)*fl - rho3*m*(1-tau)*flk*Uc - Eta + fl *Xi)/fll
    
    rho1 = (Uc +Ucc*mu*((1-tau)*(1-xi_l)*f + (1-delta_i)*k_ - R_ *k_ 
                +W-c-Uc/Ucc)   - Xi)/(m*Ucc*(1-1/beta))
    
    rho2 = - rho1 - rho3
    
    foc_R = k_*Uc*mu - rho2*m*Uc
    
    
    foc_k = ( mu*Uc*( (1-tau)*(1-xi_l)*fk + (1-delta_i) - R_) - rho3*m*(1-tau)*fkk*Uc
                        -phi*flk - Xi/beta + (fk + 1 - Delta)*Xi )
                
    res = (f + (1-Delta)*k_ - c - Gov - k_)
    
    return np.vstack((
    logm,muhat,nu_a,np.log(c),np.log(l),phi,foc_k,r,rho2,rho3,res,foc_R,x_,np.log(k_),rho1
    ))
    
    

    
def GSS(YSS,y_i,weights):
    '''
    Aggregate conditions for the steady state
    '''
    logm,muhat,nu_a,logc,logl,phi,foc_k,r,rho2_,rho3_,res,foc_R_,x_,logk_,rho1_   = y_i
    logXi,logAlpha,logK_,R_,W,tau,Eta = YSS
    
    c,l,k_,m = np.exp(logc),np.exp(logl),np.exp(logk_),np.exp(logm)
    fk = xi_k * k_**(xi_k-1) * l**xi_l
    f =  k_**(xi_k) * l**xi_l    
    mu = muhat*m
    
    #mu = muhat*m
    K = np.exp(logK_)
    Uc = c**(-sigma)    
    
    return np.hstack(
    [weights.dot(K - k_), weights.dot(res), weights.dot(1 - l), weights.dot( m*rho3_*fk*Uc - mu*Uc*(1-xi_l)*f), R_-1/beta,
     weights.dot(mu*Uc + phi), weights.dot(foc_R_) ]
    )
    
    
    
def nomalize(Gamma,weights =None):
    '''
    Normalizes the distriubtion of states if need be
    '''
    if weights == None:            
        Gamma[:,0] -= np.mean(Gamma[:,0])
        Gamma[:,1] -= np.mean(Gamma[:,1])
    else:
        Gamma[:,0] -= weights.dot(Gamma[:,0])
        Gamma[:,1] -= weights.dot(Gamma[:,1])
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
    if YSS[2] < -1.5:
        return False
    return True
    