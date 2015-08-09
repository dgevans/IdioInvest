# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 12:18:27 2014

@author: dgevans
"""
import numpy as np
import pycppad as ad

beta = 0.97
sigma = 1.5
gamma = 2.
#sigma_e = np.array([0.04,0.05,0.1,0.05])
sigma_e = np.array([0.0,0.03])
sigma_E = 0.0*np.eye(1)

delta = 0.04


xi_l = 0.66*(1-0.19) #*.75
xi_k = 0.34*(1-0.19) #*.75

eta = xi_k/(1-xi_l)

Gov = 0.32



n = 4 # number of measurability constraints
nG = 3 # number of aggregate measurability constraints.
ny = 19 # number of individual controls (m_{t},mu_{t},c_{t},l_{t},rho1_,rho2,phi,x_{t-1},kappa_{t-1}) Note that the forward looking terms are at the end
ne = 6 # number of Expectation Terms (E_t u_{c,t+1}, E_t u_{c,t+1}mu_{t+1} E_{t}x_{t-1 E_t rho_{1,t-1}} [This makes the control x_{t-1},rho_{t-1} indeed time t-1 measuable])
nY = 9 # Number of aggregates (alpha_1,alpha_2,tau,eta,lambda)
nz = 3 # Number of individual states (m_{t-1},mu_{t-1})
nv = 4 # number of forward looking terms (x_t,rho1_t)
n_p = 1 #number of parameters
nZ = 2 # number of aggregate states
nEps = 1
neps = len(sigma_e)

phat = np.array([-0.01])



def F(w):
    '''
    Individual first order conditions
    '''
    logm,muhat,nu_a,logc,logl,r,rho2_,rho3_,phi,res,foc_k,foc_Tau_k,foc_Tau_k_,foc_R,foc_R_,x_,logk_,rho1_,alpha_ = w[:ny] #y
    EUc,EUc_r,EUc_mu,Efoc_k_,Efoc_Tau_k,Efoc_R= w[ny:ny+ne] #e
    logXi,Tau_k,logAlpha_,logK_,R_,W,Tau_l,Psi,T = w[ny+ne:ny+ne+nY] #Y
    logm_,muhat_,nu_a_= w[ny+ne+nY:ny+ne+nY+nz] #z
    x,logk,rho1,alpha = w[ny+ne+nY+nz:ny+ne+nY+nz+nv] #v
    nu = w[ny+ne+nY+nz+nv] #p
    logXi_,Tau_k_ = w[ny+ne+nY+nz+nv+n_p:ny+ne+nY+nz+nv+n_p+nZ] #Z
    eps_p,eps_t= w[ny+ne+nY+nz+nv+n_p+nZ:ny+ne+nY+nz+nv+n_p+nZ+neps] #shock
    Eps = w[ny+ne+nY+nz+nv+n_p+nZ+neps] #aggregate shock
    eps_p = eps_p*(1-beta) 
            
    Alpha_ = np.exp(logAlpha_)
    Xi_,Xi = np.exp(logXi_),np.exp(logXi)  
    m_,m = np.exp(logm_),np.exp(logm)
    c,l,k_,k = np.exp(logc),np.exp(logl),np.exp(logk_),np.exp(logk)
    mu,mu_ = muhat*m,muhat_*m_    
    
    Uc = c**(-sigma)
    Ucc = -sigma*c**(-sigma-1)
    Ul = - l**gamma
    Ull = -gamma * l**(gamma-1)
    
    a = np.exp(nu_a + eps_t + Eps)
    nl = (a*xi_l/W)**(1./(1-xi_l)) * k_**eta
    f = a * k_**(xi_k) * nl**(xi_l)
    f_k = xi_k*f/k_
    f_n = xi_l*f/nl
    pi = (1-xi_l)*f
    pi_k = eta * pi/k_
    
    pi_kk = eta*(eta-1)*pi/k_**2
    n_k = eta*(a*xi_l/W)**(1./(1-xi_l)) * k_**(eta-1)
    
     
    
    ret = np.empty(ny+n,dtype=w.dtype)
    ret[0] = x_
    ret[1] = rho2_
    ret[2] = rho3_
    ret[3] = logk_
    
    ret[4] = x_*Uc/(beta*EUc) + Uc*((1-Tau_k_)*(pi/k_)-(R_-1+delta))*k_ - Uc*(c-T) - Ul*l - x #mu_
    ret[5] = m*Uc - alpha #rho1_
    ret[6] = (1-Tau_l)*W*Uc + Ul #phi
    ret[7] = rho1_ + rho2_ + rho3_ #logm
#    ret[8] = (Uc + (Ucc/Uc)*(x-x_*Uc/(beta*EUc) + Ul*l)*mu - Uc*mu + Ucc*x_/(beta*EUc)*(mu-mu_)  #c
#            +m*Ucc*rho1 + R_*m_*Ucc*rho2_ + m_*Ucc*((1-Tau_k_)*(pi_k-delta)+1)*rho3_
#            +(1-Tau_l)*W*Ucc*phi - Xi) 
            
    ret[8] = (Uc + Ucc*(((1-Tau_k_)*(pi/k_) - R_+1.-delta)*k_ - (c-T))*mu
                -Uc *mu + Ucc*x_/(beta*EUc)*(mu-mu_) + m*Ucc*rho1 + R_*m_*Ucc*rho2_
                +m_*Ucc*((1-Tau_k_)*(pi_k)+1-delta)*rho3_ + (1-Tau_l)*W*Ucc*phi - Xi)
    ret[9] = Ul - (Ull*l + Ul)*mu + Ull*phi + Psi #l
    ret[10] = foc_k - (Uc*((1-Tau_k_)*(pi_k)-(R_-1+delta))*mu + m_ * Uc*(1-Tau_k_)*pi_kk*rho3_
                -n_k*(Psi-f_n*Xi) + (f_k-delta+1)*Xi - Xi_/beta) #foc
    ret[11] = res - (f+(1-delta)*k_ - c - Gov -k)
    ret[12] = nu_a - nu*nu_a_ - eps_p
    ret[13] = foc_Tau_k - (-Uc*(pi)*mu - m_ * Uc*(pi_k)*rho3_)
    ret[14] = foc_Tau_k_ - Efoc_Tau_k
    ret[15] = foc_R - (m_*Uc*rho2_ - Uc*k_*mu)
    ret[16] = foc_R_ - Efoc_R
    ret[17] = r - (1+(1-Tau_k_)*(pi_k)-delta)
    ret[18] = alpha_ - Alpha_

    ret[19] = mu_ - EUc_mu/EUc #x_
    ret[20] = Alpha_ - beta*R_*m_*EUc#rho2_
    ret[21] = Alpha_ - beta*m_*EUc_r#rho3_
    ret[22] = Efoc_k_
    
    return ret
    
def G(w):
    '''
    Aggregate equations
    '''
    logm,muhat,nu_a,logc,logl,r,rho2_,rho3_,phi,res,foc_k,foc_Tau_k,foc_Tau_k_,foc_R,foc_R_,x_,logk_,rho1_,alpha_= w[:ny] #y
    logXi,Tau_k,logAlpha_,logK_,R_,W,Tau_l,Psi,T = w[ny+ne:ny+ne+nY] #Y
    logm_,muhat_,nu_a_= w[ny+ne+nY:ny+ne+nY+nz] #z
    logXi_,Tau_k_ = w[ny+ne+nY+nz+nv+n_p:ny+ne+nY+nz+nv+n_p+nZ] #Z
    eps_p,eps_t= w[ny+ne+nY+nz+nv+n_p+nZ:ny+ne+nY+nz+nv+n_p+nZ+neps] #shock
    Eps = w[ny+ne+nY+nz+nv+n_p+nZ+neps] #aggregate shock
    
    c,l,k_,m,m_ = np.exp(logc),np.exp(logl),np.exp(logk_),np.exp(logm),np.exp(logm_)
    K_,Xi = np.exp(logK_),np.exp(logXi)
    mu,mu_ = muhat*m,muhat_*m_    
    
    Uc = c**(-sigma)
    eps_p = eps_p*(1-beta) 
    a = np.exp(nu_a + eps_t + Eps)
    nl = (a*xi_l/W)**(1./(1-xi_l)) * k_**eta
    f = a * k_**(xi_k) * nl**(xi_l)
    nl_W = 1./(xi_l-1) * nl/W
    f_n = xi_l*f/nl
    pi_W = -xi_l *f /W
    pi_Wk = -xi_l*eta*f /(W*k_)
    
    
    ret = np.empty(nY+nG,dtype=w.dtype)
    ret[0] = muhat
    ret[1] = logAlpha_
    ret[2] = R_
    
    ret[3] = l-nl
    ret[4] = K_ - k_
    ret[5] = res
    ret[6] = Uc*(1-Tau_k_)*pi_W*mu + m_*Uc*(1-Tau_k_)*pi_Wk*rho3_ + (1-Tau_l)*Uc*phi - nl_W*(Psi-f_n*Xi)
    ret[7] = -W*Uc*phi
    ret[8] = logm
    
    ret[9] = T
    ret[10] = foc_Tau_k_
    ret[11] = foc_R_
    
    
    return ret
    
def f(y):
    '''
    Expectational equations that define e=Ef(y)
    '''
    logm,muhat,nu_a,logc,logl,r,rho2_,rho3_,phi,res,foc_k,foc_Tau_k,foc_Tau_k_,foc_R,foc_R_,x_,logk_,rho1_,alpha_= y
    
    c,k_ = np.exp(logc),np.exp(logk_)
    Uc = c**(-sigma)
    m = np.exp(logm)
    mu = m*muhat
            
    #EUc,EUc_r,Ea_,Ek_,Emu,Erho2_,Erho3_,Efoc_k
    ret = np.empty(ne,dtype=y.dtype)
    ret[0] = Uc
    ret[1] = Uc * r
    ret[2] = Uc*mu
    ret[3] = foc_k   
    ret[4] = foc_Tau_k
    ret[5] = foc_R
        
    
    return ret
    
def Finv(YSS,z):
    '''
    Given steady state YSS solves for y_i
    '''
    logm,muhat,nu_a = z
    logXi,Tau_k_,logAlpha,logK_,R_,W,Tau_l,Psi,T = YSS
    
    Xi = np.exp(logXi)
    Alpha_ = np.exp(logAlpha)
        
    
    m = np.exp(logm)
    mu = muhat*m
    Uc =Alpha_/m
    c = (Uc)**(-1/sigma)
    Ucc = -sigma * c**(-sigma-1)
    Ul = -(1-Tau_l)*W*Uc
    l = (-Ul)**(1./gamma)
    Ull = -gamma * l**(gamma-1)
    
    a = np.exp(nu_a)
    k_ = ( (W/(a*xi_l))**(xi_l/(1-xi_l))/(eta*(1-xi_l)*a)  *((1-beta+beta*delta)/(beta*(1-Tau_k_)) ) )**(1./(eta-1))
        
    nl = (a*xi_l/W)**(1./(1-xi_l)) * k_**eta
    f = a * k_**(xi_k) * nl**(xi_l)
    f_k = xi_k*f/k_
    f_n = xi_l*f/nl
    pi = (1-xi_l)*f
    pi_k = eta * pi/k_
    
    pi_kk = eta*(eta-1)*pi/k_**2
    n_k = eta*nl/k_
    
    
    x_ = (Uc*(c-T) + Ul*l - Uc*(1-Tau_k_)*(1-eta)*pi)/(1/beta-1)
    
    phi = -(Ul - (Ull*l+Ul)*mu + Psi)/Ull
    #(Uc + (Ucc/Uc)*(x-x_*Uc/beta*EUc + Ul*l)*mu - Uc*mu + Ucc*x_/(beta*EUc)*(mu-mu_)  #c
    #        +m*Ucc*rho1 + R_*m_*Ucc*rho2_ + m_*Ucc*((1-Tau_k_)*(pi_k-delta)-1)*rho3_
    #        +(1-Tau_l)*W*Ucc*phi - Xi) 
    rho3_ = (n_k*(Psi-f_n*Xi) + Xi/beta - (f_k+1-delta)*Xi)/(m*Uc*(1-Tau_k_)*pi_kk )
    
    rho2_ = -(Uc + Ucc/Uc *(x_-x_/beta + Ul*l)*mu - Uc*mu
            +(1/beta-1)*Ucc*m*rho3_ + (1-Tau_l)*W*Ucc*phi-Xi)/((1/beta-1)*Ucc*m)
    
    rho1_ = -(rho2_+rho3_)
    
    r = ((1-Tau_k_)*(pi_k) + 1-delta)
    
    
    res = (f+(1-delta)*k_ - c - Gov -k_)
    
    foc_k = (m * Uc*(1-Tau_k_)*pi_kk*rho3_
                -n_k*(Psi-f_n*Xi) + (f_k-delta+1)*Xi - Xi/beta) 
    
                
    foc_Tau_k = (-Uc*(pi)*mu - m * Uc*(pi_k)*rho3_)
    
    foc_R = (m*Uc*rho2_ - Uc*k_*mu)
    
    alpha_ = np.ones(c.shape)*Alpha_
    #logm,muhat,nu_a,logc,logl,r,rho2_,rho3_,phi,res,foc_k,foc_Tau_k,foc_Tau_k_,foc_R,foc_R_,x_,logk_,rho1_
    return np.vstack((
    logm,muhat,nu_a,np.log(c),np.log(l),r,rho2_,rho3_,phi,res,foc_k,foc_Tau_k,foc_Tau_k,foc_R,foc_R,x_,np.log(k_),rho1_,alpha_
    ))
    
    

    
def GSS(YSS,y_i,weights):
    '''
    Aggregate conditions for the steady state
    '''
    logm,muhat,nu_a,logc,logl,r,rho2_,rho3_,phi,res,foc_k,foc_Tau_k,foc_Tau_k_,foc_R,foc_R_,x_,logk_,rho1_,alpha_   = y_i
    logXi,Tau_k_,logAlpha,logK_,R_,W,Tau_l,Psi,T  = YSS
    
    c,l,k_,m,a = np.exp(logc),np.exp(logl),np.exp(logk_),np.exp(logm),np.exp(nu_a)
 
    mu = muhat*m
    K_,Xi = np.exp(logK_), np.exp(logXi)

    nl = (a*xi_l/W)**(1./(1-xi_l)) * k_**eta
    f = a * k_**(xi_k) * nl**(xi_l)
    nl_W = 1./(xi_l-1) * nl/W
    f_n = xi_l*f/nl
    pi_W = -xi_l *f /W
    pi_Wk = -xi_l*eta *f /(W*k_)
    
    #mu = muhat*m
    Uc = c**(-sigma)    
    
    return np.hstack(
    [
    weights.dot(l-nl),
    weights.dot(K_ - k_),
    weights.dot(res),
    weights.dot(Uc*(1-Tau_k_)*pi_W*mu + m*Uc*(1-Tau_k_)*pi_Wk*rho3_ + (1-Tau_l)*Uc*phi - nl_W*(Psi-f_n*Xi)),
    weights.dot(-W*Uc*phi),
    R_-1/beta,
    T,
    weights.dot(foc_Tau_k_),
    weights.dot(foc_R_)
    ]
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
    
def check_extreme(z_i,Gamma):
    '''
    Checks for extreme positions in the state space
    '''
    std1 = np.std(Gamma[:,0])
    std2 = np.std(Gamma[:,1])
    extreme = False
    if z_i[0] < -4*std1 or z_i[0] > 4*std1:
        extreme = True
    if z_i[1] > 4*std2 or z_i[1] < -4 *std2:
        extreme = True
    return extreme
    
def check_SS(YSS):
    if YSS[2] < -1.5:
        return False
    return True
    