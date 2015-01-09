# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 12:18:27 2014

@author: dgevans
"""
import numpy as np
import pycppad as ad

beta = 0.96
sigma = 1.5
gamma = 2.
#sigma_e = np.array([0.04,0.05,0.1,0.05])
sigma_e = np.array([0.0,0.03])
sigma_E = 0.0*np.eye(1)
mu_a = 0.
Delta = 0.06
xi_l = 0.66*(1-0.21) #*.75
xi_k = 0.34*(1-0.21) #*.75

Gov = 0.17



n = 4 # number of measurability constraints
nG = 4 # number of aggregate measurability constraints.
ny = 30 # number of individual controls (m_{t},mu_{t},c_{t},l_{t},rho1_,rho2,phi,x_{t-1},kappa_{t-1}) Note that the forward looking terms are at the end
ne = 14 # number of Expectation Terms (E_t u_{c,t+1}, E_t u_{c,t+1}mu_{t+1} E_{t}x_{t-1 E_t rho_{1,t-1}} [This makes the control x_{t-1},rho_{t-1} indeed time t-1 measuable])
nY = 15 # Number of aggregates (alpha_1,alpha_2,tau,eta,lambda)
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
    logm,muhat,nu_a,logc,lognl,logl,phi,phi2,foc_k,r,rho2_,rho3_,res,foc_R_,foc_tau_k,Efoc_tau_k,A,xi,dk_dtau_k_, tR_khat,tW_khat,tR_k,tW_k, dnl_dtau, tR_l, tE_l,x_,logk_,rho1_,alpha_ = w[:ny] #y
    EUc,EUc_r,Ex_,Ek_,EUc_mu,Erho2_,Erho3_,Efoc_k,hatEfoc_tau_k,EXi,EUcfk,EUcfkk,EtR_k,EtW_k= w[ny:ny+ne] #e
    logXi,tau_k,logAlpha_,logK_,R_,W,Eta,tau_l,T,dK_dtau_k_,dUc_dtau_l,TR_k,TW_k,TR_l,TE_l = w[ny+ne:ny+ne+nY] #Y
    logm_,muhat_,nu_a_= w[ny+ne+nY:ny+ne+nY+nz] #z
    x,logk,rho1,alpha = w[ny+ne+nY+nz:ny+ne+nY+nz+nv] #v
    nu = w[ny+ne+nY+nz+nv] #v
    logXi_,tau_k_ = w[ny+ne+nY+nz+nv+n_p:ny+ne+nY+nz+nv+n_p+nZ] #Z
    eps_p,eps_t= w[ny+ne+nY+nz+nv+n_p+nZ:ny+ne+nY+nz+nv+n_p+nZ+neps] #shock
    Eps = w[ny+ne+nY+nz+nv+n_p+nZ+neps] #aggregate shock
    
    eps_p = eps_p*(1-beta)        
    Alpha_ = np.exp(logAlpha_)
    Xi_,Xi = np.exp(logXi_),np.exp(logXi)  
    m_,m = np.exp(logm_),np.exp(logm)
    c,l,k_,k,nl = np.exp(logc),np.exp(logl),np.exp(logk_),np.exp(logk),np.exp(lognl)
    mu,mu_ = muhat*m,muhat_*m_    
    
    Uc = c**(-sigma)
    Ucc = -sigma*c**(-sigma-1)
    Un = -nl**(gamma)
    Unn = -gamma*nl**(gamma-1)
    delta_i = Delta   
    
    fkk = A*xi_k*(xi_k-1) * k_**(xi_k-2) * l**xi_l
    fll = A*xi_l*(xi_l-1) * k_**(xi_k) * l**(xi_l-2)
    fl = A*xi_l * k_**(xi_k) * l**(xi_l-1)
    fk = A*xi_k * k_**(xi_k-1) * l**xi_l
    flk = A*xi_k * xi_l * k_**(xi_k-1) * l**(xi_l-1)
    f =  A*k_**(xi_k) * l**xi_l
    
     
    
    ret = np.empty(ny+n,dtype=w.dtype)
    ret[0] = x_ - Ex_ # x is independent of eps
    ret[1] = k_ - Ek_
    ret[2] = rho2_ - Erho2_
    ret[3] = rho3_ - Erho3_
    
    
    
    ret[4] = x_*Uc/(beta*EUc) + Uc*((1-tau_k_)*((1-xi_l)*f-delta_i*k_) - R_ *k_) + Uc*W*(1-tau_l)*nl - Uc*(c-T) - x #x
    ret[5] = alpha - m*Uc #rho1
    ret[6] = r -  ( (1-tau_k_)*(fk-delta_i) + 1 )#r
    ret[7] = W - fl#phi2
    ret[8] = nu_a - (nu*nu_a_+eps_p + (1-nu)*mu_a ) #a
    
    #ret[9] = (Uc + Ucc*mu*(x - Uc/Ucc) - x_*Ucc*mu_/(beta*EUc) 
    #            - rho1*m*Ucc - rho2_*m_*R_*Ucc - rho3_*m_*r*Ucc - Xi )
    ret[9] = (Uc + x_*Ucc/(beta*EUc) *(mu-mu_) +Ucc*mu*((1-tau_k_)*((1-xi_l)*f-delta_i*k_) - R_ *k_ 
                +W*(1-tau_l)*nl-c+T-Uc/Ucc) - rho1*m*Ucc - rho2_*m_*(1+R_)*Ucc - rho3_*m_*r*Ucc - Xi  + Ucc*(1-tau_l)*phi2/Uc )
                
    ret[10] = (Uc*mu*(1-tau_k_)*(1-xi_l)*fl  - rho3_*m_*(1-tau_k_)*flk*Uc  - phi*fll - Eta + fl*Xi)
    ret[11] = foc_k - ( mu*Uc*( (1-tau_k_)*((1-xi_l)*fk-delta_i) -R_) - rho3_*m_*(1-tau_k_)*fkk*Uc
                        -phi*flk - Xi_/beta + (fk + 1 - delta_i)*Xi ) #fock
    ret[12] = rho1_ + rho2_ + rho3_
    ret[13] = res - (f +(1-delta_i)*k_ - c - Gov - k)
    ret[14] = foc_R_ - (k_*EUc_mu + rho2_*m_*EUc)
    ret[15] = (1-tau_l) + Un/(W*Uc)
    ret[16] = Un + Unn/(W*Uc)*phi2 + mu*Uc*W*(1-tau_l) + Eta
    ret[17] = alpha_ - Alpha_
    ret[18] = foc_tau_k - (m_*rho3_*(fk-delta_i)*Uc - mu*Uc*((1-xi_l)*f-delta_i*k_) )
    ret[19] = Efoc_tau_k - hatEfoc_tau_k
    ret[20] = xi - Xi
    ret[21] = dk_dtau_k_ - ( EUcfk/((1-tau_k_)*EUcfkk) )
    ret[22] = tR_khat - mu*Uc*((1-xi_l)*f - delta_i*k_ - dk_dtau_k_ *( (1-tau_k_)*((1-xi_l)*fk-delta_i) - R_  ) )*(1-tau_k_)/(R_*dK_dtau_k_*EXi)
    ret[23] = tW_khat - (dk_dtau_k_ * flk * phi) *(1-tau_k_)/(R_*dK_dtau_k_*EXi)
    ret[24] = tR_k - EtR_k
    ret[25] = tW_k - EtW_k
    ret[26] = dnl_dtau - ( -Un/((1-tau_l)*Unn) )
    ret[27] = tR_l  + ((1-tau_l)*W*mu*Uc*dnl_dtau -W*nl*Uc*mu)/dUc_dtau_l#+ (mu*Uc*w_e*W*(dl_dtau*(1-tau_l) - l)/dUc_dtau_l)
    ret[28] = tE_l - ((dnl_dtau*W*(Uc-Xi))/dUc_dtau_l)
    ret[29] = A - np.exp((1-Eps)*nu_a + eps_t)
    
    # xi,dk_dtau_k_, tR_khat,tW_khat, dnl_tau, tR_l, tE_l
    
    ret[30] = Efoc_k #k_
    ret[31] = mu_  - EUc_mu/EUc 
    ret[32] = Alpha_ - beta*(1+R_)*m_*EUc #rho2_
    ret[33] = Alpha_ - beta*m_*EUc_r #rho3
    
    return ret
    
def G(w):
    '''
    Aggregate equations
    '''
    logm,muhat,nu_a,logc,lognl,logl,phi,phi2,foc_k,r,rho2_,rho3_,res,foc_R_,foc_tau_k,Efoc_tau_k,A,xi,dk_dtau_k_, tR_khat,tW_khat,tR_k,tW_k, dnl_dtau, tR_l, tE_l,x_,logk_,rho1_,alpha_ = w[:ny] #y
    logXi,tau_k,logAlpha_,logK_,R_,W,Eta,tau_l,T,dK_dtau_k_,dUc_dtau_l,TR_k,TW_k,TR_l,TE_l = w[ny+ne:ny+ne+nY] #Y
    logm_,muhat_,nu_a_= w[ny+ne+nY:ny+ne+nY+nz] #z
    logXi_,tau_k_ = w[ny+ne+nY+nz+nv+n_p:ny+ne+nY+nz+nv+n_p+nZ] #Z
    eps_p,eps_t= w[ny+ne+nY+nz+nv+n_p+nZ:ny+ne+nY+nz+nv+n_p+nZ+neps] #shock
    Eps = w[ny+ne+nY+nz+nv+n_p+nZ+neps] #aggregate shock
    
    c,l,k_,m,m_,nl = np.exp(logc),np.exp(logl),np.exp(logk_),np.exp(logm),np.exp(logm_),np.exp(lognl)
    K_ = np.exp(logK_)
    Uc = c**(-sigma)
    mu,mu_ = muhat*m,muhat_*m_    
    
    
    ret = np.empty(nY+nG,dtype=w.dtype)
    ret[0] = R_
    ret[1] = logK_
    ret[2] = logAlpha_
    ret[3] = muhat
    
    ret[4] = res
    ret[5] = nl-l
    ret[6] = logm-0.
    ret[7] = mu*Uc*(1-tau_l)*nl + phi + phi2*(1-tau_l)/W
    ret[8] = -mu*Uc*nl*W  - phi2 
    ret[9] = dK_dtau_k_ - dk_dtau_k_
    ret[10] = dUc_dtau_l - dnl_dtau*W*Uc
    ret[11] = TR_k - tR_k
    ret[12] = TW_k - tW_k
    ret[13] = TR_l - tR_l
    ret[14] = TE_l - tE_l
    
    ret[15] = K_ - k_
    ret[16] = foc_R_
    ret[17] = T - Gov
    ret[18] = Efoc_tau_k
    
    return ret
    
def f(y):
    '''
    Expectational equations that define e=Ef(y)
    '''
    logm,muhat,nu_a,logc,lognl,logl,phi,phi2,foc_k,r,rho2_,rho3_,res,foc_R_,foc_tau_k,Efoc_tau_k,A,xi,dk_dtau_k_, tR_khat,tW_khat,tR_k,tW_k, dnl_dtau, tR_l, tE_l,x_,logk_,rho1_,alpha_ = y
    
    c,k_,l = np.exp(logc),np.exp(logk_),np.exp(logl)
    Uc = c**(-sigma)
    m = np.exp(logm)
    mu = m*muhat
    
    fkk = A*xi_k*(xi_k-1) * k_**(xi_k-2) * l**xi_l
    fk = A*xi_k * k_**(xi_k-1) * l**xi_l
    delta_i = Delta
            
    #EUc,EUc_r,Ea_,Ek_,Emu,Erho2_,Erho3_,Efoc_k,Efoc_tau_k,EXi,EUcfk,EUcfkk,EtR_k,EtW_k
    ret = np.empty(ne,dtype=y.dtype)
    ret[0] = Uc
    ret[1] = Uc * r
    ret[2] = x_
    ret[3] = k_
    ret[4] = Uc*mu
    ret[5] = rho2_
    ret[6] = rho3_
    ret[7] = foc_k
    ret[8] = foc_tau_k
    ret[9] = xi
    ret[10] = Uc*(fk-delta_i)
    ret[11] = Uc*fkk
    ret[12] = tR_khat
    ret[13] = tW_khat
        
    
    return ret
    
def Finv(YSS,z):
    '''
    Given steady state YSS solves for y_i
    '''
    logm,muhat,nu_a = z
    logXi,tau_k,logAlpha,logK_,R_,W,Eta,tau_l,T,dK_dtau_k_,dUc_dtau_l,TR_k,TW_k,TR_l,TE_l = YSS
    
    Xi = np.exp(logXi)
    Alpha_ = np.exp(logAlpha)
        
    
    m = np.exp(logm)
    mu = muhat*m
    Uc =Alpha_/m
    c = (Uc)**(-1/sigma)
    Ucc = -sigma * c**(-sigma-1)
    
    Un = -(1-tau_l)*W*Uc
    nl = (-Un)**(1/gamma)
    Unn = -gamma*nl**(gamma-1)
    
    r = 1/beta * np.ones(c.shape)
    
    delta_i = Delta
    A = np.exp(nu_a)
    
    #T1 = xi_l*((1/beta-1+delta_i)/(xi_k*(1-tau)))**(xi_k/(xi_k-1))
    #T2 = xi_l - 1 + xi_k*xi_l/(1-xi_k)
    #l = (W/T1)**(1/T2)

    #k_ = ((1/beta-1+delta_i)/(xi_k*(1-tau)))**(1./(xi_k-1)) * l**(xi_l/(1-xi_k))
    
    T1 = ((1/beta-1)/(1-tau_k) + delta_i)/ ( A*xi_k *(W/(A*xi_l))**((xi_k-1)/xi_k))
    T2 = (xi_k+xi_l - 1.)/xi_k
    l = T1**(1/T2)
    k_ = (W/(A*xi_l))**(1/xi_k) * l**((1-xi_l)/xi_k)  
    
    
    fkk = A*xi_k*(xi_k-1) * k_**(xi_k-2) * l**xi_l
    fll = A*xi_l*(xi_l-1) * k_**(xi_k) * l**(xi_l-2)
    fl = A*xi_l * k_**(xi_k) * l**(xi_l-1)
    fk = A*xi_k * k_**(xi_k-1) * l**xi_l
    flk = A*xi_k * xi_l * k_**(xi_k-1) * l**(xi_l-1)
    f =  A*k_**(xi_k) * l**xi_l
    
    
    x_ = -( Uc*((1-tau_k)*((1-xi_l)*f-delta_i*k_) - R_ *k_) +Uc*(1-tau_l)*W*nl - Uc*(c-T) )/(1/beta-1)
        
    phi2 = -(Un + mu*Uc*W*(1-tau_l) + Eta)*W*Uc/Unn
    
    rho3 = (Uc*mu*((1-tau_k)*((1-xi_l)*(fk-fl*flk/fll)-delta_i) + 1 - 1/beta) + Eta*flk/fll - Xi/beta + ((fk-fl*flk/fll)+1-delta_i)*Xi  ) / (m*(1-tau_k)*Uc*(fkk-flk*flk/fll))
    
    phi = (Uc*mu*(1-tau_k)*(1-xi_l)*fl - rho3*m*(1-tau_k)*flk*Uc - Eta + fl *Xi)/fll
    
    rho1 = (Uc +Ucc*mu*((1-tau_k)*((1-xi_l)*f-delta_i*k_) -R_*k_ 
                +W*(1-tau_l)*nl-c+T-Uc/Ucc) - Xi + Ucc*(1-tau_l)*phi2/Uc)/(m*Ucc*(1-1/beta))
    
    rho2 = - rho1 - rho3
    
    foc_R = k_*Uc*mu + rho2*m*Uc
    
    
    foc_k = ( mu*Uc*( (1-tau_k)*((1-xi_l)*fk-delta_i) -R_) - rho3*m*(1-tau_k)*fkk*Uc
                        -phi*flk - Xi/beta + (fk + 1 - delta_i)*Xi )
                
    res = (f + (1-delta_i)*k_ - c - Gov - k_)
    alpha = Alpha_*np.ones(c.shape)
    foc_tau_k = (m*rho3*(fk-delta_i)*Uc - mu*Uc*((1-xi_l)*f-delta_i*k_) )
    
    xi = Xi*np.ones(c.shape)
    dk_dtau_k_ = ( (fk-delta_i)/((1-tau_k)*fkk) )
    tR_k = mu*Uc*((1-xi_l)*f - delta_i*k_ - dk_dtau_k_ *( (1-tau_k)*((1-xi_l)*fk-delta_i) - R_  ) )*(1-tau_k)/((R_)*dK_dtau_k_*Xi)
    tW_k = (dk_dtau_k_ * flk * phi) *(1-tau_k)/((R_)*dK_dtau_k_*Xi)
    dnl_dtau = ( -Un/((1-tau_l)*Unn) )
    tR_l  = -((1-tau_l)*W*mu*Uc*dnl_dtau -W*nl*Uc*mu)/dUc_dtau_l#+ (mu*Uc*w_e*W*(dl_dtau*(1-tau_l) - l)/dUc_dtau_l)
    tE_l = ((dnl_dtau*W*(Uc-Xi))/dUc_dtau_l)
    #A,xi,dk_dtau_k_, tR_khat,tW_khat,tR_k,tW_k, dnl_dtau, tR_l, tE_l
    return np.vstack((
    logm,muhat,nu_a,np.log(c),np.log(nl),np.log(l),phi,phi2,foc_k,r,rho2,rho3,res,foc_R,foc_tau_k,foc_tau_k,A,xi,dk_dtau_k_,tR_k,tW_k,tR_k,tW_k, dnl_dtau, tR_l, tE_l,x_,np.log(k_),rho1,alpha
    ))
    
    

    
def GSS(YSS,y_i,weights):
    '''
    Aggregate conditions for the steady state
    '''
    logm,muhat,nu_a,logc,lognl,logl,phi,phi2,foc_k,r,rho2_,rho3_,res,foc_R_,foc_tau_k,Efoc_tau_k,A,xi,dk_dtau_k_, tR_khat,tW_khat,tR_k,tW_k, dnl_dtau, tR_l, tE_l,x_,logk_,rho1_,alpha_   = y_i
    logXi,tau_k,logAlpha,logK_,R_,W,Eta,tau_l,T,dK_dtau_k_,dUc_dtau_l,TR_k,TW_k,TR_l,TE_l = YSS
    
    c,l,k_,m,nl = np.exp(logc),np.exp(logl),np.exp(logk_),np.exp(logm),np.exp(lognl)   
    mu = muhat*m
    
    #mu = muhat*m
    K = np.exp(logK_)
    Uc = c**(-sigma)    
    
    return np.hstack(
    [weights.dot(K - k_), weights.dot(res), weights.dot(nl - l), weights.dot( Efoc_tau_k), R_-(1/beta-1),
     weights.dot(mu*Uc*(1-tau_l)*nl + phi + phi2*(1-tau_l)/W), weights.dot(foc_R_), weights.dot(-mu*Uc*nl*W  - phi2),T-Gov,
     weights.dot(dK_dtau_k_ - dk_dtau_k_),weights.dot(dUc_dtau_l - dnl_dtau*W*Uc),weights.dot(TR_k - tR_k),
    weights.dot(TW_k - tW_k),weights.dot( TR_l - tR_l),weights.dot(TE_l - tE_l)
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
    if z_i[0] < -3*std1 or z_i[0] > 3*std1:
        extreme = True
    if z_i[1] > 3*std2 or z_i[1] < -3 *std2:
        extreme = True
    return extreme
    
def check_SS(YSS):
    if YSS[3] < -1.5:
        return False
    return True
    