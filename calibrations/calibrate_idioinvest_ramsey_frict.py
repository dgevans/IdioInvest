# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 12:18:27 2014

@author: dgevans
"""
import numpy as np
import pycppad as ad

beta = 0.95
gamma = 2.
sigma = 0.5
#sigma_e = np.array([0.04,0.05,0.1,0.05])
sigma_e = np.array([0.02,0.0,0.,0.])
sigma_E = 0.03
mu_e = 0.
mu_a = 0.
chi = 1.
delta = 0.06
xi_l = 0.66*(1-0.21) #*.75
xi_k = 0.34*(1-0.21) #*.75

Gov = 0.17



n = 4 # number of measurability constraints
nG = 1 # number of aggregate measurability constraints.
ny = 29 # number of individual controls (m_{t},mu_{t},c_{t},l_{t},rho1_,rho2,phi,x_{t-1},kappa_{t-1}) Note that the forward looking terms are at the end
ne = 8 # number of Expectation Terms (E_t u_{c,t+1}, E_t u_{c,t+1}mu_{t+1} E_{t}x_{t-1 E_t rho_{1,t-1}} [This makes the control x_{t-1},rho_{t-1} indeed time t-1 measuable])
nY = 12 # Number of aggregates (alpha_1,alpha_2,tau,eta,lambda)
nz = 4 # Number of individual states (m_{t-1},mu_{t-1})
nv = 6 # number of forward looking terms (x_t,rho1_t)
n_p = 3 #number of parameters
nZ = 1 # number of aggregate states
neps = len(sigma_e)

phat = np.array([-0.01,-0.005,0.0005])



def F(w):
    '''
    Individual first order conditions
    '''
    logm,muhat,nu_a,nu_e,logc,logl,logk_,lognl,w_e,r,pi,f,rho1,phi1,phi2,foc_k,foc_R,foc_K,rho2_,rho3_,b_,labor_res,foc_W,x_,alpha_,kappa_,rho2hat_,rho3hat_,lamb_ = w[:ny] #y
    EUc,EUc_r,Ex_,Ek_,EUc_mu,Erho2_,Erho3_,Efoc_k= w[ny:ny+ne] #e
    logK,Alpha_,Alpha2_,W,tau_l,tau_k,Xi,Eta,Kappa_,T,Zeta,B_ = w[ny+ne:ny+ne+nY] #Y
    logm_,muhat_,nu_a_,nu_e_= w[ny+ne+nY:ny+ne+nY+nz] #z
    x,alpha,kappa,rho2hat,rho3hat,lamb = w[ny+ne+nY+nz:ny+ne+nY+nz+nv] #v
    nu,nu_l,chi_psi = w[ny+ne+nY+nz+nv:ny+ne+nY+nz+nv+n_p] #v
    logK_ = w[ny+ne+nY+nz+nv+n_p+nZ-1] #Z
    eps_p,eps_t,eps_l_p,eps_l_t = w[ny+ne+nY+nz+nv+n_p+nZ:ny+ne+nY+nz+nv+n_p+nZ+neps] #shock
    Eps = w[ny+ne+nY+nz+nv+n_p+nZ+neps] #aggregate shock
    
    #if (ad.value(logm) < -3.5):
    #    shock = 0.
    #else:
    #    shock = 1.
    psi_hat,dpsi_hat,psi = 0.,0.,0.
    try:
        if (ad.value(b_) < 0):
            psi_hat = (chi_psi-1.) * b_**2
            dpsi_hat = (chi_psi-1.) * 2 * b_
            psi = -psi_hat*b_
        #if(ad.value(b_) <= -10.):
        #    psi_hat = (chi_psi-1.) * 100.
        #    dpsi_hat = -(chi_psi-1.) * 0.
    except:
        if (b_ < 0):
            psi_hat = (chi_psi-1.) * b_**2
            dpsi_hat = (chi_psi-1.) * 2 * b_
            psi = -(chi_psi-1.) * b_**3 
        
    R_ = Alpha_/(beta*Alpha2_)
    Ri_ = R_ + psi_hat * W
    m_,m = np.exp(logm_),np.exp(logm)
    K,K_= np.exp(logK),np.exp(logK_)
    c,l,k_,nl = np.exp(logc),np.exp(logl),np.exp(logk_),np.exp(lognl)
    mu_,mu = muhat_*m_,muhat*m
    
    Uc = c**(-sigma)
    Ucc = -sigma*c**(-sigma-1)
    Ul = -chi*l**(gamma)
    Ull = -chi*gamma*l**(gamma-1)
    A = np.exp(nu_a_+eps_p+eps_t)    
    
    r_k = A * xi_k*(xi_k-1) * k_**(xi_k-2) * nl**xi_l
    fnn = A  * xi_l*(xi_l-1) * k_**(xi_k) * nl**(xi_l-2)
    fn = A * xi_l * k_**(xi_k) * nl**(xi_l-1)
    fk = A * xi_k * k_**(xi_k-1) * nl**xi_l + (1-delta)
    fnk = A * xi_k * xi_l * k_**(xi_k-1) * nl**(xi_l-1)
    
    ret = np.empty(ny+n,dtype=w.dtype)
    ret[0] = x_ - Ex_ # x is independent of eps
    ret[1] = k_ - Ek_
    ret[2] = rho2_ - Erho2_
    ret[3] = rho3_ - Erho3_
    
    ret[4] = x_*Uc/(beta*EUc) + Uc*((1-tau_k)*pi - Ri_ *k_)  +(1-tau_l)*W*w_e*l*Uc - Uc*(c-T) - x #x
    ret[5] = alpha - m*Uc #rho1
    ret[6] = (1-tau_l)*W*Uc*w_e + Ul #phi1
    ret[7] = r -  fk#r
    ret[8] = W - fn #phi2
    ret[9] = pi - ( A*(1-xi_l)*k_**(xi_k)*nl**(xi_l) + (1-delta)*k_   ) #pi
    ret[10] = Alpha_ - alpha_ #alpha1_
    ret[11] = nu_a - (nu*(nu_a_+eps_p) + (1-nu)*mu_a ) #a
    ret[12] = f - A * k_**(xi_k) * nl**(xi_l) - (1-delta)*k_ #f  
    ret[13] = (Uc + Ucc*x_/(beta*EUc)*(mu-mu_) + mu*Ucc*( (1-tau_k)*pi-Ri_*k_  + (1-tau_l)*W*w_e*l-(c-T) -Uc/Ucc )
                +m*Ucc*rho1 + Ri_ *Ucc/(beta*EUc) * rho2_  
                + (1-tau_k)*Ucc*r/(beta*EUc_r)*rho3_ +(1-tau_l)*Ucc/Uc*phi1 - Xi)#c
    ret[14] = Ul +mu*(1-tau_l)*W*w_e*Uc - (1-tau_l)*Ull/Ul*phi1 + Eta*w_e - tau_l*W*w_e*Zeta#l
    ret[15] = foc_k - ( mu*Uc*((1-tau_k)*r - Ri_) + (1-tau_k)*Uc*r_k/EUc_r *rho3_/beta 
                -fnk*phi2 -Kappa_/beta + fk*Xi - lamb_/beta - tau_k*r*Zeta ) #fock
    ret[16] = -phi2*fnn - Eta + Xi*fn #nl
    ret[17] = rho1*Uc + rho2hat/m + rho3hat/m + x*lamb/alpha  #logm
    ret[18] = Kappa_ - kappa_ #kappa_
    ret[19] = rho2hat_ - Ri_ * rho2_ #temp1
    ret[20] = rho3hat_ - (1-tau_k) *rho3_ #temp2
    ret[21] = foc_R - (-beta*mu*Uc*k_ + rho2_ ) #foc_alpha2
    ret[22] = foc_K - (kappa-Xi) #foc_K
    ret[23] = nu_e - nu_l*(nu_e_ + eps_l_p)
    ret[24] = w_e - np.exp(nu_e_+ eps_l_p + eps_l_t)
    ret[25] = b_ - (x_*m_/Alpha_ - k_)
    ret[26] = lamb_ - ((rho2_ -beta*k_*EUc_mu + beta*Eta*b_)*dpsi_hat*W + beta*Eta*psi_hat*W)
    ret[27] = labor_res - (l*w_e - nl - psi)
    ret[28] = foc_W - ((1-tau_l)*w_e*l*Uc*mu + (1-tau_l)*phi1/W +phi2 - tau_l*w_e*l*Zeta +(rho2_/beta -k_*EUc_mu + Eta*b_)*psi_hat  )
    
    ret[29] = Efoc_k #k_
    ret[30] = mu_  - EUc_mu/EUc - m_*lamb_/Alpha_#muhat
    ret[31] = Alpha_ - beta*Ri_*m_*EUc #rho2_
    ret[32] = Alpha_ - beta*(1-tau_k)*m_*EUc_r #rho3
    
    return ret
    
def G(w):
    '''
    Aggregate equations
    '''
    logm,muhat,nu_a,nu_e,logc,logl,logk_,lognl,w_e,r,pi,f,rho1,phi1,phi2,foc_k,foc_R,foc_K,rho2_,rho3_,b_,labor_res,foc_W,x_,alpha_,kappa_,rho2hat_,rho3hat_,lamb_ = w[:ny] #y
    logK,Alpha_,Alpha2_,W,tau_l,tau_k,Xi,Eta,Kappa_,T,Zeta,B_ = w[ny+ne:ny+ne+nY] #Y
    logm_,muhat_,nu_a_,nu_e_= w[ny+ne+nY:ny+ne+nY+nz] #z
    logK_ = w[ny+ne+nY+nz+nv+n_p+nZ-1] #Z
    Eps = w[ny+ne+nY+nz+nv+n_p+nZ+neps] #aggregate shock
    
    c,l,k_,nl,m = np.exp(logc),np.exp(logl),np.exp(logk_),np.exp(lognl),np.exp(logm)
    mu = muhat*m
    K,K_= np.exp(logK),np.exp(logK_)
    Uc = c**(-sigma)
    
    
    ret = np.empty(nY+nG,dtype=w.dtype)
    ret[0] = logm-0.
    
    ret[1] = K_ - k_
    ret[2] = f - c - Gov - K
    ret[3] = labor_res
    ret[4] = pi*mu*Uc + rho3_/beta + pi*Zeta
    ret[5] = W*w_e*l*Uc*mu + phi1 + W*w_e*l*Zeta
    ret[6] = foc_R
    ret[7] = foc_W#(1-tau_l)*w_e*l*Uc*mu + (1-tau_l)*phi1/W +phi2 - tau_l*w_e*l*Zeta
    ret[8] = foc_K
    ret[9] = T + Gov#- (tau_k*pi + tau_l*W*w_e*l - Gov)
    ret[10] = Zeta #Uc*mu + Zeta
    ret[11] = B_ - b_
    
    ret[12] = logm-0.
    
    return ret
    
def f(y):
    '''
    Expectational equations that define e=Ef(y)
    '''
    logm,muhat,nu_a,nu_e,logc,logl,logk_,lognl,w_e,r,pi,f,rho1,phi1,phi2,foc_k,foc_R,foc_K,rho2_,rho3_,b_,labor_res,foc_W,x_,alpha_,kappa_,rho2hat_,rho3hat_,lamb_ = y
    
    c,k_ = np.exp(logc),np.exp(logk_)
    Uc = c**(-sigma)
    m = np.exp(logm)
    mu = muhat*m
    
            
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
    logm,muhat,nu_a,nu_e = z
    logK,Alpha_,Alpha2_,W,tau_l,tau_k,Xi,Eta,Kappa_,T,Zeta,B_ = YSS
    
    K = np.exp(logK)
    m = np.exp(logm)
    w_e = np.exp(nu_e)
    mu = muhat*m
    Uc =Alpha_/m
    c = (Uc)**(-1/sigma)
    Ucc = -sigma * c**(-sigma-1)
    
    Ul = -(1-tau_l)*Uc*W*w_e
    l = ( -Ul/chi )**(1./gamma)
    Ull = -chi * gamma * l**(gamma-1)
    
    r = 1./(beta*(1-tau_k)) * np.ones(c.shape)
    A = np.exp(nu_a)
    
    temp = A*xi_k*(W/(A*xi_l))**((xi_k-1)/xi_k)
    temp2 = xi_l + (xi_k-1)*(1-xi_l)/xi_k
    
    nl = ( (r-1+delta)/temp )**(1/temp2)
    k_ = (W/(xi_l*A))**(1/xi_k) * nl**((1-xi_l)/xi_k)
    alpha = Alpha_*np.ones(c.shape)
    alpha_ = alpha
    
    f = A*k_**(xi_k)*nl**(xi_l) + (1-delta) * k_
    fn = xi_l *A*k_**(xi_k)*nl**(xi_l-1)
    fnn = xi_l*(xi_l-1)*A*k_**(xi_k)*nl**(xi_l-2)
    fk = xi_k * A*k_**(xi_k-1)*nl**(xi_l) + (1-delta)
    fkk = xi_k *(xi_k-1) * A*k_**(xi_k-2)*nl**(xi_l)
    fnk = xi_k * xi_l * A*k_**(xi_k-1)*nl**(xi_l-1)
    
    pi = f - W*nl
    
    
    x_ = -( Uc*((1-tau_k)*pi- 1/beta*k_) -Ul*l - Uc*(c-T) )/(1/beta-1)
    b_ = x_/Uc-k_
    
    phi1 = (Ul + mu*(1-tau_l)*W*w_e*Uc + Eta*w_e - tau_l*W*w_e*Zeta)/( (1-tau_l)*Ull/Ul )
    #Ul - mu*(Ull*l + Ul) - phi1*Ull - Eta
    phi2 = (Xi*fn-Eta)/fnn
    
    rho3 = beta*(fnk*phi2 + Kappa_/beta - fk*Xi - tau_k*r*Zeta)/((1-tau_k)*fkk/r )
    rho1 = (Uc +mu*Ucc*( (1-tau_k)*pi-k_/beta + (1-tau_l)*W*w_e*l -(c-T) -Uc/Ucc ) + (1-tau_l)*Ucc/Uc*phi1 - Xi)/(Alpha_*Ucc/(beta*Uc) -Ucc*m)
    rho2 = -beta*(alpha*rho1 + (1-tau_k)*rho3)
    
    rho2hat_ = (1/beta) * rho2
    rho3hat_ = (1-tau_k) *rho3
    
    foc_R = (-beta*mu*Uc*k_ + rho2) #foc_alpha2
    foc_K = (Kappa_-Xi) *np.ones(c.shape) #foc_K
    kappa_ = Kappa_ * np.ones(c.shape)
    
    foc_k = ( (1-tau_k)*fkk/r *rho3/beta 
                -fnk*phi2 -Kappa_/beta + fk*Xi ) 
                
    lamb = np.zeros(c.shape)
    labor_res = w_e*l-nl
    foc_W = ((1-tau_l)*w_e*l*Uc*mu + (1-tau_l)*phi1/W +phi2 - tau_l*w_e*l*Zeta )
    
    return np.vstack((
    logm,muhat,nu_a,nu_e,np.log(c),np.log(l),np.log(k_),np.log(nl),w_e,r,pi,f,rho1,phi1,phi2,foc_k,foc_R,foc_K,rho2,rho3,b_,labor_res,foc_W,x_,alpha_,kappa_,rho2hat_,rho3hat_,lamb   
    ))
    
    

    
def GSS(YSS,y_i,weights):
    '''
    Aggregate conditions for the steady state
    '''
    logm,muhat,nu_a,nu_e,logc,logl,logk_,lognl,w_e,r,pi,f,rho1,phi1,phi2,foc_k,foc_R,foc_K,rho2_,rho3_,b_,labor_res,foc_W,x_,alpha_,kappa_,rho2hat_,rho3hat_,lamb_  = y_i
    logK,Alpha_,Alpha2_,W,tau_l,tau_k,Xi,Eta,Kappa_,T,Zeta,B_ = YSS
  
    c,l,k_,nl,m = np.exp(logc),np.exp(logl),np.exp(logk_),np.exp(lognl),np.exp(logm)
    mu = muhat*m
    K = np.exp(logK)   
    Uc = c**(-sigma)    
    
    return np.hstack(
    [weights.dot(K - k_), weights.dot(f - c - Gov - K), weights.dot(l*w_e - nl), weights.dot(pi*mu*Uc + rho3_/beta+pi*Zeta), weights.dot(W*w_e*l*Uc*mu + phi1 + W*w_e*l*Zeta),
     Alpha_-Alpha2_,weights.dot(foc_R), weights.dot((1-tau_l)*w_e*l*Uc*mu+ (1-tau_l)*phi1/W +phi2 - tau_l*w_e*l*Zeta),weights.dot(foc_K),T+Gov,Zeta,weights.dot(B_-b_)]#weights.dot(Zeta + Uc*mu),weights.dot(T - (tau_k*pi + tau_l*W*w_e*l-Gov)) ]    
    )
    
    
    
def nomalize(Gamma,weights =None):
    '''
    Normalizes the distriubtion of states if need be
    '''
    if weights == None:            
        Gamma[:,0] -= np.mean(Gamma[:,0])
        #Gamma[:,1] -= np.mean(Gamma[:,1])
    else:
        Gamma[:,0] -= weights.dot(Gamma[:,0])
        #Gamma[:,1] -= weights.dot(Gamma[:,1])
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
    if YSS[0] < -5.:
        return False
    return True
    