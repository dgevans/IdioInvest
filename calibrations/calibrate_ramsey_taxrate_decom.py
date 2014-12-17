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
sigma_E = 0.*np.eye(1)
mu_e = 0.
mu_a = 0.
chi = 1.
delta = 0.06
xi_l = 0.66*(1-0.21) #*.75
xi_k = 0.34*(1-0.21) #*.75

Gov = 0.17



n = 4 # number of measurability constraints
nG = 3 # number of aggregate measurability constraints.
ny = 41 # number of individual controls (m_{t},mu_{t},c_{t},l_{t},rho1_,rho2,phi,x_{t-1},kappa_{t-1}) Note that the forward looking terms are at the end
ne = 14 # number of Expectation Terms (E_t u_{c,t+1}, E_t u_{c,t+1}mu_{t+1} E_{t}x_{t-1 E_t rho_{1,t-1}} [This makes the control x_{t-1},rho_{t-1} indeed time t-1 measuable])
nY = 19 # Number of aggregates (alpha_1,alpha_2,tau,eta,lambda)
nz = 4 # Number of individual states (m_{t-1},mu_{t-1})
nv = 6 # number of forward looking terms (x_t,rho1_t)
n_p = 3 #number of parameters
nZ = 1 # number of aggregate states
nEps = 1
neps = len(sigma_e)

phat = np.array([-0.01,-0.005,0.0005])

temp = 0.



def F(w):
    '''
    Individual first order conditions
    '''
    logm,muhat,nu_a,nu_e,logc,logl,lognl,w_e,r,pi,f,rho1,phi1,phi2,foc_k,foc_R,foc_tau_k,rho2_,rho3_,b_,labor_res,res,foc_W,r_k,dk_dtau_k,tR_khat,tW_khat,tE_khat,tR_k,tW_k,tE_k,dl_dtau,tR_l,tW_l,tE_l,x_,rho2hat_,rho3hat_,lamb_,logk_,alpha_ = w[:ny] #y
    EUc,EUc_r,Ex_,Ek_,EUc_mu,Erho2_,Erho3_,Efoc_k,Efoc_tau_k,EUc_r_k,Er,EtR_khat,EtW_khat,EtE_khat= w[ny:ny+ne] #e
    logXi,logAlpha_,logK_,R_,W,tau_l,tau_k,Eta,T,B_,dK_dtau_k,dL_dtau,dUc_dtau_l,TR_k,TW_k,TE_k,TR_l,TW_l,TE_l = w[ny+ne:ny+ne+nY] #Y
    logm_,muhat_,nu_a_,nu_e_= w[ny+ne+nY:ny+ne+nY+nz] #z
    x,rho2hat,rho3hat,lamb,logk,alpha = w[ny+ne+nY+nz:ny+ne+nY+nz+nv] #v
    nu,nu_l,chi_psi = w[ny+ne+nY+nz+nv:ny+ne+nY+nz+nv+n_p] #p
    logXi_ = w[ny+ne+nY+nz+nv+n_p] #Z
    eps_p,eps_t,eps_l_p,eps_l_t = w[ny+ne+nY+nz+nv+n_p+nZ:ny+ne+nY+nz+nv+n_p+nZ+neps] #shock
    Eps = w[ny+ne+nY+nz+nv+n_p+nZ+neps] #aggregate shock
    
    #if (ad.value(logm) < -3.5):
    #    shock = 0.
    #else:
    #    shock = 1.
    psi_hat,psi,dpsi_hat = 0.,0.,0.
    try:
        if (ad.value(b_) < 0):
            psi_hat = -2*0.001*(chi_psi-1.) * b_
            psi = -0.001*(chi_psi-1.) * b_**2
            dpsi_hat = -2*0.001*(chi_psi-1.)
    except:
        if (b_ < 0):
            psi_hat = -2*0.001*(chi_psi-1.) * b_
            psi = -0.001*(chi_psi-1.) * b_**2
            dpsi_hat = -2*0.001*(chi_psi-1.)
            
    Alpha_ = np.exp(logAlpha_)
    Xi_,Xi = np.exp(logXi_),np.exp(logXi)
    Ri_ = R_ + psi_hat * W
    m_,m = np.exp(logm_),np.exp(logm)
    c,l,k_,nl,k = np.exp(logc),np.exp(logl),np.exp(logk_),np.exp(lognl),np.exp(logk)
    mu_,mu = muhat_*m_,muhat*m
    
    cov_Uc_r = EUc_r - EUc*Er
    
    Uc = c**(-sigma)
    Ucc = -sigma*c**(-sigma-1)
    Ul = -chi*l**(gamma)
    Ull = -chi*gamma*l**(gamma-1)
    A = np.exp((1-temp)*Eps*xi_l + nu_a_+eps_p+eps_t)    
    
    
    fnn = A  * xi_l*(xi_l-1) * k_**(xi_k) * nl**(xi_l-2)
    fn = A * xi_l * k_**(xi_k) * nl**(xi_l-1)
    fk = A * xi_k * k_**(xi_k-1) * nl**xi_l + (1-delta)
    fnk = A * xi_k * xi_l * k_**(xi_k-1) * nl**(xi_l-1)
    
    pi_k = A*(1-xi_l)*xi_k*k_**(xi_k-1)*nl**(xi_l) + (1-delta)
    pi_n = A*(1-xi_l)*xi_l*k_**(xi_k)*nl**(xi_l-1) 
    
    ret = np.empty(ny+n,dtype=w.dtype)
    ret[0] = x_ - Ex_ # x is independent of eps
    ret[1] = k_ - Ek_
    ret[2] = rho2_ - Erho2_
    ret[3] = rho3_ - Erho3_
    
    ret[4] = x_*Uc/(beta*EUc) + Uc*W*(psi-psi_hat*b_) + Uc*((1-tau_k)*(pi-k_) - (Ri_-1) *k_)  +(1-tau_l)*W*w_e*l*Uc - Uc*(c-T) - x #x
    ret[5] = alpha - m*Uc #rho1
    ret[6] = (1-tau_l)*W*Uc*w_e + Ul #phi1
    ret[7] = r -  (fk-1)#r
    ret[8] = W - fn #phi2
    ret[9] = pi - ( A*(1-xi_l)*k_**(xi_k)*nl**(xi_l) + (1-delta)*k_   ) #pi
    ret[10] = nu_a - (nu*(nu_a_+eps_p) + (1-nu)*mu_a ) #a
    ret[11] = f - A * k_**(xi_k) * nl**(xi_l) - (1-delta)*k_ #f  
    ret[12] = (Uc + Ucc*x_/(beta*EUc)*(mu-mu_) + mu*Ucc*( W*(psi-psi_hat*b_)+(1-tau_k)*(pi-k_)-(Ri_-1.)*k_  + (1-tau_l)*W*w_e*l-(c-T) -Uc/Ucc )
                +m*Ucc*rho1 + Ri_ *Ucc/(beta*EUc) * rho2_  
                + (1-tau_k)*Ucc*r/(beta*EUc_r)*rho3_ +(1-tau_l)*Ucc/Uc*phi1 - Xi)#c
    ret[13] = Ul +mu*(1-tau_l)*W*w_e*Uc - (1-tau_l)*Ull/Ul*phi1 + Eta*w_e#l
    ret[14] = foc_k - ( mu*Uc*((1-tau_k)*(pi_k-1) - (Ri_-1)) + (1-tau_k)*Uc*r_k/EUc_r *rho3_/beta 
                -fnk*phi2 -Xi_/beta + fk*Xi - lamb_/beta  ) #fock
    ret[15] = -phi2*fnn - Eta + Xi*fn  + mu*Uc*(1-tau_k)*pi_n + (1-tau_k)*Uc*fnk/(beta*EUc_r)*rho3_ #nl
    ret[16] = rho1*Uc + rho2hat/m + rho3hat/m + x*lamb/alpha  #logm
    ret[17] = rho2hat_ - Ri_ * rho2_ #temp1
    ret[18] = rho3hat_ - (1-tau_k) *rho3_ #temp2
    ret[19] = foc_R - (-beta*EUc_mu*k_ - (1-tau_k)*rho3_/(Ri_*(Ri_-1))+ rho2_ ) #foc_alpha2
    ret[20] = nu_e - nu_l*(nu_e_ + eps_l_p) - (1-nu_l)*mu_e
    ret[21] = w_e - np.exp(temp*Eps+nu_e_+ eps_l_p + eps_l_t)
    ret[22] = b_ - (x_*m_/Alpha_ - k_)
    ret[23] = lamb_ - (beta*Eta*psi_hat - beta*Uc*mu*dpsi_hat*b_ + W*dpsi_hat*foc_R)#((rho2_ -beta*k_*EUc_mu - (1-tau_k)*rho3_/(Ri_*(Ri_-1))  + beta*Eta*b_)*dpsi_hat*W + beta*Eta*psi_hat*W)
    ret[24] = labor_res - (l*w_e - nl + psi)
    ret[25] = foc_W - ((1-tau_l)*w_e*l*Uc*mu + (1-tau_l)*phi1/W +phi2 + Uc*mu*(psi-dpsi_hat*b_) + foc_R*psi_hat/beta) #+(rho2_/beta -k_*EUc_mu - (1-tau_k)*rho3_/(Ri_*(Ri_-1))/beta+ Eta*b_)*psi_hat  )
    ret[26] = foc_tau_k - Efoc_tau_k
    ret[27] = res - (f - c - Gov - k)
    ret[28] = alpha_ - Alpha_
    ret[29] = r_k - (A * xi_k*(xi_k-1) * k_**(xi_k-2) * nl**xi_l)
    ret[30] = dk_dtau_k - ( EUc_r/((1-tau_k)*EUc_r_k) )
    ret[31] = tR_khat - mu*Uc*( (pi-k_) - dk_dtau_k * ((1-tau_k)*(pi_k-1) - (R_-1)))*(1-tau_k)/((R_-1)*dK_dtau_k*Xi)
    ret[32] = tW_khat - (dk_dtau_k * fnk * phi2) *(1-tau_k)/((R_-1)*dK_dtau_k*Xi)
    ret[33] = tE_khat - (dK_dtau_k * (Xi_/beta - Xi * R_) + dk_dtau_k*cov_Uc_r*Xi/EUc)*(1-tau_k)/((R_-1)*dK_dtau_k*Xi)
    ret[34] = tR_k - EtR_khat
    ret[35] = tW_k - EtW_khat
    ret[36] = tE_k - EtE_khat
    ret[37] = dl_dtau - ( -Ul/((1-tau_l)*Ull) )
    ret[38] = tR_l  + ((1-tau_l)*W*w_e*mu*Uc*dl_dtau -W*w_e*l*Uc*mu)/dUc_dtau_l#+ (mu*Uc*w_e*W*(dl_dtau*(1-tau_l) - l)/dUc_dtau_l)
    ret[39] = tW_l - (W*Xi-Eta)*dL_dtau/dUc_dtau_l#(dl_dtau*W*w_e*(Uc-Xi))/dUc_dtau_l
    ret[40] = tE_l - ((dl_dtau*W*w_e*(Uc-Xi))/dUc_dtau_l)
    
    
    ret[41] = Efoc_k #k_
    ret[42] = mu_  - EUc_mu/EUc - m_*lamb_/Alpha_#muhat
    ret[43] = Alpha_ - beta*Ri_*m_*EUc #rho2_
    ret[44] = Alpha_*(1-1/Ri_) - beta*(1-tau_k)*m_*EUc_r #rho3
    
    return ret
    
def G(w):
    '''
    Aggregate equations
    '''
    logm,muhat,nu_a,nu_e,logc,logl,lognl,w_e,r,pi,f,rho1,phi1,phi2,foc_k,foc_R,foc_tau_k,rho2_,rho3_,b_,labor_res,res,foc_W,r_k,dk_dtau_k,tR_khat,tW_khat,tE_khat,tR_k,tW_k,tE_k,dl_dtau,tR_l,tW_l,tE_l,x_,rho2hat_,rho3hat_,lamb_,logk_,alpha_ = w[:ny] #y
    logXi,logAlpha_,logK_,R_,W,tau_l,tau_k,Eta,T,B_,dK_dtau_k,dL_dtau,dUc_dtau_l,TR_k,TW_k,TE_k,TR_l,TW_l,TE_l = w[ny+ne:ny+ne+nY] #Y
    logm_,muhat_,nu_a_,nu_e_= w[ny+ne+nY:ny+ne+nY+nz] #z
    logXi_ = w[ny+ne+nY+nz+nv+n_p] #Z
    Eps = w[ny+ne+nY+nz+nv+n_p+nZ+neps] #aggregate shock
    
    c,l,k_,nl,m = np.exp(logc),np.exp(logl),np.exp(logk_),np.exp(lognl),np.exp(logm)
    mu = muhat*m
    K_ = np.exp(logK_)
    Uc = c**(-sigma)
    
    
    ret = np.empty(nY+nG,dtype=w.dtype)
    ret[0] = R_
    ret[1] = tau_k
    ret[2] = logK_
    
    ret[3] = res
    ret[4] = labor_res
    ret[5] = W*w_e*l*Uc*mu + phi1 
    ret[6] = foc_W#(1-tau_l)*w_e*l*Uc*mu + (1-tau_l)*phi1/W +phi2 - tau_l*w_e*l*Zeta
    ret[7] = B_ - b_
    ret[8] = logm-0.
    ret[9] = T + Gov
    ret[10] = dK_dtau_k - dk_dtau_k
    ret[11] = TR_k - tR_k
    ret[12] = TW_k - tW_k
    ret[13] = TE_k - tE_k
    ret[14] = dUc_dtau_l - dl_dtau*W*Uc*w_e
    ret[15] = dL_dtau - dl_dtau*w_e
    ret[16] = TR_l - tR_l
    ret[17] = TW_l - tW_l
    ret[18] = TE_l - tE_l
    
    ret[19] = K_ - k_
    ret[20] = foc_tau_k
    ret[21] = foc_R
    
    return ret
    
def f(y):
    '''
    Expectational equations that define e=Ef(y)
    '''
    logm,muhat,nu_a,nu_e,logc,logl,lognl,w_e,r,pi,f,rho1,phi1,phi2,foc_k,foc_R,foc_tau_k,rho2_,rho3_,b_,labor_res,res,foc_W,r_k,dk_dtau_k,tR_khat,tW_khat,tE_khat,tR_k,tW_k,tE_k,dl_dtau,tR_l,tW_l,tE_l,x_,rho2hat_,rho3hat_,lamb_,logk_,alpha_ = y
    
    c,k_ = np.exp(logc),np.exp(logk_)
    Uc = c**(-sigma)
    m = np.exp(logm)
    mu = muhat*m
    
            
    #EUc,EUc_r,Ex_,Ek_,EUc_mu,Erho2_,Erho3_,Efoc_k,Efoc_tau_k,EUc_r_k,Er,EtR_khat,EtW_khat,EtE_khat
    ret = np.empty(ne,dtype=y.dtype)
    ret[0] = Uc
    ret[1] = Uc * r
    ret[2] = x_
    ret[3] = k_
    ret[4] = Uc*mu
    ret[5] = rho2_
    ret[6] = rho3_
    ret[7] = foc_k   
    ret[8] = (pi-k_)*mu*Uc + rho3_/beta
    ret[9] = Uc* r_k
    ret[10] = r
    ret[11] = tR_khat
    ret[12] = tW_khat
    ret[13] = tE_khat
        
    
    return ret
    
def Finv(YSS,z):
    '''
    Given steady state YSS solves for y_i
    '''
    logm,muhat,nu_a,nu_e = z
    logXi,logAlpha,logK_,R_,W,tau_l,tau_k,Eta,T,B_,dK_dtau_k,dL_dtau,dUc_dtau_l,TR_k,TW_k,TE_k,TR_l,TW_l,TE_l = YSS
    
    Xi = np.exp(logXi)
    Alpha_ = np.exp(logAlpha)
        
    
    m = np.exp(logm)
    w_e = np.exp(nu_e)
    mu = muhat*m
    Uc =Alpha_/m
    c = (Uc)**(-1/sigma)
    Ucc = -sigma * c**(-sigma-1)
    
    Ul = -(1-tau_l)*Uc*W*w_e
    l = ( -Ul/chi )**(1./gamma)
    Ull = -chi * gamma * l**(gamma-1)
    
    r = (1./beta-1)/(1-tau_k) * np.ones(c.shape)
    A = np.exp(nu_a)
    
    temp = A*xi_k*(W/(A*xi_l))**((xi_k-1)/xi_k)
    temp2 = xi_l + (xi_k-1)*(1-xi_l)/xi_k
    
    nl = ( (r+delta)/temp )**(1/temp2)
    k_ = (W/(xi_l*A))**(1/xi_k) * nl**((1-xi_l)/xi_k)
    alpha = Alpha_*np.ones(c.shape)
    
    f = A*k_**(xi_k)*nl**(xi_l) + (1-delta) * k_
    fn = xi_l *A*k_**(xi_k)*nl**(xi_l-1)
    fnn = xi_l*(xi_l-1)*A*k_**(xi_k)*nl**(xi_l-2)
    fk = xi_k * A*k_**(xi_k-1)*nl**(xi_l) + (1-delta)
    fkk = xi_k *(xi_k-1) * A*k_**(xi_k-2)*nl**(xi_l)
    fnk = xi_k * xi_l * A*k_**(xi_k-1)*nl**(xi_l-1)
    
    pi = f - W*nl
    pi_k = A*(1-xi_l)*xi_k*k_**(xi_k-1)*nl**(xi_l) + (1-delta)
    pi_n = A*(1-xi_l)*xi_l*k_**(xi_k)*nl**(xi_l-1) 
    
    
    x_ = -( Uc*((1-tau_k)*(pi-k_)- (1/beta-1)*k_) -Ul*l - Uc*(c-T) )/(1/beta-1)
    b_ = x_/Uc-k_
    
    phi1 = (Ul + mu*(1-tau_l)*W*w_e*Uc + Eta*w_e )/( (1-tau_l)*Ull/Ul )
    #Ul - mu*(Ull*l + Ul) - phi1*Ull - Eta
    #rho3 = beta*(-Uc*mu*((1-tau_k)*(pi_k-pi_n*fnk/fnn)- 1/beta) + Xi*fn*fnk/fnn - Eta*fnk/fnn
    #                + Xi/beta - fk*Xi -   tau_k*r*Zeta)/((1-tau_k)*(fkk/r-fnk*fnk/(fnn*r)) )
    
    rho3 = -(beta*(mu*Uc*((1-tau_k)*((pi_k-1)-pi_n*fnk/fnn)-(1/beta-1)) - Xi/beta + fk*Xi - Xi*fn*fnk/fnn + Eta*fnk/fnn)/
                   ((1-tau_k)*(fkk/r-fnk*fnk/(fnn*r)) ) )

    phi2 = (Xi*fn-Eta +Uc*mu*(1-tau_k)*pi_n + (1-tau_k)*fnk/(beta*r)*rho3  )/fnn
    
    rho1 = (Uc +mu*Ucc*( (1-tau_k)*(pi-k_)-k_*(1/beta-1) + (1-tau_l)*W*w_e*l -(c-T) -Uc/Ucc ) + (1-tau_l)*Ucc/Uc*phi1 - Xi)/(Alpha_*Ucc/(beta*Uc) -Ucc*m)
    rho2 = -beta*(alpha*rho1 + (1-tau_k)*rho3)
    
    rho2hat_ = (1/beta) * rho2
    rho3hat_ = (1-tau_k) * rho3
    
    foc_R = (-beta*mu*Uc*k_-(1-tau_k)*rho3/(R_*(R_-1)) + rho2) #foc_alpha2
    
    foc_k = ( mu*Uc*((1-tau_k)*(pi_k-1) - (1/beta-1))+ (1-tau_k)*fkk/r *rho3/beta 
                -fnk*phi2 -Xi/beta + fk*Xi ) 
                
    lamb = np.zeros(c.shape)
    labor_res = w_e*l-nl
    foc_W = ((1-tau_l)*w_e*l*Uc*mu + (1-tau_l)*phi1/W +phi2  )
    
    foc_tau_k = (pi-k_)*mu*Uc + rho3/beta 
    res = (f - c - Gov - k_)
    alpha_ = Alpha_ *np.ones(c.shape)
    
    r_k = (A * xi_k*(xi_k-1) * k_**(xi_k-2) * nl**xi_l)
    dk_dtau_k = ( r/((1-tau_k)*r_k) )
    tR_khat = mu*Uc*( (pi-k_) - dk_dtau_k * ((1-tau_k)*(pi_k-1) - (R_-1)))*(1-tau_k)/((R_-1)*dK_dtau_k*Xi)
    tW_khat = (dk_dtau_k * fnk * phi2) *(1-tau_k)/((R_-1)*dK_dtau_k*Xi)
    tE_khat = (dK_dtau_k * (Xi/beta - Xi * R_) )*(1-tau_k)/((R_-1)*dK_dtau_k*Xi) * np.ones(c.shape)
    
    dl_dtau = ( -Ul/((1-tau_l)*Ull) )
    tR_l = -((1-tau_l)*W*w_e*mu*Uc*dl_dtau -W*w_e*l*Uc*mu)/dUc_dtau_l#+ (mu*Uc*w_e*W*(dl_dtau*(1-tau_l) - l)/dUc_dtau_l)
    tW_l = (W*Xi-Eta)*dL_dtau/dUc_dtau_l * np.ones(c.shape)#(dl_dtau*W*w_e*(Uc-Xi))/dUc_dtau_l
    tE_l =  ((dl_dtau*W*w_e*(Uc-Xi))/dUc_dtau_l)
    
    
    #logm,muhat,nu_a,nu_e,logc,    logl,       lognl,    w_e,r,pi,f,rho1,phi1,phi2,foc_k,foc_R,foc_tau_k,rho2_,rho3_,b_,labor_res,res,foc_W,zeta,x_,rho2hat_,rho3hat_,lamb_,logk_
    return np.vstack((
    logm,muhat,nu_a,nu_e,np.log(c),np.log(l),np.log(nl),w_e,r,pi,f,rho1,phi1,phi2,foc_k,foc_R,foc_tau_k,rho2,rho3,b_,  labor_res,res,foc_W,r_k,dk_dtau_k,tR_khat,tW_khat,tE_khat,tR_khat,tW_khat,tE_khat,dl_dtau,tR_l,tW_l,tE_l,x_,rho2hat_,rho3hat_,lamb,np.log(k_),alpha_   
    ))
    
    

    
def GSS(YSS,y_i,weights):
    '''
    Aggregate conditions for the steady state
    '''
    logm,muhat,nu_a,nu_e,logc,logl,lognl,w_e,r,pi,f,rho1,phi1,phi2,foc_k,foc_R,foc_tau_k,rho2_,rho3_,b_,labor_res,res,foc_W,r_k,dk_dtau_k,tR_khat,tW_khat,tE_khat,tR_k,tW_k,tE_k,dl_dtau,tR_l,tW_l,tE_l,x_,rho2hat_,rho3hat_,lamb_,logk_,alpha_  = y_i
    logXi,logAlpha,logK_,R_,W,tau_l,tau_k,Eta,T,B_,dK_dtau_k,dL_dtau,dUc_dtau_l,TR_k,TW_k,TE_k,TR_l,TW_l,TE_l = YSS
  
    Xi = np.exp(logXi)
    Alpha_ = np.exp(logAlpha)
    c,l,k_,nl,m = np.exp(logc),np.exp(logl),np.exp(logk_),np.exp(lognl),np.exp(logm)
    mu = muhat*m
    K = np.exp(logK_)   
    Uc = c**(-sigma)   
    
    
    return np.hstack(
    [weights.dot(K - k_), weights.dot(f - c - Gov - K), weights.dot(l*w_e - nl), weights.dot((pi-k_)*mu*Uc + rho3_/beta), weights.dot(W*w_e*l*Uc*mu + phi1 ),
     R_-1/beta,weights.dot(foc_R), weights.dot((1-tau_l)*w_e*l*Uc*mu+ (1-tau_l)*phi1/W +phi2 ),T+Gov,weights.dot(B_-b_) ,
     weights.dot(dK_dtau_k-dk_dtau_k),weights.dot(TR_k- tR_k),weights.dot(TW_k-tW_k),weights.dot(TE_k-tE_k),
     weights.dot(dUc_dtau_l - dl_dtau*W*Uc*w_e), weights.dot(dL_dtau - dl_dtau*w_e), weights.dot(TR_l - tR_l),
     weights.dot(TW_l - tW_l),weights.dot(TE_l - tE_l)]#weights.dot(Zeta + Uc*mu),weights.dot(T - (tau_k*pi + tau_l*W*w_e*l-Gov)) ]    
    )
    
    
    
def nomalize(Gamma,weights =None):
    '''
    Normalizes the distriubtion of states if need be
    '''
    if weights == None:            
        Gamma[:,0] -= np.mean(Gamma[:,0])
        if phat[2]  == 0:
            Gamma[:,1] -= np.mean(Gamma[:,1])
    else:
        Gamma[:,0] -= weights.dot(Gamma[:,0])
        if phat[2] == 0:
            Gamma[:,1] -= weights.dot(Gamma[:,1])
    return Gamma
    
def check_extreme(z_i):
    '''
    Checks for extreme positions in the state space
    '''
    extreme = False
    if z_i[0] < -3. or z_i[0] > 6.:
        extreme = True
    if z_i[1] > 8. or z_i[1] < -8.:
        extreme = True
    return extreme
    
def check_SS(YSS):
    if YSS[2] < -1.5:
        return False
    return True
    
def check(y_i):
    return False
    