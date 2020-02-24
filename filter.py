import math as m
import numpy as np
import sympy as sp
from numpy import reshape as rs
from numpy import matrix as mat
import cvxpy as cp

def lin_obs_model():

    x1, x2 = sp.symbols('x1 x2')

    hx = sp.Matrix([x1, x2])

    return hx, x1, x2

def observation(M,X, Sigma_nu):

    hx, x1, x2 = lin_obs_model()
    nu1 = np.random.normal(0,np.sqrt(Sigma_nu))
    nu2 = np.random.normal(0,np.sqrt(Sigma_nu))
    nu = np.matrix([[nu1],[nu2]])
    Y = hx.subs([(x1,X[0]),(x2,X[1])]) + np.matmul(M,nu)

    return np.matrix(Y).astype(np.float64)

def obs_jacobian(X):

    hx, x1, x2 = lin_obs_model()

    H = hx.jacobian([x1,x2])

    return H.subs([(x1,X[0]),(x2,X[1])])

def EIF(system, X_prev_est, X_prev_act, O_prev):

    #prediction steps
    O_prev = mat(O_prev.reshape((3,3))).astype(np.float64)

    S_prev_est = np.matmul(O_prev,mat(rs(X_prev_est,(system.n,1))).astype(np.float64))
    S_prev_est = mat(rs(S_prev_est,(system.n,1))).astype(np.float64) #getting S in matrix form

    #X_prev_est = np.matmul(O_prev.I,S_prev_est)
    X_prior = mat(system.state_propagate(X_prev_est,0).reshape(system.n,1)).astype(np.float64)

    #print "X_prior", X_prior
    A = mat(system.jacobian(X_prev_est)).astype(np.float64)

    #print "A:",A

    O_prior = (np.matmul(np.matmul(A,O_prev.I),A.T) + np.matmul(np.matmul(system.G,system.Q),system.G.T)).I
    S_prior = np.matmul(O_prior,X_prior)

    #update
    X_act = rs(system.state_propagate(X_prev_act,system.Sigma_w),(system.n,))

    Y_act = observation(system.M,X_act, system.Sigma_nu)
    #print "Yact:",Y_act
    Y_est = observation(system.M,X_prior,0)


    #print "Y est:",Y_est

    #call all connected nodes and get data.
    H = mat(obs_jacobian(X_prior)).astype(np.float64)

    #print "H:",H
    z = np.matmul(np.matmul(H.T,system.R.I),Y_est)
    I = np.matmul(np.matmul(H.T,system.R.I),H)
    #print "I:",I

    O_post = O_prior + I
    #print "O_post:", O_post

    S_est = rs(X_prior,(3,1)) + z
    #print "S est:", S_est
    X_est = np.matmul(O_post.I,S_est)

    return rs(X_est,(3,)), X_act, O_post

def central_EIF(system, agents, X_prev_est, X_prev_act, O_prev):

    #prediction steps
    O_prev = mat(O_prev.reshape((system.n,system.n))).astype(np.float64)
    #print "O_prev",O_prev
    S_prev_est = np.matmul(O_prev,mat(rs(X_prev_est,(system.n,1))).astype(np.float64))
    S_prev_est = mat(rs(S_prev_est,(system.n,1))).astype(np.float64) #getting S in matrix form

    #X_prev_est = np.matmul(O_prev.I,S_prev_est)
    X_prior = mat(system.state_propagate(X_prev_est,0).reshape(system.n,1)).astype(np.float64)

    #print "X_prior", X_prior
    A = mat(system.jacobian(X_prev_est)).astype(np.float64)

    ##print "A:",A

    O_prior = (np.matmul(np.matmul(A,O_prev.I),A.T) + np.matmul(np.matmul(system.G,system.Q),system.G.T)).I
    S_prior = np.matmul(O_prior,X_prior)
    #print "O_prior:", O_prior
    #update
    X_act = rs(system.state_propagate(X_prev_act,system.Sigma_w),(system.n,))
    #print "Xact:",X_act

    H = mat(obs_jacobian(X_prior)).astype(np.float64)

    S_est =  rs(S_prior,(system.n,1))
    O_post = O_prior

    for i in range(len(agents)):

        #print i, " Status:", agents[i].check_fov(X_act[0],X_act[1])
        if agents[i].check_fov(X_act[0],X_act[1]):
            Y_act = observation(system.M,X_act, system.Sigma_nu)
            #print "Yact:",Y_act
            #Y_est = observation(system.M,X_prior,0)
            ##print "Y est:",Y_est

            z = np.matmul(np.matmul(H.T,system.R.I),Y_act)
            #print "z:", z
            I = np.matmul(np.matmul(H.T,system.R.I),H)
            #print "I:",I

            O_post = O_post + I
            #print "O_post:", O_post

            S_est = S_est + z
        #print "S est:", S_est

    X_est = np.matmul(O_post.I,S_est)
    #print "X_est:", X_est

    return rs(X_est,(system.n,)), X_act, O_post



def partial_EIF(system, agent, X_prev_est, X_act, O_prev):

    #prediction steps
    O_prev = mat(O_prev.reshape((system.n,system.n))).astype(np.float64)
    #print "O_prev",O_prev
    S_prev_est = np.matmul(O_prev,mat(rs(X_prev_est,(system.n,1))).astype(np.float64))
    S_prev_est = mat(rs(S_prev_est,(system.n,1))).astype(np.float64) #getting S in matrix form

    #X_prev_est = np.matmul(O_prev.I,S_prev_est)
    X_prior = mat(system.state_propagate(X_prev_est,0).reshape(system.n,1)).astype(np.float64)

    #print "X_prior", X_prior
    A = mat(system.jacobian(X_prev_est)).astype(np.float64)

    #print "A:",A

    O_prior = (np.matmul(np.matmul(A,O_prev.I),A.T) + np.matmul(np.matmul(system.G,system.Q),system.G.T)).I
    S_prior = np.matmul(O_prior,X_prior)
    #print "O_prior:", O_prior

    #update
    if agent.check_fov(X_act[0],X_act[1]):

        H = mat(obs_jacobian(X_prior)).astype(np.float64)

        Y_act = observation(system.M,X_act, system.Sigma_nu)
        z = np.matmul(np.matmul(H.T,system.R.I),Y_act)
        #print "z:", z
        I = np.matmul(np.matmul(H.T,system.R.I),H)

        S_est =  rs(S_prior,(system.n,1)) + z
        O_post = O_prior + I

        X_est = np.matmul(O_post.I,S_est)

    else:
        O_post = O_prior
        X_est = X_prior

    return rs(X_est,(system.n,)), O_post

def priors_meas(system, agent, X_prev_est, X_act, O_prev):

    #prediction steps
    O_prev = mat(O_prev.reshape((system.n,system.n))).astype(np.float64)
    #print "O_prev",O_prev
    S_prev_est = np.matmul(O_prev,mat(rs(X_prev_est,(system.n,1))).astype(np.float64))
    S_prev_est = mat(rs(S_prev_est,(system.n,1))).astype(np.float64) #getting S in matrix form

    #X_prev_est = np.matmul(O_prev.I,S_prev_est)
    X_prior = mat(system.state_propagate(X_prev_est,0).reshape(system.n,1)).astype(np.float64)

    #print "X_prior", X_prior
    A = mat(system.jacobian(X_prev_est)).astype(np.float64)

    #print "A:",A

    O_prior = (np.matmul(np.matmul(A,O_prev.I),A.T) + np.matmul(np.matmul(system.G,system.Q),system.G.T)).I
    S_prior = np.matmul(O_prior,X_prior)
    #print "O_prior:", O_prior

    #update
    if agent.check_fov(X_act[0],X_act[1]):

        H = mat(obs_jacobian(X_prior)).astype(np.float64)

        Y_act = observation(system.M,X_act, system.Sigma_nu)
        z = np.matmul(np.matmul(H.T,system.R.I),Y_act)
        #print "z:", z
        I = np.matmul(np.matmul(H.T,system.R.I),H)


    else:

        z = np.zeros(system.n)
        I = (10**(-6))*np.eye(system.n) #setting to 0 information

    return X_prior.reshape((system.n,)),O_prior,z.reshape((system.n,)),I.reshape((system.n**2,))

def cost_func(W,agents,conn_agents,agent_idx,t):

    n = len(agents[0].position) #dim of state
    S = np.zeros((n,n))

    j = 0
    for i in conn_agents:

        S = S + W[j]*mat(agents[i].O[:,t].reshape(n,n)).astype(np.float64)
        j = j+1

    #return -cp.log_det(S)
    return cp.atoms.affine.trace.trace(S)

def call_sum(W):

    W_sum = 0
    for i in range(W.size()[0]):
        W_sum = W_sum + W[i]

    return W_sum

def CI(agents,adj_mat,agent_idx,t):

    conn_agents = []
    for i in range(len(adj_mat[agent_idx,:])):

        #find connected agent indexes
        if adj_mat[agent_idx,i] == 1:
            conn_agents.append(i)

    #optimize to find weights
    W = cp.Variable(len(conn_agents))
    A = np.ones((1,len(conn_agents)))
    prob = cp.Problem(cp.Minimize(cost_func(W,agents,conn_agents,agent_idx,t)),[A*W==1,W>=np.zeros(len(conn_agents))])
    prob.solve()
    #print "W_CI:",W.value

    n = len(agents[0].position)
    O_new = np.zeros((n,n))
    S_est = np.zeros((n,1))
    j=0
    X_est = np.zeros((n,1))
    for i in conn_agents:

        O_new = O_new + W.value[j]*agents[i].O[:,t].reshape(n,n)
        S_est = S_est + W.value[j]*np.matmul(mat(agents[i].O[:,t].reshape(n,n)).astype(np.float64),mat(agents[i].X_est[:,t].reshape((n,1))).astype(np.float64))

        j = j+1

    #print "S:",S
    O_new = mat(O_new).astype(np.float64)
    X_est = np.matmul(O_new.I,S_est)

    return X_est.reshape((n,)), O_new

def consensus(agents,adj_mat,agent_idx,t):

    conn_agents = []
    for i in range(len(adj_mat[agent_idx,:])):

        #find connected agent indexes
        if adj_mat[agent_idx,i] == 1:
            conn_agents.append(i)

    W = np.zeros(len(agents))
    for j in conn_agents:

        if j != agent_idx:

            W[j] = 1.0/(1.0 + max(len(conn_agents)-1,np.sum(adj_mat[j,:])-1))

    W[agent_idx] = 1.0 - np.sum(W)
    #print "agent_idx:",agent_idx
    #print "W_HYB:",W

    del_i_bar = np.zeros(len(agents[0].position))
    del_I_bar = np.zeros((len(agents[0].position)**2))

    for j in conn_agents:

        del_i_bar = del_i_bar + W[j]*agents[j].del_i[:,t]
        del_I_bar = del_I_bar + W[j]*agents[j].del_I[:,t]

    return del_i_bar, del_I_bar
