import problem_framework as pb
import math as m
import numpy as np
import pylab
from matplotlib.patches import Rectangle
from scipy import stats, misc
import scipy
import filter
from numpy import reshape as rs
from numpy import matrix as mat



# variables
grid_x = 10.0
grid_y = 10.0
fov = [grid_x/1.5, grid_y/2.5] #width, height
num_agents = 4 # defined the number of observers
position_vec = np.array([[4,6],[5,4],[7,7],[0,0]])
orientation_vec = np.array([45,30,-135,60]) #in degrees
#position_vec = np.array([[40,60],[15,40],[74,77],[0,0],[50,50],[65,80],[74,3],[80,10]])
#orientation_vec = np.array([45,30,-135,60,32,-80,90,-215]) #in degrees
t_step = 30# number of different configuraions of random communication connectivity
comm_prob_ref = 0.5 # define the communication probablity threshold. Any communcation channel cdf generated below this threshold are identifed as connected
FLAG_CONVERGENCE = False
FLAG_CONVERGENCE_ICI = False
max_iter_conv = 10
no_MC_iters = 50
FILE_WRITE = True

if FILE_WRITE:
    filename = "/home/naveed/Documents/distribute_state_estimation/NEES_Hyb1.csv"
    file = open(filename,"a")

# function to plot the agents FOV, and position
def plot(num_agents,agents,grid_x,grid_y,fov,x,y):

    # create color vector
    color_temp = np.array(['red','blue','green','yellow'])
    color =  color_temp # assigned by default, size adjustments are performed in next line

    #assign color
    if num_agents > len(color_temp):
        pad_size = num_agents - len(color_temp)
        color = np.pad(color_temp,(0,pad_size),'wrap')

    pylab.figure(1)
    ax = pylab.gca()

    pylab.plot(x,y,'ok',label='Target initial pos')

    for i in range(num_agents):
        pylab.plot(agents[i].position[0],agents[i].position[1],'*',color = color[i], label='Observer '+str(i))
        # for j in range(4):
        #     pylab.plot(agents[i].fov[2*j],agents[i].fov[2*j+1],'o',color = color[j])
        ax.add_patch(Rectangle((agents[i].fov[0],agents[i].fov[1]),fov[0],fov[1],angle=agents[i].orientation,fill=None,linewidth=1,color=color[i]))

    pylab.legend()
    pylab.xlim(-1,grid_x+1)
    pylab.ylim(-1,grid_y+1)
    #pylab.savefig('/home/naveed/Dropbox/Sem 3/Aero 626/Project/'+'Env.pdf', format='pdf',bbox_inches='tight',pad_inches = .06)
    #pylab.show()

# function to generate random probablities for comminucation from guassian
def rndm_ntwrk(num_agents,num_samples,num_channels,comm_prob_ref):

    comm_prob_vec = np.zeros(num_channels) # intiate an empty array for storing one instance of network connectivity
    comm_prob_vec_vec = np.zeros((num_samples+1,num_channels)) # create an empty vector to store the communication probablity vectors for all the samples
    # syntax of comm_prob_vec_vec: [[11,12,13,22,23,33],[..],[..]..num_samples]] (value = probablity of communication exsitance for correponsding channel)
    network_vec = np.zeros([num_samples+1,num_channels]) # initiate a vector to store the configurations based on probablities generated
    network_vec[0,:] = np.ones([1,num_channels]) # the fisrt row corresponding to condition that all the channels are actively communicating
    comm_prob_vec_vec[0,:] = np.ones([1,num_channels]) # the fisrt row corresponding to condition that all the channels are actively communicating
    # generate a random vector and select sample its cdf for probablity of connection at each instance
    rv_num_samples = 10000 # define the number of samples of RV's to be extracted.
    x = np.random.randn(rv_num_samples) # generate a random vector
    norm_cdf_x = scipy.stats.norm.cdf(x) # calculate the cdf


    for t in range(1,num_samples+1):

        # Define connectivty probablity for the current sampling instance
        for j in range(num_channels):

            idx = int(np.random.uniform(0,rv_num_samples,1)) # pick a random index for elements in the random vector
            comm_prob_vec[j] = norm_cdf_x[idx] # this is the probability of connection based on the previous distribution

        # Define the connectivity based on probablity thresholding

        for j in range(num_channels):

            if comm_prob_vec[j] >= comm_prob_ref :

                network_vec[t,j] = 1 # assign 1 to the element corresponding to the channel whose connection probablity is greater than equal to defined referenc

        comm_prob_vec_vec[t,:] = comm_prob_vec # store the current network connectivty probablities
        comm_prob_vec = np.zeros(num_channels) # reset the array for next loop



    return network_vec # return the communication matrix

def create_adjacency_mat(network_vec, num_agents):

    adj_mat = np.eye(num_agents)

    channel_no = 0
    for i in range(num_agents):
        for j in range(i + 1, num_agents):

            adj_mat[i,j] = network_vec[channel_no]
            adj_mat[j,i] = network_vec[channel_no]
            channel_no = channel_no + 1

    return adj_mat

def plot_states(X_est,X_act,P):

    t = np.linspace(0,t_step,t_step+1)
    fig, ax = pylab.subplots(1,2)
    ax[0].plot(t,X_act[0,:],'bo',markersize=5,linewidth=2,label='Actual x1')
    ax[0].errorbar(t,X_est[0,:],yerr=3*np.sqrt(P[0,:]),fmt='r*',markersize=5,linewidth=2,label='Estimated x1')
    ax[0].legend()
    ax[0].set_xlim(-1,t_step+1)
    ax[1].plot(t,X_act[1,:],'bo',markersize=5,linewidth=2,label='Actual x2')
    ax[1].errorbar(t,X_est[1,:],yerr=3*np.sqrt(P[3,:]),fmt='r*',markersize=5,linewidth=2,label='Estimated x2')
    ax[1].legend()
    ax[1].set_xlim(-1,t_step+1)
    #
    #pylab.savefig('/home/naveed/Dropbox/Sem 3/Aero 626/Project/'+'Central_EIF_connec_wfov.pdf', format='pdf',bbox_inches='tight',pad_inches = .06)
    # pylab.figure(2)
    #
    # pylab.plot(np.zeros((6,1)),X_act[0,:],'bo',markersize=5,linewidth=2,label='Actual x1')
    # pylab.errorbar(np.zeros((6,1)),X_est[0,:],yerr=3*np.sqrt(P[0,:]),fmt='r*',markersize=5,linewidth=2,label='Estimated x1')
    # pylab.savefig('/home/naveed/Dropbox/Sem 3/Aero 626/HW3/'+'1_1_PF_x1.pdf', format='pdf',bbox_inches='tight',pad_inches = .06)
    # pylab.legend()

    pylab.show()

def dist_CI(target,agents,adj_mat,X_prev_act,t):

    global FLAG_CONVERGENCE_ICI

    X_act = rs(target.state_propagate(X_prev_act,target.Sigma_w),(target.n,))

    #finding partials
    for i in range(len(agents)):

        agents[i].X_est[:,t+1],O_new = filter.partial_EIF(target,agents[i],agents[i].X_est[:,t], X_act, agents[i].O[:,t])
        agents[i].O[:,t+1] = O_new.reshape((target.n**2,))
        agents[i].P[:,t+1] = O_new.I.reshape((target.n**2,))

    #do CI
    l = 0
    while not FLAG_CONVERGENCE_ICI:
        for i in range(len(agents)):

            agents[i].X_est[:,t+1],O_new= filter.CI(agents,adj_mat,i,t+1)
            agents[i].O[:,t+1] = O_new.reshape((target.n**2,))
            agents[i].P[:,t+1] = O_new.I.reshape((target.n**2,))

        if l>max_iter_conv:
            FLAG_CONVERGENCE_ICI = True

        else:
            l = l+1

    FLAG_CONVERGENCE_ICI = False

    return X_act

def dist_hybrid(target,agents,adj_mat,X_prev_act,t):
    global FLAG_CONVERGENCE

    X_act = rs(target.state_propagate(X_prev_act,target.Sigma_w),(target.n,))

    #finding priors
    for i in range(len(agents)):

        agents[i].X_est[:,t+1],O_new,agents[i].del_i[:,t+1],agents[i].del_I[:,t+1] = filter.priors_meas(target,agents[i],agents[i].X_est[:,t], X_act, agents[i].O[:,t])
        agents[i].O[:,t+1] = O_new.reshape((target.n**2,))
        agents[i].P[:,t+1] = O_new.I.reshape((target.n**2,))


    l = 0
    while not FLAG_CONVERGENCE:
        for i in range(len(agents)):

            #do CI on priors
            agents[i].X_est[:,t+1],O_new = filter.CI(agents,adj_mat,i,t+1)
            agents[i].O[:,t+1] = O_new.reshape((target.n**2,))
            agents[i].P[:,t+1] = O_new.I.reshape((target.n**2,))

            #consensus on likelihoods
            agents[i].del_i[:,t+1],agents[i].del_I[:,t+1] = filter.consensus(agents,adj_mat,i,t+1)


        if l>max_iter_conv:
            FLAG_CONVERGENCE = True

        else:
            l = l+1

    for i in range(len(agents)):
        #hybrid update
        S_post = np.matmul(mat(agents[i].O[:,t+1].reshape((target.n,target.n))).astype(np.float64),
                mat(agents[i].X_est[:,t+1].reshape((target.n,1))).astype(np.float64)) + \
                 np.sum(adj_mat[i,:])*agents[i].del_i[:,t+1].reshape((target.n,1))

        O_post = mat(agents[i].O[:,t+1].reshape((target.n,target.n))).astype(np.float64) + \
                    np.sum(adj_mat[i,:])*mat(agents[i].del_I[:,t+1].reshape((target.n,target.n))).astype(np.float64)

        agents[i].X_est[:,t+1] = np.matmul(O_post.I,S_post).reshape((target.n,))
        agents[i].O[:,t+1] = O_post.reshape((target.n**2,))
        agents[i].P[:,t+1] = O_post.I.reshape((target.n**2,))

    FLAG_CONVERGENCE = False

    return X_act

def plot_nees(err_vec):

    t = np.linspace(1,31,30)
    lb = 1.2953*np.ones(t_step)
    ub = .7423*np.ones(t_step)
    pylab.plot(t,err_vec,'.-',markersize=5,linewidth=2,color='b')
    pylab.plot(t,lb)
    pylab.plot(t,ub)
    pylab.xlabel('time')
    pylab.ylabel('Average NEES over 50 runs')
    pylab.savefig('/home/naveed/Dropbox/Sem 3/Aero 626/Project/'+'agent1_nees.pdf', format='pdf',bbox_inches='tight',pad_inches = .06)
    #pylab.show()


if __name__ == "__main__":

    np.random.seed(1)
    comm_status_vec = np.ones(num_agents) # 1 for on and 0 for off. Initial configuration is set default to all connected
    agents = [] # intiating a list of agents (objects for class defined in the framework)
    num_channels =  int(scipy.misc.comb(num_agents,2)) # since an agent is connected to all agents in fully connected condition (nc2)

    #agents object created
    target = pb.target_system()

    for i in range(num_agents):
        agents.append(pb.agent(position_vec[i,:],orientation_vec[i],comm_status_vec[i],num_channels,grid_x,grid_y,fov,t_step,target.n))

    #target object creation
    target = pb.target_system()


    #for i in range(num_agents):
    #    print("i:",i, " status:",agents[i].check_fov(x,y))



    X_est = np.zeros((target.X0.shape[0],t_step+1))
    X_act = np.zeros((target.X0.shape[0],t_step+1))
    P = np.zeros((target.X0.shape[0]*target.X0.shape[0],t_step+1))
    O = np.zeros((target.X0.shape[0]*target.X0.shape[0],t_step+1))

    X_est[:,0] = np.reshape(target.X0,(target.X0.shape[0],))
    X_act[:,0] = np.reshape(target.X0,(target.X0.shape[0],))

    P[:,0] = target.P0.reshape((target.n**2,))
    O0 = target.P0.I
    #print "O0:",O0
    O[:,0] = O0.reshape((target.n**2,))

    #initialisation for all agents
    for i in range(num_agents):

        agents[i].X_est[:,0] = np.reshape(target.X0,(target.X0.shape[0],))
        O0 = target.P0.I
        agents[i].O[:,0] = O0.reshape((target.n**2,))
        agents[i].P[:,0] = target.P0.reshape((target.n**2,))

    error_vec = np.zeros((no_MC_iters,t_step))
    nees_err_vec = np.zeros(t_step)

    for iter in range(no_MC_iters):
        print "Iter #:", iter
        # generating random networks with disruptions in communication
        comm_net_samples = rndm_ntwrk(num_agents,t_step,num_channels,comm_prob_ref)
        #print comm_net_samples
        for t in range(t_step):

            # create an adjaceny matrix based on the network vector
            adj_mat = create_adjacency_mat(comm_net_samples[t,:], num_agents)
            #print "adj_mat:", adj_mat

            #call EIF
            # X_est[:,t+1], X_act[:,t+1], O_new = filter.central_EIF(target, agents, X_est[:,t],X_act[:,t],O[:,t])
            # O[:,t+1] = O_new.reshape((target.n**2,))
            # P[:,t+1] = O_new.I.reshape((target.n**2,))

            #CI
            #X_act[:,t+1] = dist_CI(target, agents, adj_mat,X_act[:,t],t)

            #Hybrid
            X_act[:,t+1] = dist_hybrid(target, agents, adj_mat,X_act[:,t],t)


        # for i in range(len(agents)):
        #     print "i",i,"X_est:", agents[i].X_est
        # print "X_act:", X_act

        #plot(num_agents,agents,grid_x,grid_y,fov,target.X0[0],target.X0[1])
        #calculate rmse
        """
        for t in range(1,t_step+1):

            err_sum = 0
            for i in range(len(agents)):

                #err_sum = err_sum + np.linalg.norm(X_act[:,t] - agents[i].X_est[:,t]) #2 norm of the error
                err_sum = err_sum + np.linalg.norm(X_act[:,t] - X_est[:,t]) #2 norm of the error
            error_vec[iter,t-1] = np.sqrt(err_sum/num_agents)

            file.write(str(error_vec[iter,t-1]) + ',')

        file.write('\n')
        """
        #NEES
        for t in range(1,t_step+1): #ignoring 1st step

            err = mat((X_act[:,t] - agents[0].X_est[:,t]).reshape((target.n,1))).astype(np.float64)
            O_t = mat(agents[0].O[:,t].reshape((target.n,target.n))).astype(np.float64)

            nees_err = np.matmul(np.matmul(err.T,O_t),err)
            print "nees_error:",nees_err
            nees_err_vec[t-1] = nees_err_vec[t-1] + nees_err
        print "Nees err vec:", nees_err_vec

    nees_err_vec = np.true_divide(nees_err_vec,target.n*no_MC_iters)
    for i in range(t_step):
        file.write(str(nees_err_vec[i]) + ',')
    file.write('\n')
    plot_nees(nees_err_vec)

    # for i in range(len(agents)):
    #     plot_states(agents[i].X_est,X_act,agents[i].P)
    if FILE_WRITE:
        file.close()
