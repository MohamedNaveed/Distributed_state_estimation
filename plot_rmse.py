import math as m
import numpy as np
import pylab

filename = "/home/naveed/Documents/distribute_state_estimation/error_ICI.csv"
file1 = open(filename,"r")

filename = "/home/naveed/Documents/distribute_state_estimation/error_hyb.csv"
file2 = open(filename,"r")

filename = "/home/naveed/Documents/distribute_state_estimation/error_central.csv"
file3 = open(filename,"r")

t_step = 30
no_MC_iters = 50

error_vec_CI = np.zeros((no_MC_iters,t_step+1))
lines = file1.read().splitlines()

for i in range(len(lines)):
    data = lines[i].split(',')

    for t in range(t_step):
        error_vec_CI[i,t] = float(data[t])


error_vec_hyb = np.zeros((no_MC_iters,t_step+1))
lines = file2.read().splitlines()

for i in range(len(lines)):
    data = lines[i].split(',')
    for t in range(t_step):
        error_vec_hyb[i,t] = float(data[t])

error_vec_central = np.zeros((no_MC_iters,t_step+1))
lines = file3.read().splitlines()

for i in range(len(lines)):
    data = lines[i].split(',')
    for t in range(t_step):
        error_vec_central[i,t] = float(data[t])


#print "error hyb:", error_vec_hyb
mean_CI = np.zeros(t_step-1)
var_CI = np.zeros(t_step-1)
for t in range(1,t_step):
    mean_CI[t-1] = np.mean(error_vec_CI[:,t])
    var_CI[t-1] = np.var(error_vec_CI[:,t])

mean_hyb = np.zeros(t_step-1)
var_hyb = np.zeros(t_step-1)
for t in range(1,t_step):
    mean_hyb[t-1] = np.mean(error_vec_hyb[:,t])
    var_hyb[t-1] = np.var(error_vec_hyb[:,t])

mean_central = np.zeros(t_step-1)
var_central = np.zeros(t_step-1)
for t in range(1,t_step):
    mean_central[t-1] = np.mean(error_vec_central[:,t])
    var_central[t-1] = np.var(error_vec_central[:,t])

t_s = np.linspace(1,t_step,t_step-1)
pylab.plot(t_s,mean_hyb,'.-',markersize=5,linewidth=2,color='g',label='Hybrid')
pylab.plot(t_s,mean_CI,'.-',markersize=5,linewidth=2,color='m',label='ICI')
pylab.plot(t_s,mean_central,'.-',markersize=5,linewidth=2,color='b',label='Centralised')
pylab.legend()
pylab.xlabel('time')
pylab.ylabel('RMSE')
pylab.grid()

pylab.savefig('/home/naveed/Dropbox/Sem 3/Aero 626/Project/'+'RMSE_comp_3.pdf', format='pdf',bbox_inches='tight',pad_inches = .06)
