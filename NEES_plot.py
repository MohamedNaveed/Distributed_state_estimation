import math as m
import numpy as np
import pylab

params_pylab = {'axes.labelsize':18,
            'font.size':14,
            'legend.fontsize':16,
            'xtick.labelsize':14,
            'ytick.labelsize':14,
            'text.usetex':True,
            'figure.figsize':[6,4.5]}
pylab.rcParams.update(params_pylab)
"""
filename = "/home/naveed/Documents/distribute_state_estimation/NEES_Hyb1.csv"
file1 = open(filename,"r")

filename = "/home/naveed/Documents/distribute_state_estimation/NEES_Hyb2.csv"
file2 = open(filename,"r")

filename = "/home/naveed/Documents/distribute_state_estimation/NEES_Hyb3.csv"
file3 = open(filename,"r")

filename = "/home/naveed/Documents/distribute_state_estimation/NEES_Hyb4.csv"
file4 = open(filename,"r")
"""
filename = "/home/naveed/Documents/distribute_state_estimation/NEES_central.csv"
file1 = open(filename,"r")

filename = "/home/naveed/Documents/distribute_state_estimation/NEES_Hyb.csv"
file2 = open(filename,"r")

filename = "/home/naveed/Documents/distribute_state_estimation/NEES_ICI.csv"
file3 = open(filename,"r")

t_step = 30
"""
nees_vec_agent1 = np.zeros((t_step))
lines = file1.read().splitlines()

for i in range(len(lines)):
    data = lines[i].split(',')

    for t in range(t_step):
        nees_vec_agent1[t] = float(data[t])

nees_vec_agent2 = np.zeros((t_step))
lines = file2.read().splitlines()

for i in range(len(lines)):
    data = lines[i].split(',')

    for t in range(t_step):
        nees_vec_agent2[t] = float(data[t])

nees_vec_agent3 = np.zeros((t_step))
lines = file3.read().splitlines()

for i in range(len(lines)):
    data = lines[i].split(',')

    for t in range(t_step):
        nees_vec_agent3[t] = float(data[t])


nees_vec_agent4 = np.zeros((t_step))
lines = file4.read().splitlines()

for i in range(len(lines)):
    data = lines[i].split(',')

    for t in range(t_step):
        nees_vec_agent4[t] = float(data[t])

"""
nees_vec_Hyb = np.zeros((t_step))
lines = file1.read().splitlines()

for i in range(len(lines)):
    data = lines[i].split(',')

    for t in range(t_step):
        nees_vec_Hyb[t] = float(data[t])

nees_vec_central = np.zeros((t_step))
lines = file2.read().splitlines()

for i in range(len(lines)):
    data = lines[i].split(',')

    for t in range(t_step):
        nees_vec_central[t] = float(data[t])

nees_vec_ICI = np.zeros((t_step))
lines = file3.read().splitlines()

for i in range(len(lines)):
    data = lines[i].split(',')

    for t in range(t_step):
        nees_vec_ICI[t] = float(data[t])

pylab.figure(figsize = (16,16))
t = np.linspace(1,31,30)
lb = 1.2953*np.ones(t_step)
ub = .7423*np.ones(t_step)

pylab.plot(t,nees_vec_Hyb,'.-',markersize=5,linewidth=2,color='g',label='Hybrid')
pylab.plot(t,nees_vec_ICI,'.-',markersize=5,linewidth=2,color='m',label='ICI')
pylab.plot(t,nees_vec_central,'.-',markersize=5,linewidth=2,color='b',label='Centralised')

pylab.plot(t,lb)
pylab.plot(t,ub)
pylab.legend()
pylab.xlabel('time')
pylab.ylabel('Average NEES over 50 runs')

"""
pylab.subplot(221)
pylab.plot(t,nees_vec_agent1,'.-',markersize=5,linewidth=2,color='b')
pylab.plot(t,lb)
pylab.plot(t,ub)
pylab.legend()
pylab.xlabel('time')
pylab.ylabel('Average NEES over 50 runs: Agent I')

pylab.subplot(222)
pylab.plot(t,nees_vec_agent2,'.-',markersize=5,linewidth=2,color='b')
pylab.plot(t,lb)
pylab.plot(t,ub)
pylab.legend()
pylab.xlabel('time')
pylab.ylabel('Average NEES over 50 runs: Agent II')

pylab.subplot(223)
pylab.plot(t,nees_vec_agent3,'.-',markersize=5,linewidth=2,color='b')
pylab.plot(t,lb)
pylab.plot(t,ub)
pylab.legend()
pylab.xlabel('time')
pylab.ylabel('Average NEES over 50 runs: Agent III')

pylab.subplot(224)
pylab.plot(t,nees_vec_agent4,'.-',markersize=5,linewidth=2,color='b')
pylab.plot(t,lb)
pylab.plot(t,ub)
pylab.legend()
pylab.xlabel('time')
pylab.ylabel('Average NEES over 50 runs: Agent IV')
"""
#pylab.show()
pylab.savefig('/home/naveed/Dropbox/Sem 3/Aero 626/Project/'+'comp_nees.png', format='png',bbox_inches='tight',pad_inches = .06)
