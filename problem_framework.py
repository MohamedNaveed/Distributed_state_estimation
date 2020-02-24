import math as m
import numpy as np
import sympy as sp
import numpy.matlib
import pylab
from matplotlib.patches import Rectangle
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from numpy import reshape as rs
from numpy import matrix as mat


class agent:

    def __init__(self,pos,orient,status,num_channels,grid_x,grid_y,fov,t_step, dim_state):

        self.position = pos
        self.orientation = orient
        # define the field of view of the agents
        fov_temp = np.empty(8,dtype=np.float64)
        # define the rectangle points
        fov_temp[0:2] = [pos[0]+fov[1]*0.5*m.sin(m.radians(orient)) , pos[1]-fov[1]*0.5*m.cos(m.radians(orient))] #bl
        fov_temp[2:4] = [pos[0]-fov[1]*0.5*m.sin(m.radians(orient)) , pos[1]+fov[1]*0.5*m.cos(m.radians(orient))] #tl
        fov_temp[4:6] = [pos[0]-fov[1]*0.5*m.sin(m.radians(orient))+fov[0]*m.cos(m.radians(orient)) , pos[1]+fov[1]*0.5*m.cos(m.radians(orient))+fov[0]*m.sin(m.radians(orient))] #br
        fov_temp[6:8] = [pos[0]+fov[1]*0.5*m.sin(m.radians(orient))+fov[0]*m.cos(m.radians(orient)) , pos[1]-fov[1]*0.5*m.cos(m.radians(orient))+fov[0]*m.sin(m.radians(orient))] #tr
        # for i in range(len(fov_temp)):
        #     if fov_temp[i] > 0:
        #         if i % 2 != 0 :
        #             if fov_temp[i] > grid_x :
        #                 fov_temp[i] = grid_x
        #         else:
        #             if fov_temp[i] > grid_y :
        #                 fov_temp[i] = grid_y
        #     else:
        #         fov_temp[i] = 0;
        self.fov = fov_temp
        self.comm = status # assign the connectivity between agents as defined

        self.X_est = np.zeros((dim_state,t_step+1))
        self.del_i = np.zeros(((dim_state,t_step+1)))
        self.del_I = np.zeros((dim_state**2,t_step+1))
        self.O = np.zeros((dim_state**2,t_step+1))
        self.P = np.zeros((dim_state**2,t_step+1))


    def change_comm_status(self,ipt): # function to change the communication status of an agent
        self.comm = ipt

    def check_fov(self,x,y):

        point = Point(x,y)
        polygon = Polygon([(self.fov[0],self.fov[1]),(self.fov[2],self.fov[3]),(self.fov[4],self.fov[5]),(self.fov[6],self.fov[7])])

        return polygon.contains(point)

class target_system(object):

    t0 = 0
    n = 2
    X0 = np.array([[5.0],[5.0]])
    P0 = mat([[.5,0],[0,.5]])
    Sigma_w = 1.0
    Sigma_nu = .1
    h = 0.1
    tf = .5
    G = mat([[h,0],[0,h]])
    M = mat([[1.0,0],[0,1.0]])
    Q = mat([[Sigma_w,0],[0,Sigma_w]])
    R = mat([[Sigma_nu,0],[0,Sigma_nu]])

    def kinematics(self):

        x1, x2 = sp.symbols('x1 x2')

        F = sp.Matrix([x1 , x2])

        return F, x1, x2

    def jacobian(self,X):

        F, x1, x2 = self.kinematics()

        J = F.jacobian([x1,x2])

        return J.subs([(x1,X[0]),(x2,X[1])])

    def state_propagate(self,X0, Sigma_w):

        F, x1, x2 = self.kinematics()

        w1 = np.random.normal(0,np.sqrt(Sigma_w))
        w2 = np.random.normal(0,np.sqrt(Sigma_w))
        w = mat([[w1],[w2]])

        X = rs(F.subs([(x1,X0[0]),(x2,X0[1])]) + np.matmul(self.G,w),(self.n,))

        return X

def main():

    num_agents = 1 # defined the number of observers
    position_vec = np.array([[4,6],[5,4]])
    orientation_vec = np.array([45]) #in degrees
    comm_status_vec = np.array([1])

    agents = [] # intiating a list of agents

    for i in range(num_agents):
        agents.append(agent(position_vec[i,:],orientation_vec[i],comm_status_vec[i]))
    color_temp = np.array(['red','blue','green','yellow'])
    color =  color_temp # assigned by default, size adjustments are performed in next line

    if num_agents > len(color_temp):
        pad_size = num_agents - len(color_temp)
        color = np.pad(color_temp,(0,pad_size),'wrap')

    pylab.figure(1)
    ax = pylab.gca()
# foor debugging. doesn't work for more than 1 agent
    for i in range(4):

        print(color[i])
        #pylab.plot(agents[i].position[0],agents[i].position[1],'*',color = color[i])
        pylab.plot(agents[0].fov[i*2],agents[0].fov[i*2+1],'*',color = color[i])
        ax.add_patch(Rectangle((agents[0].fov[0],agents[0].fov[1]),fov[0],fov[1],angle=agents[0].orientation,fill=None,linewidth=1,color='b'))

    print("fov:", agents[0].fov)
    pylab.xlim(0,grid_x)
    pylab.ylim(0,grid_y)
    pylab.show()

if __name__ == "__main__":
    main()
