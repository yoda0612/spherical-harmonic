from scipy.special import lpmv
import scipy.io
import numpy as np
from scipy.special import sph_harm 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

mat = scipy.io.loadmat('a01_l_hippo_CALD_smo.mat')

sph_verts=mat['sph_verts']
vertices=mat['vertices']
vnum=sph_verts.shape[0]

phi=np.arctan2(sph_verts[:,1],sph_verts[:,0])
theta=np.arctan2(sph_verts[:,2],np.sqrt(sph_verts[:,0]**2+sph_verts[:,1]**2))  
phi[np.where(phi<0)]+=2*np.pi
theta=np.pi/2-theta



def spharm_basis(degree, theta, phi):
    z=np.array([])
    for d in range(degree+1):
        if d>0:
            for order in reversed(range(1,d+1)):
                Y_lm=sph_harm(-order,d,phi,theta)
                z=np.append(z,Y_lm)

        Y_lm=sph_harm(0,d,phi,theta)
        z=np.append(z,Y_lm)

        if d>0:
            for order in range(1,d+1):
                Y_lm=sph_harm(order,d,phi,theta)
                z=np.append(z,Y_lm)

    return z


degree=2
Z=np.array([])
for v in zip(theta,phi):
    z=spharm_basis(degree,v[0],v[1])    
    Z=np.append(Z,z)

Z=np.reshape(Z,(vnum,int(Z.shape[0]/vnum)))
fvec=np.zeros([(degree+1)**2,3],dtype=complex)

for vi in range(3):
    v=np.reshape(vertices[:,vi],(vnum,1))        
    x=np.linalg.lstsq(Z,v,rcond=None)
    fvec[:,vi]=x[0].T



vec=np.dot(Z,fvec).real
x=vec[:,0]
y=vec[:,1]
z=vec[:,2]


#x=vertices[:,0]
#y=vertices[:,1]
#z=vertices[:,2]




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, marker='*')

plt.show()