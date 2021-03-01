import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
a=1
b=1
c=1
d=1
e=1
f=1
g=1
def model(r,t):
    x=r[0]
    y=r[1]
    z=r[2]
    dxdt= a*x-b*x*y
    dydt=-c*y+d*x*y-e*y*z
    dzdt=-f*z+g*y*z
    return[dxdt, dydt, dzdt]
z0=[10, 10, 10]
t= np.linspace(0,10,1000)
r= odeint(model, z0, t)
x = r[:,0]
y = r[:,1]
z= r[:,2]
plt.plot(t,x, label="Species X")
plt.plot(t,y, label="Species Y")
plt.plot(t,z, label="Species Z")
plt.xlabel('time')
plt.ylabel('Population size')
plt.legend()
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x,y,z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()