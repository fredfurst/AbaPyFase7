import nolds
import numpy as np
# Plot the solution that was generated
import matplotlib.pyplot as plt
# Use ODEINT to solve the differential equations defined by the vector field
from scipy.integrate import odeint, solve_ivp
# ks entropy
import EntropyHub as EH
import vapeplot
# %matplotlib inline
# vapeplot.available()
vapeplot.set_palette('macplus')

forca_inicial = 0.0
forca_final = -249.97 # -1.0 # F0 = 249.97
tempo_inicial = 5.0
tempo_final = 10.0

n=6 # n > 2 seria a condicao de teste

# Parameter values
# Masses:
# m1 = 2.0
m = 45.94/n # 2.0/n # m0 = 8.4 gramas

# Spring constants
# k1 = 5.0
k = 20970.7*n # 2.5*n 
# d = - 0.01192 mm
# F = 249.97 N
# k ~ 20970.7 N/mm

# Natural lengths
# L1 = 1.0
L = 200.0/n # 4.0/n
# L0 = 200.0 mm

# Friction coefficients
# b1 = 0.6
b = 10.6

# Initial conditions
# x1 and x2 are the initial displacements; y1 and y2 are the initial velocities
xizesys = [ (n+2)*L/2 if n%2==0 else 0.0 for n in range(2*n) ]
xizesys_desl = xizesys[:-2]

# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 25.0
numpoints = 10000

# Create the time samples for the output of the ODE solver.
# I use a large number of points, only because I want to make
# a plot of the solution that looks nice.
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

# degrau de forca
def degrau_forca(t):
    if t <= 5.0:
        forca = forca_inicial
    else:
        forca = forca_final
    return forca

def degrau_forca_vectorfield(t, w, p):
    """
    Defines the differential equations for the coupled spring-mass system.

    Arguments:
        w :  vector of the state variables:
                  w = [x1,y1,x2,y2,x3,y3]
        t :  time
        p :  vector of the parameters:
                  p = [m1,m2,m3,k1,k2,k3,L1,L2,L3,b1,b2,b3]
    """
#    x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, \
#      x10, y10, x11, y11, x12, y12, x13, y13, x14, y14, x15, y15, x16, y16 = w
    # print(t)
    # print(w)
    # print(p)
    m, k, L, b = p

    # Create f = (x1',y1',x2',y2' ...):
    f = [ ]
    f.append( w[1] )
    if n == 1:
        f.append( (-b * w[1] - k * (w[0] - L) + degrau_forca(t))/ m )
    elif n >= 2:
        f.append( (-b * w[1] + b * (w[3] - w[1]) - k * (w[0] - L) + k * (w[2] - w[0] - L ) )/ m )
                # (-b * y1   + b * (y2   - y1)   - k * (x1   - L) + k * (x2     - x1    - L) ) / m,
        for i in range(2,2*n-2):
            if i % 2 == 0:
                f.append( w[i+1] )
            else:
                f.append( (-b * (w[i]-w[i-2]) + b * (w[i+2] - w[i]) - k * (w[i-1] - w[i-3] - L) + k * (w[i+1] - w[i-1] - L) ) / m )
    # item 2*n-2 = 2*16-2 = 32 - 2 = 30
        f.append( w[2*n-1] )
        f.append( (-b * (w[2*n-1] - w[2*n-3])  - k * (w[2*n-2] - w[2*n-4] - L) + degrau_forca(t)) / m )
    # y16,
    # (-b * (y16 - y15)  - k * (x16 - x15 - L) + degrau_forca(t)) / m,
    return f

# Pack up the parameters and initial conditions:
p = [m, k, L, b]
w0 = xizesys

# Call the ODE solver.
# wsol = odeint(degrau_forca_vectorfield, w0, t, args=(p,),
#               atol=abserr, rtol=relerr)
wsol = solve_ivp(degrau_forca_vectorfield, (0.0, 25.0), w0, args=(p, ),
                 t_eval=t, atol=abserr, rtol=relerr)

at, ax, ay, = [],[[] for i in range(n)],[[] for i in range(n)],
# adendo
IE, KE = [[] for i in range(n)],[[] for i in range(n)],
ET = []
y = [list(i) for i in zip(*wsol.y)]
for t1, w1 in zip(wsol.t, y):
    # print( t1, w1[0], w1[1], w1[2], w1[3])
    at.append(t1)
    for i in range(n):
        ax[i].append(w1[2*i])
        ay[i].append(w1[2*i+1])
        if i==0:
            IE[i].append(k*(w1[2*i]-L)**2/2)
        else:
            IE[i].append(k*(w1[2*i]-w1[2*i-2]-L)**2/2)
        KE[i].append(m*w1[2*i+1]**2/2)
    ET.append(np.sum([k*(w1[2*i]-L)**2/2+m*w1[2*i+1]**2/2 if i == 0 else k*(w1[2*i]-w1[2*i-2]-L)**2/2+m*w1[2*i+1]**2/2 for i in range(n) ]))

plt.figure(figsize=(8,8))
F = [[] for i in range(n)]
for i in range(n):
    if i == 0:
        F[i] = -k*np.array(ax[i]) + np.array([k*L for i in range(len(ax[i]))])
    else:
        F[i] = -k*np.array(ax[i]) +k*np.array(ax[i-1]) + np.array([k*L for i in range(len(ax[i]))])
    plt.plot(at, F[i], linewidth=1)
plt.plot(at, -np.array([ degrau_forca(t) for t in at ]), linewidth=1)

plt.grid(visible=True)
plt.xlabel('time (s)')
plt.ylabel('force (N)')
plt.legend([r'$F_{'+str(i)+r'}$' for i in range(n) ],loc='lower right')
plt.title('Reaction and Applied Forces for '+str(n)+'-Discrete Spring-Mass System degrau forca')
plt.show()

plt.figure(figsize=(8,8))
F = [[] for i in range(n)]
for i in range(n):
    if i == 0:
        F[i] = -k*np.array(ax[i]) + np.array([k*L for i in range(len(ax[i]))])
    else:
        F[i] = -k*np.array(ax[i]) +k*np.array(ax[i-1]) + np.array([k*L for i in range(len(ax[i]))])
    plt.plot(at, F[i], linewidth=1)
plt.plot(at, -np.array([ degrau_forca(t) for t in at ]), linewidth=1)
axis = plt.gca()
axis.set_xlim(4.9, 5.4)
plt.grid(visible=True)
plt.xlabel('time (s)')
plt.ylabel('force (N)')
plt.legend([r'$F_{'+str(i)+r'}$' for i in range(n) ],loc='lower right')
plt.title('Reaction and Applied Forces for '+str(n)+'-Discrete Spring-Mass System degrau forca')
plt.show()

plt.figure(figsize=(8,8))
F = [[] for i in range(n)]
for i in range(n):
    if i == 0:
        F[i] = -k*np.array(ax[i]) + np.array([k*L for i in range(len(ax[i]))])
    else:
        F[i] = -k*np.array(ax[i]) +k*np.array(ax[i-1]) + np.array([k*L for i in range(len(ax[i]))])
    plt.plot(at, F[i], linewidth=1)
plt.plot(at, -np.array([ degrau_forca(t) for t in at ]), linewidth=1)
axis = plt.gca()
axis.set_xlim([24.5, 25])
plt.grid(visible=True)
plt.xlabel('time (s)')
plt.ylabel('force (N)')
plt.legend([r'$F_{'+str(i)+r'}$' for i in range(n) ],loc='lower right')
plt.title('Reaction and Applied Forces for '+str(n)+'-Discrete Spring-Mass System degrau forca')
plt.show()

###
##
#
# Phase space
#
##
###

r1, r2, r3 = 0.01, 0.02, 0.03
p=1

def ffplot2D_LE11_ER11(x3pre,y1pre):
    u = x3pre*1e3
    v = y1pre*np.sqrt(ms)*1e3
    plt.figure(figsize=(10,8))
    plt.grid(visible=True)
    plt.xlabel('Strain minus normalized ramp for '+prepos+' ($\perthousand$)')
    plt.ylabel('Strain rate ($\perthousand.s^{-1}$)')
    plt.plot(u,v,)#'.')
    assert len(u) == len(v)
    n = len(u)
    distances = []
    delta_u = (max(u)-min(u))/2
    print('delta_u: '+str(delta_u))
    delta_v = (max(v)-min(v))/2
    print('delta_v: '+str(delta_v))
    for k in range(n-p):
        distances.append(np.sqrt(((u[k+p]-u[k])/delta_u)**2+((v[k+p]-v[k])/delta_v)**2))
    u1,u2,u3, v1,v2, v3 = [],[],[],[],[],[]
    for k in range(len(distances)):
        if distances[k] < r1:
            u1.append(u[k])
            v1.append(v[k])
        if distances[k] < r2:
            u2.append(u[k])
            v2.append(v[k])
        if distances[k] < r3:
            u3.append(u[k])
            v3.append(v[k])
    plt.title('1. Trajectories for strain rate x strain minus $v='+"{:.2f}".format(ramp)+\
              '$ mm/s ramp\nat '+nalongitude+', and '+nalargura+' \nwith periods $p='+str(p)+\
              '$ and $r$ the distance between subsequent points in '+prepos)
    plt.plot(u3,v3,'.')
    plt.plot(u2,v2,'.')
    plt.plot(u1,v1,'.')
    plt.legend((r'$'+str(r3)+' \leq r$',r'$'+str(r2)+'\leq r<'+str(r3)+\
                '$',r'$'+str(r1)+' \leq r<'+str(r2)+'$',r'$0.0 \leq r<'+str(r1)+'$'))
    plt.savefig('LE11_ER11_'+nalongitude+'_'+nalargura+'_'+prepos+'_'+str(ig)+'_'+str(ms)+'.png')
    plt.show()