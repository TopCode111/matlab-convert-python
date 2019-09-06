# *************************************************************************
# Example Problem From
# Implementation of Dynamic Programming for n-Dimensional Optimal Control
# Problems with Final State Constraints
# Philipp Elbert, Soren Ebbesen, Lino Guzzella
#IEEE Transactions on Control Systems Technology
# DOI: 10.1109/TCST.2012.219035
import matplotlib.pyplot as plt
import numpy as np
import math
import dpm
import fishery_main

model = 'twostates'
par = ''
prb.Ts = 0.01
prb.N  = 2/prb.Ts
prb.N0 = 1

#options = dpm()
Nx = 51
#% PREPARE DP:
#grd = grd_class()
#grd.X0[0] = 0
Xn[0][0] = 0
Xn[0][1] = 1
XN[0][0] = 0.5
XN[0][1] = 1
grd.Nx[0] = Nx

grd.X0[1] = 0
Xn[1][0] = 0
Xn[1][1] = 1
XN[1][0] = 0.5
XN[1][1] = 1
grd.Nx[2] = Nx

grd.Un[0].hi = 1
grd.Un[0].lo = 0
grd.Nu[0] = 21

grd.Un[1][1] = 1
grd.Un[1][0] = 0
grd.Nu[1] = 21

options.MyInf = (grd.Un[1][1]*ones(prb.N,1)*prb.Ts).sum()

#% BASIC DP:
options.BoundaryMethod = 'none'
dynb = dpm(model,par,grd,prb,options)
tmeb = toc
#--------------------------------
#% LEVEL SET DP:
#--------------------------------

options.BoundaryMethod = 'LevelSet'
x1 = linspace(grd.Xn[0][0],grd.Xn[0][1],grd.Nx[1])
x2 = linspace(grd.Xn[1][0],grd.Xn[1][1],grd.Nx[2])
[X1,X2] = ndgrid(x1,x2)
options.gN[1] = max(grd.XN[0][0]-X1,0) + max(grd.XN[1][0]-X2,0)

dyn = dpm(model,par,grd,prb,options)
tme = toc
#--------------------------------------------------------------------------
# PLOT RESULTS:
#--------------------------------------------------------------------------
#Calculate Optimal Solution:
t = rage(0,2,prb.Ts)
ton = 2+2*math.log(0.5)
xopt = max(0,1-math.exp(-0.5*(t-ton)))
u1 = t>=ton
u2 = 0.5*(t>=0)

#Plot State Trajectory
fgr = figure(1)
ax1 = subplot(211)
k=2
plt.plot(t,xopt,'Color',concatenate((0.8, 0.8, 0.8)),'LineWidth',2)

plt.plot(t,outb.X[1])
plt.plot(t,out.X[1],'--','Color',[0, 0.5, 0])
plt.plot([.35, 0.45],[0.3, 0.3])
text(.5,.3,'basic DP')
plt.plot([1.2, 1.3],[0.2, 0.2],'--','Color',[0, 0.5, 0])
text(1.35,.2,'level-set DP')
plt.plot([1.2, 1.3],[0.1, 0.1],'Color',[0.8, 0.8, 0.8],'LineWidth',2)
text(1.35,.1,'analytic solution')
ylabel('x_1, x_2 [-]')

#Plot Control Inputs
ax2 = subplot(212)
plt.plot(t,u2,'Color',[0.8, 0.8, 0.8],'LineWidth',2)

plt.plot(t[0:-1],outb.u2,'b')
plt.plot(t[0:-1],out.u2,'--','Color',[0 ,0.5, 0])
ylabel('$u_2$ [-]')
xlabel('time [s]')
plt.plot(t,u1,'Color',[0.8, 0.8 ,0.8],'LineWidth',2)
plt.plot(t[0:-1],outb.u1,'b')
plt.plot(t[0:-1],out.u1,'--','Color',[0, 0.5, 0])
ylabel('u_1, u_2 [-]')
plt.show()
