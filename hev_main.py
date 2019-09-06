# load driving cycl
# create grid
import hev
import dpm
class grd:
    Nx = [0,0]
    Xn = [[0,0],['hi','lo']]
    Nu = [0,0]
    X0 =[0,0]
    XN = [[0, 0], ['hi', 'lo']]
    Un = [[0, 0], ['hi', 'lo']]
class prb:
    W = [0,0,0]
    Ts = 1
    N = 0
grd.Nx[1]= 61
grd.Xn[1][0] = 0.7
grd.Xn[1][1] = 0.4

grd.Nu[1]= 21
grd.Un[1][0] = 1
grd.Un[1][1] = -1 	# Att: Lower bound may vary with engine size.

# set initial state
grd.X0[1] = 0.55

# final state constraints
grd.XN[1][0] = 0.56
grd.XN[1][0] = 0.55

# define problem

prb.W[0] = 1#speed_vector  # (661 elements)
prb.W[1] = 1#acceleration_vector  # (661 elements)
prb.W[2] = 1 # gearnumber_vector  # (661 elements)
prb.Ts = 1
prb.N = 660*1/prb.Ts + 1

# set options

options = dpm()
options.MyInf = 1000 
options.BoundaryMethod = 'Line'  # also possible: 'none' or 'LevelSet' 
if options.BoundaryMethod=='Line':
    #these options are only needed if 'Line' is used
    options.Iter = 5 
    options.Tol = 10**(-8)
    options.FixedGrid = 0
[res, dyn] = dpm(hev,[],grd,prb,options)
