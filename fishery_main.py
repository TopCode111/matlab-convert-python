import fishery
class grd_class:
    #def __init__(self,X0,Nx,Xn,Nu,Un):
        Xn = array('i',[1,2])
        Xn[0] = array('c',['lo','hi'])
        Xn[1] = array('lo', 'hi')
        XN = array(1, 2)
        XN[0] = array('lo', 'hi')
        XN[1] = array('lo', 'hi')
        Nx = array('c',['',''])
        Nu = array('i',[0,1])
        X0 = array('i',[0,1])
grd = grd_class()
grd.Nx[1] = 201
grd.Xn[1]['lo'] = 0
grd.Xn[1]['hi'] = 1000
grd.Nu[1] = 21
grd.Un[1]['lo'] = 0
grd.Un[1]['hi'] = 10

# set initial state
grd.X0[1] = 250

# set final state constraints
grd.XN[1].hi = 1000
grd.XN[1].lo = 750

# define problem
prb.Ts = 1/5
prb.N = 200*1/prb.Ts + 1

#set options
options = dpm()
options.BoundaryMethod = 'Line'
# also possible: 'none' or 'LevelSet'
if options.BoundaryMethod =='Line':
    options.FixedGrid = 1
    options.Iter = 10
    options.Tol = 10**(-1)
######################################
dyn = dpm(fishery(),'',grd, prb, options)
#[res dyn] = dpm(@fishery,[],grd,prb,options)
