import math
import datetime
import operator
import sys
from operator import truediv
from array import *


class options:
    def __init__(self, Warnings, CalcLine, InfCost, MyInf, BoundaryMethod, BoundaryLineMethod, HideWaitbar, Waitbar,
                 UseLine, Tol, Iter, FixedGrid, SaveMap, UseUmap, UseLevelSet):
        self.Warnings = Warnings
        self.CalcLine = CalcLine
        self.InfCost = InfCost
        self.MyInf = MyInf
        self.BoundaryMethod = BoundaryMethod
        self.BoundaryLineMethod = BoundaryLineMethod
        self.HideWaitbar = HideWaitbar
        self.Waitbar = Waitbar
        self.UseLine = UseLine


# typecode='c'
# Initializers=['dd','daf','fs','df']
# my_array = array(typecode,[Initializers])
class grd:
    Nx = 0
    Xn = 0
    Nu = 0
    Un = 0

nargout = 1
varargout = []
def dpm(*args):
    nargin = len(args)
    naragin = args
    try:

        if nargin == 3:
            file_obj = open(varargin[0]+'.py','r')
            if ~file_obj:
                str[1] = ''
                str[0]=['function [X, C, I, signals] = '+varargin[0]+'(inp,par)']
                str[0]='\n%% inp.X[i] states'

                to_len = my_array.count()
                ##my_array[to_len+1] = "''"
                for i in range(1,varargin[1]):
                    x = round(rand * (varargin[1]-1)) + 1
                    u = round(rand * (varargin[2]-1)) + 1
                    str[0]=['X[' +str(i)+ '] = '+ str(rand) +'*(inp.X[' +str(x) +'] + inp.U[' +str(u) +'] + inp.W[1])/inp.Ts + inp.X[' +str(i)+ ']']
                    ##my_array[to_len+1] = "''"
                    print("my_array[to_len+1] = ")
                str[0] = '\n%% cost (out.C[1] must be set within model function)'
                str[0] = 'C[1] = -inp.Ts.*inp.U[1];'
                str[0] = '\n%% Infeasibility (out.I [zero=feasible/one=infeasible] must be set within model function)'
                str[0] = '\n%% for example if the state is outside the grid or infeasible combinations of inputs and states occur.'
                str[0] = '\n%% The cost of these state-input combinations will be set to options.MyInf by the DPM function.'
                str[0] = 'I = 0;'
                str[0] = '\n%% store signals (store any other signals in the out struct)'
                for i in range(1,varargin[2]):
                    str[0] = ['signals.U['+ str(i) +'] = inp.U{'+ str(i)+ ']']
                fid = open(varargin[0]+'.py','w')
                for i in range(1,len(str)):
                    print(fid,[str[i]+'\n'])
                fid.close()
            else:
                print('DPM:Internal', 'Filename exist in the current directory')
        elif nargin == 4:
            model = varargin[0]
            par = varargin[1]
            grd = varargin[2]
            dis = varargin[3]
            inp = dpm_get_empty_inp(grd, dis, 'zero')
            varargin[0] = dpm_get_empty_out(model, inp, par, grd, 'zero')
        elif nargin == '2':
            print("DPM:Internal")
        elif nargin == 0:
            Waitbar = 'off'
            Verbose = 'on'
            Warnings = 'on'
            SaveMap = 'on'
            MyInf = 10000
            Minimize = 1
            BoundaryMethod = 'none'
            #varargout[0] = options
            print("Waitbar: " + Waitbar)
            print("Verbose: " + Verbose)
            print("Warnings: " + Warnings)
            print("SaveMap: " + SaveMap)
            print("MyInf: " + str(MyInf))
            print("Minimize: " + str(Minimize))
            print("BoundaryMethod: " + BoundaryMethod)
        elif nargin == 5:
            RunForward = nargout > 1
            RunBackward = 1
            model = varargin[0]
            par = varargin[1]
            grd = varargin[2]
            dis = varargin[3]
            options = varargin[4]
        elif nargin == 6:
            RunForward = 1
            RunBackward = 0
            dyn = varargin[0]
            model = varargin[1]
            par = varargin[2]
            grd = varargin[3]
            dis = varargin[4]
            options = varargin[5]
        elif nargin == 7:
            RunForward = 1
            RunBackward = 0
            t0 = varargin[0]
            dyn = varargin[1]
            model = varargin[2]
            par = varargin[3]
            grd = varargin[5]
            dis = varargin[5]
            options = varargin[6]
        else:
            print("DPM:Internal")
        if nargin == 4 | nargin == 5 | nargin == 6 | nargin == 7:
            # if !isfield(dis, 'W'):
            #  dis.W = []
            grd = input_check_grd(grd, dis.N)
        if options != NULL:
            if options.warnings and options.warnings is 'on':
                print('on:DPM:Backward')
                print('on:DPM:Forward')
                print('on:DPM:General')
            if ~options.CalcLine:
                if options.InfCost:
                    print('DPM:Internal')
                    options.MyInf = options.InfCost
                    del(options.InfCost)
                if options.BoundaryLineMethod:
                    print('DPM:Internal')
                    options.BoundaryMethod = options.BoundaryLineMethod
                    del(options.BoundaryLineMethod)
                if options.HideWaitbar:
                    print('DPM:Internal')
                    if options.HideWaitbar:
                        options.Waitbar = 'off'
                    else:
                        options.Waitbar= 'on'
                    del(options.HideWaitbar)
                if options.UseLine:
                    print('DPM:Internal The option ''UseLine'' has been renamed to ')
                    if options.UseLine:
                        options.BoundaryMethod = 'Line'
                    else:
                        options.BoundaryMethod = 'none'
                    del(options.UseLine)
                ###############################
                onames = array('c', ['Waitbar', 'Verbose', 'Warnings', 'SaveMap', 'MyInf', 'Minimize', 'BoundaryMethod',
                                     'Iter', 'Tol', 'FixedGrid''gN', 'InputType', 'CalcLine', 'UseUmap', 'UseLine',
                                     'UseLevelSet'])
                ii = 0
                for i in range(1,len(options)):
                    ok = 0
                    jj = 0
                    for i in range(1, len(options)):
                         ok = ok | options[ii] is onames[jj]
                    if ~ok:
                        print('Unknown option: ' + options[ii])
                if ~options.SaveMap:
                    options.SaveMap = 0
                else:
                    # Backwards compatibility
                    if ~options.SaveMap:
                        if options.SaveMap == 'on':
                            options.SaveMap = 1
                        elif options.SaveMap == 'off':
                            options.SaveMap = 0
                        else:
                            print('Unable to interpret the option "SaveMap". Using SaveMap = ''on''')
                            options.SaveMap = 1

                # interpret user input regarding boundary method
                if ~options.BoundaryMethod:
                    options.BoundaryMethod = ''
                if options.BoundaryMethod == 'none':
                    options.UseLine = 0
                    options.UseLevelSet = 0
                elif options.BoundaryMethod == 'Line':
                    options.UseLine = 1
                    options.UseLevelSet = 0
                    options.UseUmap = 1
                elif options.BoundaryMethod == 'LevelSet':
                    options.UseLine = 1
                    options.UseLevelSet = 0
                    options.UseUmap = 1
                else:
                    print('Unable to interpret the option "SaveMap". Using SaveMap = ''on''')
                    options.BoundaryMethod = 'none'
                    options.UseLine = 0
                    options.UseLevelSet = 0
                if ~options.UseUmap:
                    options.UseUmap = 1
                # Boundary Line Method
                if options.UseUmap:
                    if len(grd.Nx) > 1:
                        print('Boundary-Line method works only with one-dimensional systems. Consider using the Level-Set method.')
                    if ~options.FixedGrid:
                        print('Model inversion requires the specification of a tolerance. Using options.Tol=10**(-8)')
                        options.FixedGrid = 1
                    if ~options.Tol:
                        print('Model inversion requires the specification of a tolerance. Using options.Tol=10**(-8)')
                        options.Tol = 1 / 10 ^ 8
                    if ~options.Iter:
                        print('Model inversion requires the specification of a tolerance. Using options.Tol=10**(-8)')
                        options.Iter = 10
                # Level Set Method:
                if options.UseLevelSet:
                    if len(grd.Nx) == 1:
                        print('For one-dimensional systems, consider using the Boundary-Line method')
                    if ~options.SaveMap:
                        print('Level-Set method needs cost-to-go for forward simulation. Setting SaveMap = ''on''')
                        options.SaveMap = 1
                    options.UseUmap = 0
        # DYNAMIC  PROGRAMMING
        # _____________________________________________________
        # Returns the optimal cost-to-go dyn.Jo with the
        # size of T x len(X1)
        if RunBackward:
            h = 0
            for i in rang(0, len(grd.Xn)):
                for j in rang(0, len(grd.Xn[i])):
                    if grd.Xn[i].lo > grd.Xn[i].hi:
                        if len(grd.Xn[i]) > 1:
                            print('Upper state boundary for state' + int(j) + 'is at instance' + int(j) + ' smaller than the lower state boundary.')
                        else:
                            print('Upper state boundary for state' + int(j) + 'is  smaller than the lower state boundary.')
            if ~options.CalcLine:
                if options.index('FixedGrid') and ~options.FixedGrid and min(grd.Xn[1].hi - grd.Xn[1].lo) / eps < \
                        grd.Nx[1]:
                    print(
                        'Final state constraints are too tight to include grd.Nx[1](end) points.\n\t Widen the final constraints.')
            dyn = dpm_backward(model, par, grd, dis, options)
        # EVAULATE RESULT
        # Uses the optimal input matrix dyn.Uo to simulate
        if RunForward:
            out = dpm_forward(dyn, model, par, grd, dis, options)
        else:
            clear_waitbars()
            #err = lasterror
            print('DPM:Forward simulation error \n \t Make sure the problem is feasible.\n')
            inp = dpm_get_empty_inp(grd, dis, 'zero')
            out = dpm_get_empty_out(model, inp, par, grd, 'nan')
        varargout[len(varargout) + 1] = out
        varargout[len(varargout) + 1] = dyn
        Warning('off' + 'DPM:Backward')
        Warning('off' + 'DPM:General')
        Warning('off' + 'DPM:Forward')
    except:
        #err = lasterror
        #notify_user_of_error(err)

        for i in range(1, nargout):
            varargout[i] = ''

dpm()
def dpm_backward(model, par, grd, dis, options):
    x_sze = array('i', len(grd.Nx))
    for i in range(0, len(grd.Nx)):
        x_sze[i] = grd.Nx[i][-1]
    u_sze = array('i', len(grd.Nu))
    for i in range(0, len(grd.Nu)):
        u_sze[i] = grd.Nu[i][-1]
    if options.UseLine:
        dyn.B.lo = dpm_boundary_line_lower(model, par, grd, dis, options)
        dyn.B.hi = dpm_boundary_line_upper(model, par, grd, dis, options)
    for i in range(len(grd.Nx)):
        if options.UseLine and i == 1 and ~options.FixedGrid:
            if len(grd.Nx) == 1:
                current_grd.X[i] = linspace(dyn.B.lo.Xo[-1], dyn.B.hi.Xo[-1], grd.Nx[i][-1])
            else:
                current_grd.X[i] = linspace(min(dyn.B.lo.Xo[i]), max(dyn.B.hi.Xo[i]), grd.Nx[i][i])
        else:
            current_grd.X[i] = array('i', [grd.Xn[i].lo[-1], grd.Xn[i].hi[-1], grd.Nx[i][-1]])
            if options.CalcLine:
                grd.Xn[1].lo = grd.Xn_true[1].lo
                grd.Xn[1].hi = grd.Xn_true[1].hi
    # generate current input grid
    for i in range(0, len(grd.Nu)):
        current_grd.U[i] = array('i', [grd.Un[i].lo[-1], grd.Un[i].hi[-1], grd.Nu[i][-1]])
    inp0 = dpm_get_empty_inp(grd, dis, 'zero')
    out0 = dpm_empty_out(model, inp0, par, grd, 'zero')
    ## Initialize the outputs dyn.Uo and dyn.Jo
    for i in range(0, len(out0.C)):
        if options.index('gN') and len(options.gN) >= i:
            if dpm_sizecmp(options.gN[i], zeros(sys.getsizeof(current_grd))):
                dyn.Jo[i] = options.gN[i]
            else:
                print('options.gN[' + str(i) + '] has incorrect dimesions')
        else:
            dyn.Jo[i] = sys.getsizeof(current_grd)
    ## if not using boundary line set cost of states outside feasible region
    if ~options.UseLine and ~options.CalcLine and ~options.UseLevelSet:
        for i in rang(1, grd.Nx.size()):
            eval(['dyn.Jo[1](' + repmat(':,', 1, i - 1) + 'current_grd.X[i] > grd.Xn[i].hi' + repmat(',:', 1, len(grd.Nx) - i) + ') = options.MyInf'])
            eval(['dyn.Jo[1](' + repmat(':,', 1, i - 1) + 'current_grd.X[i] > grd.Xn[i].hi' + repmat(',:', 1, len(grd.Nx) - i) + ') = options.MyInf'])
        if options.UseLevelSet:
            dyn.Jo[-2:] = -inf * zeros(sys.getsizeof(current_grd))
            if grd.Nx.size() > 1:
                code_fin_cst = '['
                for i in range(0, grd.Xn.size()):
                    code_fin_cst = [code_fin_cst, 'x[', str(i), ']']
                code_fin_cst = [code_fin_cst, '] = grid(']
                for i in range(0, grd.Xn.size()):
                    code_fin_cst = [code_fin_cst, '(current_grd.X[', str(i), '])']
                code_fin_cst = code_fin_cst(1, -2)
                code_fin_cst = [code_fin_cst, ')']
            else:
                code_fin_cst = 'x[1] = current_grd.X[1]'
            eval(code_fin_cst)
            for i in range(0, x.size()):
                dyn.Jo[-1] = max(dyn.Jo[-1], max(grd.XN[i].lo - x[i], x[i] - grd.XN[i].hi))
            if len(grd.Nx) > 1:
                code_x_grd = '['
                for i in rang(0, grd.Xn.size()):
                    code_x_grd = [code_fin_cst, 'x[', str(i), ']']
                code_x_grd = [code_fin_cst, '] = grid(']
                for i in range(0, grd.Xn.size()):
                    code_x_grd = [code_x_grd, '(1:x_sze(', str(i), '])']
                code_x_grd = code_x_grd(1, -2)
                code_x_grd = [code_x_grd, ')']
            else:
                code_x_grd = 'x1 = (1:x_sze(1))'''
    # initialization if the entire cost-to-go map should be saved
    dyn.Uo = cell(len(grd.Nu), dis.N)
    for i in range(0, len(grd.Nx)):
        code_generate_grid = code_generate_grid + 'inp.X[' + str(i) + ']'
    for i in range(0, len(grd.Nu)):
        code_generate_grid = code_generate_grid + 'inp.U[' + str(i) + ']'
    code_generate_grid = code_generate_grid + '= grid('
    for i in range(0, len(grd.Nx)):
        code_generate_grid = code_generate_grid + 'current_grd.X[' + str(i) + ']'
    for i in range(0, len(grd.Nx)):
        code_generate_grid = code_generate_grid + 'current_grd.U[' + str(i) + ']'
    code_generate_grid = code_generate_grid(1, -2)
    code_generate_grid = code_generate_grid + ')'

    # GENERATE   CODE   FOR   COST - TO - GO   INTERPOLATION
    eval(['xsize = [' + dpm_code('len(current_grd.X[#])', range(1, len(grd.Nx))) + ']'])
    for k in range(1, len(dyn.Jo)):
        code_cost_to_go_interp[k] = 'cost_to_go[' + int(k) + '] = dpm_interpn('
        for i in fliplr(find(xsize > 1)):
            code_cost_to_go_interp[k] = code_cost_to_go_interp[k] + 'previous_grd.X[' + str(i) + '],'
        code_cost_to_go_interp[k] = code_cost_to_go_interp[k] + 'dyn.Jo[' + int(k) + '],'
        for i in fliplr(find(xsize > 1)):
            if options.CalcLine:
                code_cost_to_go_interp[k] = code_cost_to_go_interp[k] + 'inp.X[' + str(i) + '],'
            else:
                code_cost_to_go_interp[k] = code_cost_to_go_interp[k] + 'out.X[' + str(i) + '],'
        code_cost_to_go_interp[k] = code_cost_to_go_interp[k](1, -2)
        code_cost_to_go_interp[k] = code_cost_to_go_interp[k] + ') '
    # GENERATE   CODE   FOR   CONVERTING   IND   TO   SUB
    code_ind2str = ''
    for i in range(1, len(grd.Nu)):
        code_ind2str = code_ind2str + 'uo' + str(i)
    code_ind2str = code_ind2str + ' = ind2sub(u_sze,ui)'
    if ~isdeployed and (options.Waitbar == 'on'):
        if ~options.CalcLine:
            h = waitbar(1, 'DP running backwards. Please wait...')
        else:
            h = waitbar(1, 'DP calculating boundary line. Please wait...')
        set(h, 'name', 'DPM:Waitbar')
    if ~isdeployed and (options.Verbose == 'on'):
        if ~options.CalcLine:
            print('#s', 'DP running backwards:     ##')
        else:
            print('#s', 'DP calculating boundary line:     ##')
    if options.UseLine:
        if len(grd.Nx) == 1:
            isfeas = dyn.B.lo.Xo[:, 0] > grd.X0[1] | dyn.B.hi.Xo[:, 0] < grd.X0[1]
        elif len(grd.Nx) == 2:
            isfeas = dpm_interpn(current_grd.X[2], dyn.B.lo.Xo[:, 0], grd.X0[2]) > grd.X0[1] | dpm_interpn(
                current_grd.X[2], dyn.B.hi.Xo(1), grd.X0[2]) < grd.X0[1]
    if sum(reshape(isfeas, 1, numel(isfeas))) != 0:
        print('DPM:Backward', 'Initial value not feasible!')
    iswarned = 0
    n = dis.N + 1
    while n > 1:
        n = n - 1
        previous_grd = current_grd
        x_sze = nan(1, len(grd.Nx))
        u_sze = nan(1, len(grd.Nu))
        for i in range(1, len(grd.Nx)):
            if options.UseLine and i == 1 and ~options.FixedGrid:
                if len(grd.Nx) == 1:
                    current_grd.X[i] = linspace(dyn.B.lo.Xo(n), dyn.B.hi.Xo(n), grd.Nx[i](n))
                else:
                    current_grd.X[i] = linspace(min(min(dyn.B.lo.Xo(range(1, n, n + 1)))),
                                                max(min(dyn.B.hi.Xo(range(1, n, n + 1))), grd.Nx[i](n)))
            elif ~options.CalcLine | (i != 1):
                current_grd.X[i] = linspace(grd.Xn[i].lo(n), grd.Xn[i].hi(n), grd.Nx[i](n))
            x_sze[i] = grd.Nx[i](n)
        for i in range(1, len(grd.Nu)):
            current_grd.U[i] = linspace(grd.Un[i].lo(n), grd.Un[i].hi(n), grd.Nu[i](n))
            u_sze[i] = grd.Nu[i](n)
    # generate  input and state   grid
    eval(code_generate_grid)
    if options.CalcLine:
        eval(['inp.X[1] = repmat(dyn.Jo[1],[ones(1,len(grd.Nx))'+dpm_code('len(current_grd.U[#])',range(1, len(grd.Nu))) + '])'])
    # Call model  function  _________________________________________________
    # generate  disturbance
    for i in range(1, len(dis.W)):
        inp.W[w] = dis.W[w](n)
    inp.Ts = dis.Ts
    try:
        # call   model    function
        if options.Signals:
            [out.X, out.C, out.I, signals] = model(inp, par)
        else:
            [out.X, out.C, out.I] = model(inp, par)
        if options.UseLevelSet:
            if n == dis.N:
                Vt = -inf * ones(size(out.X[1]))
                for i in range(1, len(grd.Xn)):
                    Vt = max(Vt, max(grd.XN[i].lo - out.X[i], out.X[i] - grd.XN[i].hi))
            else:
                eval(code_cost_to_go_interp[-1])
                Vt = cost_to_go[-1]
            Vt[out.I == 1] = options.MyInf
            if len(grd.Nu) > 1:
                Vt = reshape(Vt, array[x_sze, prod(u_sze)])
            array[dyn.Jo[-1], ub] = min(Vt, '', len(x_sze) + 1)
        # determine  the  arc - cost
        for i in range(1, len(grd.Nx)):
            out.I = bitor(out.I, out.X[i] > grd.Xn[i].hi(n + 1))
            out.I = bitor(out.I, out.X[i] < grd.Xn[i].lo(n + 1))
        J = (out.I == 0) * out.C[1] + out.I * options.MyInf
        # Calculate  cost  for entire grid
        if options.UseLine:
            if len(grd.Nx) == 1:
                cost_to_go[1] = dpm_interpf1mb(previous_grd.X[1], dyn.Jo[1], out.X[1],
                                               [dyn.B.lo.Xo(n + 1), dyn.B.hi.Xo(n + 1)],
                                               [dyn.B.lo.Jo(n + 1), dyn.B.hi.Jo(n + 1)], options.MyInf)
            else:
                cost_to_go[1] = dpm_interpf2mb(previous_grd.X[1], previous_grd.X[2], dyn.Jo[1], out.X[1], out.X[2],
                                               [dyn.B.lo.Xo(1, n + 1), dyn.B.hi.Xo(1, n + 1)],
                                               [dyn.B.lo.Jo(1, n + 1), dyn.B.hi.Jo(1, n + 1)], options.MyInf)

        else:
            if len(dyn.Jo[1]) == 1:
                cost_to_go[1] = dyn.Jo[1]
            else:
                eval(code_cost_to_go_interp[1])
        # total   cost = arc - cost + cost - to - go!
        Jt = J + cost_to_go[1]
        if options.Minimize:
            Jt[Jt > options.MyInf] = options.MyInf
            if options.CalcLine:
                Jt[Jt < grd.Xn[1].lo(n)] = options.MyInf
        else:
            Jt[Jt < options.MyInf] = options.MyInf
            if options.CalcLine:
                Jt[Jt > grd.Xn[1].hi(n)] = options.MyInf
    except:
        # err = lasterror
        if exist('out', 'var'):
            if sum(reshape(isnan(out.I), 1, numel(out.I))) > 0:
                print('DPM:Internal', 'Make sure the model does not output NaN in the variable I')

            if sum(reshape(isnan(out.C[1]), 1, numel(out.C[1]))) > 0:
                print('DPM:Internal', 'Make sure the model does not output NaN in the variable C')
            if sum(reshape(isnan(out.X[1]), 1, numel(out.X[1]))) > 0:
                print('DPM:Internal', 'Make sure the model does not output NaN in the variable X')
            print(err.message + ' Error in dpm_backward at n=' + int(n))
        print(err.message)
        if len(grd.Nu) > 1:
            Jt = reshape(Jt, [x_sze, prod(u_sze)])
            # minimize the cost-to-go
            if options.Minimize:
                [Q, ui] = min(Jt, '', len(x_sze) + 1)
            else:
                [Q, ui] = max(Jt, '', len(x_sze) + 1)
        if options.UseLevelSet:
            # handle infeasible states:
            Q_inf = J + cost_to_go[1]
            Q_inf = reshape(Q_inf, [x_sze, prod(u_sze)])
            eval(code_x_grd)
            val = len(grd.Nx)
            if val == 1:
                Qinf = Q_inf(sub2ind([x_sze, prod(u_sze)], x1, ub))
            elif val == 2:
                Qinf = Q_inf(sub2ind([x_sze, prod(u_sze)], x1, x2, ub))
            elif val == 3:
                Qinf = Q_inf(sub2ind([x_sze, prod(u_sze)], x1, x2, x3, ub))
            elif val == 4:
                Qinf = Q_inf(sub2ind([x_sze, prod(u_sze)], x1, x2, x3, x4, ub))
            Q[min(Vt, [], len(x_sze) + 1) > 0] = Qinf[min(Vt, [], len(x_sze) + 1) > 0]
            ui[min(Vt, [], len(x_sze) + 1) > 0] = ub[min(Vt, [], len(x_sze) + 1) > 0]
        if ~options.CalcLine and options['Signals'] and ~options['Signals']:
            for i in range(1, len(options.Signals)):
                try:
                    eval('dyn.' + options.Signals[i] + '[n] = signals.' + options.Signals[
                        i] + '(sub2ind([x_sze u_sze],' + dpm_code('(1:x_sze(#))''', range(1, len(x_sze)) + ',ui))'))
                except:
                    print('DPM:Backward' + 'options.signals element is not found in model output.')
        if sum(reshape(Q, 1, numel(Q)) == options.MyInf) == numel(Q):
            if options.UseLine:
                if dyn.B.hi.Jo(1, 1, n) == options.MyInf and dyn.B.lo.Jo(1, 1, n) == options.MyInf or sum(
                        reshape((out.X[1] > dyn.B.lo.Xo(n + 1)) and out.X[1] < dyn.B.hi.Xo(n + 1), 1,
                                numel(out.X[1]))) > 0 and sum(reshape(
                        inp.X[1]((out.X[1] > dyn.B.lo.Xo(n + 1)) and out.X[1] < dyn.B.hi.Xo(n + 1)) > dyn.B.lo.Xo(n) and (
                        inp.X[1](out.X[1] > dyn.B.lo.Xo(n + 1)) and (out.X[1] < dyn.B.hi.Xo(n + 1)) < dyn.B.hi.Xo(n), 1,
                        numel(inp.X[1](out.X[1] > dyn.B.lo.Xo(n + 1)) and out.X[1] < dyn.B.hi.Xo(n + 1))))) > 0:
                    if options['DebugMode'] and options.DebugMode:
                        print(
                            'DPM:Model function error \n \t Entering model function at the instance where the error occured \n\t Check if the entire grid generates infeasible solutions.\n')
                        if isa(model, 'function_handle'):
                            eval(['dbstop in ' + func2str(model) + ' at 1'])
                        else:
                            eval(['dbstop in ' + model + ' at 1'])

                        n = n + 1

                    print('DPM:Backward', 'No feasible solution Q(i,j,..) = Inf   for all i,j,...')

            else:
                if ~options.CalcLine and options['DebugMode'] and options.DebugMode:
                    if isa(model, 'function_handle'):
                        eval(['dbstop in ' + func2str(model) + ' at 1'])
                    else:
                        eval(['dbstop in ' + model + ' at 1'])
                    n = n + 1
                print('DPM:Backward', 'No feasible solution Q(i,j,..) = Inf   for all i,j,...')
        # Updateptimalostdyn.Jowith the minimum cost-to-go Q
        if options.UseLine:
            if len(grd.Nx) == 1:
                below = current_grd.X[1] < dyn.B.lo.Xo(n)
                above = current_grd.X[1] > dyn.B.hi.Xo(n)
                inside = current_grd.X[1] >= dyn.B.lo.Xo(n) and (current_grd.X[1] <= dyn.B.hi.Xo(n))
            else:
                below = current_grd.X[1] < min(dyn.B.lo.Xo(1, n))
                above = current_grd.X[1] > max(dyn.B.hi.Xo(1, n))
                inside = current_grd.X[1] >= min(dyn.B.lo.Xo(1, n)) and (current_grd.X[1] <= max(dyn.B.hi.Xo(1, n)))
            dyn.Jo[1] = nan(size(Q))

            eval(['dyn.Jo[1](:' + repmat(',:', 1, len(grd.Nx) - 1) + ')= Q '])
            # Single State: if all points are infeasible and some points are between
            # boundaries: use interpolation between boundary-data
            if dyn.B.lo.Jo(n) < options.MyInf and dyn.B.hi.Jo(n) < options.MyInf and len(grd.Nx) == 1 and sum(
                    reshape(Q, 1, numel(Q)) == options.MyInf) == numel(Q) and sum(
                    reshape(current_grd.X[1] > dyn.B.lo.Xo(n) and (current_grd.X[1] < dyn.B.hi.Xo(n)), 1,
                            numel(current_grd.X[1]))) > 0:
                dyn.Jo[1][current_grd.X[1] > dyn.B.lo.Xo(n) and (current_grd.X[1] < dyn.B.hi.Xo(n))] = dpm_interpn(
                    [dyn.B.lo.Xo(n), dyn.B.hi.Xo(n)], [dyn.B.lo.Jo(n), dyn.B.hi.Jo(n)],
                    current_grd.X[1](current_grd.X[1] > dyn.B.lo.Xo(n) and (current_grd.X[1] < dyn.B.hi.Xo(n))))

            eval(['dyn.Jo[1](above' + repmat(',:', 1, len(grd.Nx) - 1) + ') = options.MyInf '])
            eval(['dyn.Jo[1](below' + repmat(',:', 1, len(grd.Nx) - 1) + ') = options.MyInf '])
        else:
            dyn.Jo[1] = Q

            eval(code_ind2str)
        for i in range(2, len(out.C)):
            Xi[1] = ''
            for j in range(2, len(grd.Nx)):
                Xi[j] = reshape(out.X[j], [prod(x_sze), prod(u_sze)])

            ind2 = dpm_sub2ind([prod(x_sze), prod(u_sze)], range(1, prod(x_sze)), ui)

            Ci = reshape(out.C[i], [prod(x_sze), prod(u_sze)])
            if len(grd.Nx) > 1:
                eval(['cost_to_go[2] = dpm_interpn(' + dpm_code('current_grd.X[#],',range(2, len(grd.Nx))) + 'dyn.Jo[i]' + dpm_code(',out.X[#]',range(2,len(grd.Nx))) + ')'])
                dyn.Jo[i] = (reshape(cost_to_go[2](ind2), x_sze) != options.MyInf and reshape(out.I(ind2), x_sze) == 0) * (
                            reshape(Ci(ind2), x_sze) + reshape(cost_to_go[2](ind2), x_sze)) + (
                                        reshape(cost_to_go[2](ind2), x_sze) == options.MyInf or (
                                    reshape(out.I(ind2), x_sze)) != 0) * options.MyInf
            else:
                dyn.Jo[i] = (dyn.Jo[i] != options.MyInf and out.I(ind2) == 0) * (Ci(ind2) + dyn.Jo[i]) + (
                            dyn.Jo[i] == options.MyInf | out.I(ind2) != 0) * options.MyInf

            if options.Minimize:
                dyn.Jo[i][dyn.Jo[i] > options.MyInf] = options.MyInf
            else:
                dyn.Jo[i][dyn.Jo[i] < options.MyInf] = options.MyInf
        # Store the optima l  input that minimized the cost Jt
        if options.UseLine:
            for i in range(1, len(grd.Nu)):
                if len(grd.Nx) > 1:
                    eval(['dyn.Uo[i,n](:' + repmat(',:', 1, len(grd.Nx) - 1) + ') = current_grd.U[i](uo' + str(i) + ')'])

                    array[ind, col] = dpm_sub2indr([grd.Nx[1](n), grd.Nx[2](n)], ones(1, grd.Nx[2](n)),
                                                   dpm_findl(current_grd.X[1], dyn.B.lo.Xo(1, n)), 2)
                    array[s1, s2] = ind2sub([grd.Nx[1], grd.Nx[2]], ind)
                    dyn.Uo[i, n][dpm_sub2ind(size(dyn.Uo[i, n]), s1, s2)] = dyn.B.lo.Uo[i][
                        dpm_sub2ind(size(dyn.B.lo.Uo[i]), ones(size(col)), col, n * ones(size(col)))]

                    array[ind, col] = dpm_sub2indr([grd.Nx[1](n), grd.Nx[2](n)],
                                                   dpm_findu(current_grd.X[1], dyn.B.hi.Xo(1, n)),
                                                   grd.Nx[1](n) * ones(1, grd.Nx[2](n)), 2)
                    array[s1, s2] = ind2sub([grd.Nx[1](n), grd.Nx[2](n)], ind)
                    dyn.Uo[i, n][dpm_sub2ind(size(dyn.Uo[i, n]), s1, s2)] = dyn.B.hi.Uo[i][
                        dpm_sub2ind(size(dyn.B.lo.Uo[i]), ones(size(col)), col, n * ones(size(col)))]
                else:
                    eval(['dyn.Uo[i,n](:) = current_grd.U[i](uo' + str(i) + ')'])
                    if dyn.B.lo.Jo(n) < options.MyInf and dyn.B.hi.Jo(n) < options.MyInf and sum(
                            reshape(Q, 1, numel(Q)) == options.MyInf) == numel(Q) and sum(
                            reshape(current_grd.X[1] > dyn.B.lo.Xo(n) and (current_grd.X[1] < dyn.B.hi.Xo(n)), 1,
                                    numel(current_grd.X[1]))) > 0:
                        dyn.Uo[i, n][
                            current_grd.X[1] > dyn.B.lo.Xo(n) and (current_grd.X[1] < dyn.B.hi.Xo(n))] = dpm_interpn(
                            [dyn.B.lo.Xo(n), dyn.B.hi.Xo(n)], [dyn.B.lo.Uo[i](n), dyn.B.hi.Uo[i](n)],
                            current_grd.X[1][current_grd.X[1] > dyn.B.lo.Xo(n)] and (current_grd.X[1] < dyn.B.hi.Xo(n)))

                    dyn.Uo[i, n][elow] = dyn.B.lo.Uo[i](n)
                    dyn.Uo[i, n][above] = dyn.B.hi.Uo[i](n)
        else:
            for i in range(1, len(grd.Nu)):
                eval(['dyn.Uo[i,n](:' + repmat(',:', 1, len(grd.Nx) - 1) + ') = current_grd.U[i](uo' + str(i) + ')'])
        # store cost - to - go if save map options is set
        if options.SaveMap:
            for i in range(1, len(dyn.Jo)):
                eval(['V_map[i,n](:' + repmat(',:', 1, len(grd.Nx) - 1) + ') = dyn.Jo[i] '])
            # Update progres bar for the dynamic programming backward
            if ~isdeployed and options.Waitbar == 'on':
                waitbar(n / dis.N, h)
            if ~isdeployed and options.Verbose == 'on' and mod(n - 1, floor(dis.N / 100)) == 0 and (
                    round(100 * n / dis.N) < 100):
                print('#s#2d ##', ones(1, 4) * 8, round(100 * n / dis.N))
    # finish time loop
    # Clear dyn if in low mem mode
    if ~options.SaveMap:
        dyn.Jo = ''
    else:
        dyn.Jo = V_map
    # Clear dyn if in low mem mode
    if ~options.SaveMap:
        dyn.Jo = []
    else:
        dyn.Jo = V_map

    # Close progres bar
    if ~isdeployed and options.Waitbar == 'on':
        waitbar(1, h)
        close(h)

    if ~isdeployed and options.Verbose == 'on':
        print('#s Done!\n', ones(1, 5) * 8)
    return dyn
def dpm_boundary_line_lower(model, par, grd, dis, options):

        optb = options
        parb = par
        grdb = grd
        grdb.Xn_true[1].lo = grd.Xn[1].lo
        grdb.Xn_true[1].hi = grd.Xn[1].hi
        grdb.Nx[1] = ones(1, dis.N + 1)
        grdb.Xn[1].lo = grd.X0[1] * ones(1, dis.N + 1)
        grdb.Xn[1].hi = grd.X0[1] * ones(1, dis.N + 1)
        optb.UseLine = 0
        optb.SaveMap = 1
        parb.model = model
        parb.options.Iter = options.Iter
        parb.options.Tol = options.Tol
        optb.CalcLine = 1

        if options['gN']:
            print('options.gN can only be used w/ 1 state boundary')
            optb.gN[1] = grd.XN[1].lo
            optb.gN[2] = dpm_interpn(grd.X[1], options.gN[1], grdb.X[1])
        else:
            optb.gN[1] = grd.XN[1].lo
            optb.gN[2] = zeros(size(grd.XN[1].lo))
        optb.MyInf = options.MyInf
        optb.Minimize = 1
        optb.Warnings = 'off'
        dynb = dpm(dpm_model_inv(parb, grdb, dis, optb))
        # convert cellarray to vectors
        vsize = dynb.Jo.size()
        for j in range(1, vsize(1)):
            V_new[j] = nan(1, len(dynb.Jo[j, 1]), vsize(2))
            for i in range(1, vsize(2)):
                V_new[j][::, i] = dynb.Jo[j, i]

        usize = size(dynb.Uo)
        for j in range(1, usize(1)):
            O_new[j] = nan(1, len(dynb.Uo[j, 1]), usize(2))
            for i in range(1, usize(2)):
                O_new[j][::, i] = dynb.Uo[j, i]

        dynb.Jo = V_new
        dynb.Uo = O_new
        # convert infeasible points to minimum state boundary
        for i in range(1, dis.N):
            dynb.Jo[1][1, dynb.Jo[1][::, i] > grd.Xn[1].hi(i), i] = grd.Xn[1].lo(i)
            dynb.Jo[1][1, isnan(dynb.Jo[1][::, i]), i] = grd.Xn[1].lo(i) * ones(
                size(dynb.Jo[1][1, isnan(dynb.Jo[1][1, :, i], i)]))
        dynb.Jo[2][isnan(dynb.Jo[2]) | dynb.Jo[2] >= optb.MyInf] = options.MyInf
        for i in range(1, len(dynb.Uo)):
            dynb.Uo[i][isnan(dynb.Uo[i])] = 1

        # insert lower boundary into original problem definition
        LineLower.Xo = dynb.Jo[1]
        LineLower.Uo = dynb.Uo
        LineLower.Jo = dynb.Jo[2]
        return LineLower


def dpm_boundary_line_upper(model, par, grd, dis, options):
        optb = options
        parb = par
        grdb = grd
        grdb.Xn_true[1].lo = grd.Xn[1].lo
        grdb.Xn_true[1].hi = grd.Xn[1].hi
        grdb.Nx[1] = ones(1, dis.N + 1)
        grdb.Xn[1].lo = grd.X0[1] * ones(1, dis.N + 1)
        grdb.Xn[1].hi = grd.X0[1] * ones(1, dis.N + 1)
        optb.UseLine = 0
        optb.SaveMap = 1
        parb.model = model
        parb.options.Iter = options.Iter
        parb.options.Tol = options.Tol
        optb.CalcLine = 1
        if options['gN']:
            print('options.gN can only be used w/ 1 state boundary')
            optb.gN[1] = grd.XN[1].hi
            optb.gN[2] = dpm_interpn(grd.X[1], options.gN[1], grdb.X[1])
        else:
            optb.gN[1] = grd.XN[1].hi
            optb.gN[2] = zeros(size(grd.XN[1].hi))
        optb.MyInf = -options.MyInf
        optb.Minimize = 0
        optb.Warnings = 'off'
        dynb = dpm(dpm_model_inv(parb, grdb, dis, optb))
        vsize = size(dynb.Jo)
        for j in range(1, vsize(1)):
            V_new[j] = nan(1, len(dynb.Jo[j, 1]), vsize(2))
            for i in range(1, vsize(2)):
                V_new[j][::, i] = dynb.Jo[j, i]
        usize = size(dynb.Uo)
        for j in range(1, usize(1)):
            O_new[j] = nan(1, len(dynb.Uo[j, 1]), usize(2))
            for i in range(1, usize(2)):
                O_new[j][1, :, i] = dynb.Uo[j, i]
        dynb.Jo = V_new
        dynb.Uo = O_new
        for i in range(1, dis.N):
            dynb.Jo[1][1, dynb.Jo[1][1, :, i] < grd.Xn[1].lo(i), i] = grd.Xn[1].hi(i)
            dynb.Jo[1][1, isnan(dynb.Jo[1][1, :, i]), i] = grd.Xn[1].hi(i) * ones(
                size(dynb.Jo[1][1, isnan(dynb.Jo[1][1, :, i], i)]))
            dynb.Jo[2][isnan(dynb.Jo[2]) | dynb.Jo[2] <= optb.MyInf] = options.MyInf
        for i in range(1, len(dynb.Uo)):
            dynb.Uo[i][isnan(dynb.Uo[i])] = 1
            LineUpper.Xo = dynb.Jo[1]
            LineUpper.Uo = dynb.Uo
            LineUpper.Jo = dynb.Jo[2]
        return LineUpper


def dpm_forward(dyn, model, par, grd, dis, options):
        if ~options['InputType'] or len(options.InputType) != len(grd.Nu):
            options.InputType = repmat('c', 1, len(grd.Nu))

        code_grid_states = ''
        for i in range(1, len(grd.Nx)):
            code_grid_states = code_grid_states + 'current_grd.X[' + str(i) + ']'
        code_input_states = ''
        for i in range(1, len(grd.Nx)):
            code_input_states = code_input_states + ',inp.X[' + str(i) + ']'
        code_states_nearest = 'ixm1'
        for i in range(2, len(grd.Nx)):
            code_states_nearest = code_states_nearest + ',ixm' + str(i)

        inp = dpm_get_empty_inp(grd, dis, 'zero')
        outn = dpm_get_empty_out(model, inp, par, grd, 'zero')
        for i in range(1, len(grd.Nx)):
            inp.X[i] = grd.X0[i]
            outn.X[i] = grd.X0[i]
        if ~isdeployed and options.Waitbar == 'on':
            h = waitbar(0, 'DP running forwards. Please wait...')
            set(h, 'name', 'DPM:Waitbar')

        if ~isdeployed and options.Verbose == 'on':
            print('DP running forwards:     0 ##')

        # backward    compability    of    dis.N0
        if ~isfield(dis, 'T0'):
            dis.N0 = 1

        if ~options.UseUmap:
            if len(grd.Nu) > 1:
                code_generate_grid = '['
                for i in range(1, len(grd.Nu)):
                    code_generate_grid = code_generate_grid + 'inpt.U[' + str(i) + '] '
            code_generate_grid = code_generate_grid + '] = grid('
            for i in range(1, len(grd.Nu)):
                code_generate_grid = code_generate_grid + 'current_grd.U[' + str(i) + '],'
            code_generate_grid = code_generate_grid(range(1, -2))
        else:
            code_generate_grid = 'inpt.U[1] = current_grd.U[1]'
        Jsze = size(dyn.Jo)
        for k in range(1, Jsze(1)):
            code_cost_to_go_interp[k] = 'cost_to_go[' + str(k) + '] = dpm_interpn('
            for i in range(len(grd.Nx), 1, -1):
                code_cost_to_go_interp[k] = code_cost_to_go_interp[k] + 'next_grd.X[' + str(i) + '],'
            code_cost_to_go_interp[k] = code_cost_to_go_interp[k] + 'dyn.Jo[' + str(k) + ',n+1],'
            for i in range(len(grd.Nx), 1, -1):
                code_cost_to_go_interp[k] = code_cost_to_go_interp[k] + 'X[' + str(i) + '],'
            code_cost_to_go_interp[k] = code_cost_to_go_interp[k](range(1, -2))
            code_cost_to_go_interp[k] = code_cost_to_go_interp[k] + ') '
        iswarned = 0
        # Forward   sim
        for n in range(disN0, dis.N):
            for i in range(1, len(grd.Nx)):
                if options.UseLine and len(grd.Nx) == 1 and i == 1 and ~options.FixedGrid:
                    current_grd.X[i] = linspace(dyn.B.lo.Xo(n), dyn.B.hi.Xo(n), grd.Nx[i](n))
                elif options.UseLine and ~options.FixedGrid:
                    current_grd.X[i] = linspace(min(dyn.B.lo.Xo(range(1, n))), max(dyn.B.hi.Xo(range(1, n))),
                                                grd.Nx[i](n))
                else:
                    current_grd.X[i] = linspace(grd.Xn[i].lo(n), grd.Xn[i].hi(n), grd.Nx[i](n))
            for i in range(1, len(grd.Nu)):
                current_grd.U[i] = linspace(grd.Un[i].lo(n), grd.Un[i].hi(n), grd.Nu[i](n))
            for w in range(1, len(dis.W)):
                inp.W[w] = dis.W[w](n)
            inp.Ts = dis.Ts
            if ~options.UseUmap:
                for i in range(1, len(grd.Nx)):
                    next_grd.X[i] = linspace(grd.Xn[i].lo(n + 1), grd.Xn[i].hi(n + 1), grd.Nx[i](n + 1))
                for w in range(1, len(dis.W)):
                    inpt.W[w] = dis.W[w](n)
                inpt.Ts = dis.Ts
                eval(code_generate_grid)
                for i in range(1, len(grd.Nx)):
                    inpt.X[i] = inp.X[i] * ones(size(inpt.U[1]))
                # input    to    system
                [X, C, I] = feval(model, inpt, par)
                # take    care    of    bounds
                for i in range(1, len(grd.Nx)):
                    I = bitor(I, X[i] > grd.Xn[i].hi(n + 1))
                    I = bitor(I, X[i] < grd.Xn[i].lo(n + 1))
                if options.UseLine:
                    I = bitor(I, X[i] > dyn.B.hi.Xo(n + 1))
                    I = bitor(I, X[i] < dyn.B.lo.Xo(n + 1))
                # arc   cost
                J = (I == 0) * C[1] + I * options.MyInf
                if options.UseLevelSet:
                    eval(code_cost_to_go_interp[-1])
                    J[cost_to_go[-1] > 0] = options.MyInf
                # minimize     total    cost
                # Calculate  cost    for entire grid
                if len(dyn.Jo[1]) == 1:
                    cost_to_go[1] = dyn.Jo[1]
                else:
                    # interpolate from cost to  go   map
                    eval(code_cost_to_go_interp[1])
                Jt = J + cost_to_go[1]
                if options.UseLevelSet and ~any(
                        reshape(cost_to_go[-1], numel(cost_to_go[-1]), 1) <= 0 and (reshape(I, numel(I), 1) == 0)):
                    Jt = cost_to_go[2]
                    Jt[I != 0] = options.MyInf
                if options.Minimize:
                    Jt[Jt > options.MyInf] = options.MyInf
                else:
                    Jt[Jt < options.MyInf] = options.MyInf
                if len(grd.Nu) > 1:
                    Jt = reshape(Jt, [1, numel(Jt)])

                # minimize   the   cost - to - go
                if options.Minimize:
                    [Q, ui] = min(Jt)
                else:
                    [Q, ui] = max(Jt)
                # use input  that  minimizes  total  cost
                for i in range(1, len(inpt.U)):
                    if len(grd.Nu) > 1:
                        inpt.U[i] = reshape(inpt.U[i], [1, numel(inpt.U[i])])
                    inp.U[i] = inpt.U[i](ui(1))
            else:
                if options.UseLine:
                    for i in range(1, len(grd.Nu)):
                        xi = cell(1, len(grd.Nx))
                        for j in range(len(grd.Nx), 1, -1):
                            if options.InputType(i) == 'd':
                                if j == 1:
                                    xistr = dpm_code('xi[#],', range(2, len(current_grd.X)))
                                    eval(['x1vec = [dyn.B.lo.Xo(1,' + xistr + 'n)current_grd.X[j](current_grd.X[j]>dyn.B.lo.Xo(1,' + xistr + 'n) and current_grd.X[j]<dyn.B.hi.Xo(1,' + xistr + 'n))  dyn.B.hi.Xo(1,' + xistr + 'n)] '])
                                    [temp, xi[j]] = min(abs(x1vec - inp.X[j]))
                                    Xin[j] = x1vec(xi[j])
                                else:
                                    xi[j] = round((inp.X[j] - current_grd.X[j][1]) / (
                                                current_grd.X[j][2] - current_grd.X[j][1])) + 1
                                    xi[j] = max(xi[j], 1)
                                    xi[j] = min(xi[j], len(current_grd.X[j]))
                                    Xin[j] = current_grd.X[j](xi[j])
                            else:
                                Xin[j] = inp.X[j]
                        if len(grd.Nx) > 1:
                            inp.U[i] = dpm_interpf2sbh(current_grd.X[1], current_grd.X[2], dyn.Uo[i, n][::, 1], Xin[1],
                                                       Xin[2], [dyn.B.lo.Xo(range(1, n)), dyn.B.hi.Xo(range(1, n))])
                        else:
                            inp.U[i] = dpm_interpf1sbh(current_grd.X[1], dyn.Uo[i, n], Xin[1],
                                                       [dyn.B.lo.Xo(n), dyn.B.hi.Xo(n)])
                elif ~isempty(grd.Nu):
                    for i in range(1, len(grd.Nu)):
                        if options.InputType(i) == 'c':
                            eval(['inp.U[i]   = dpm_interpn(' + code_grid_states + 'dyn.Uo[i,n](:' + repmat(',:', 1,len(grd.Nx) - 1) + ')' + code_input_states + ')'])
                        else:
                            for j in range(1, len(grd.Nx)):
                                eval('ixm' + str(
                                    j) + ' = round((inp.X[j]-current_grd.X[j](1))/(current_grd.X[j](2)-current_grd.X[j](1)) + 1) ')
                            eval('inp.U[i] = dyn.Uo[i,n](' + code_states_nearest + ')')

            # call model with optimal input:
            #[X, C, I, outn] = feval(model, inp, par)
            print(eval(model,inp,par))
            outn.X = inp.X
            outn.C = C
            outn.I = I
            if outn.I != 0 and ~iswarned:
                print('DPM:Forward', 'Infeasible Solution!')
            iswarned = 1
            inp.X = X

            if n > dis.N0:
                out = dpm_mergestruct(out, outn)
            else:
                out = outn
            # Update   progres    bar  for the Dynamic Programming
            if ~isdeployed and options.Waitbar == 'on':
                waitbar((n - dis.N0) / (dis.N - dis.N0), h)

            if ~isdeployed and options.Verbose == 'on' and mod(n - 1, floor(dis.N / 100)) == 0 and round(
                    100 * n / dis.N) < 100:
                print('#s#2d ##', ones(1, 4) * 8, round(100 * n / dis.N))

        for i in range(1, len(outn.X)):
            out.X[i] = [out.X[i], inp.X[i]]
        # Close progres  bar
        if ~isdeployed and options.Waitbar == 'on':
            waitbar(1, h)
            close(h)
        if ~isdeployed and options.Verbose == 'on':
            print('#s Done!\n', ones(1, 5) * 8)
        return out


def dpm_interpf1mb(xx, yy, A, xlim, ylim, myInf):
        y = zeros(size(A))
        if ~isempty(Iin):
            y[Iin] = dpm_interpn(xx + ',yy', A(Iin))

        if ~isempty(Iout):
            y[Iout] = myInf

        if ~isempty(find(xx < xlim(2) and (xx > xlim(1)), 1)):
            # interpolate points between lower boundary and closest feasible grid point
            if ~isempty(Ibel):
                y[Ibel] = dpm_interpn([xlim(1), xx(Iinl)], [ylim(1), yy(Iinl)], A(Ibel))
            # interpolate points between upper boundary and closest feasible grid point
            if ~isempty(Ibeu):
                y[Ibeu] = dpm_interpn([xx(Iinu), xlim(2)], [yy(Iinu), ylim(2)], A(Ibeu))

        else:
            y[Iinx] = dpm_interpn(xlim, ylim, A(Iinx))
        return y


def dpm_interpf2mb(xx1, xx2, YY, A1, A2, xlim, ylim, myInf):
        XX1 = repmat(xx1, len(xx2), 1)
        XLIMl = repmat(xlim(range(1, -1)), 1, len(xx1))
        XLIMu = repmat(xlim(range(1, -1)), 1, len(xx1))

        # find grid point just inside lower boundary
        [r, c] = find(XX1 > XLIMl)
        [ru, in1] = unique(r, 'first')
        Iinl = c(in1)

        # find grid point just inside upper boundary
        [r, c] = find(XX1 < XLIMu)
        [ru, in1] = unique(r, 'last')
        Iinu = c(in1)
        xliml = xlim(range(1, -1))
        xlimu = xlim(range(2, -1))
        yliml = ylim(range(1, -1))
        ylimu = ylim(range(2, -1))
        xx1l = xx1(Iinl)
        xx1u = xx1(Iinu)

        # find interpolation points between lower boundary and closest grid point
        Ibel = find(A1 >= dpm_interpn(xx2, xliml, A2) and (A1 < dpm_interpn(xx2, xx1l, A2)))
        # find interpolation points between upper boundary and closest grid point
        Ibeu = find(A1 <= dpm_interpn(xx2, xlimu, A2) and (A1 > dpm_interpn(xx2, xx1u, A2)))
        # find interpolation points outside boundary
        Iout = find(A1 < dpm_interpn(xx2, xliml, A2) | A1 > dpm_interpn(xx2, xlimu, A2))
        # find interpolation points inside boundary
        Iin = find(A1 >= dpm_interpn(xx2, xx1l, A2) and (A1 <= dpm_interpn(xx2, xx1u, A2)))
        # find interpolation points between lower and upper boundary
        Iinx = find(A1 >= dpm_interpn(xx2, xliml, A2) and (A1 <= dpm_interpn(xx2, xlimu, A2)))

        # initialize output
        y = nan(size(A1))

        # interpolate as usual with interior points
        if ~isempty(Iin):
            y[Iin] = dpm_interpn(xx2, xx1, YY, A2(Iin), A1(Iin))
        # set outside points to inf
        if ~isempty(Iout):
            y[Iout] = myInf
        # if there are grid points between boundaries
        # if ~isempty(find(xx1<xlimu and xx1>xliml,1))
        if ~isempty(min(Iinu - Iinl) > 0):
            # interpolate points between lower boundary and closest feasible grid point
            if ~isempty(Ibel):
                Xl = dpm_interpn(xx2, xliml, A2(Ibel))
                Xu = dpm_interpn(xx2, xx1l, A2(Ibel))
                Yl = dpm_interpn(xx2, yliml, Xl)
                Yu = dpm_interpn(xx2, YY(dpm_sub2ind(size(YY), Iinl, (range(1, len(xx2)))), Xu))
                y[Ibel] = Yl + (A1(Ibel) - Xl) / (Xu - Xl) * (Yu - Yl)

            # interpolate points between upper boundary and closest feasible grid point
            if ~isempty(Ibeu):
                Xu = dpm_interpn(xx2, xlimu, A2(Ibeu))
                Xl = dpm_interpn(xx2, xx1u, A2(Ibeu))
                Yu = dpm_interpn(xx2, ylimu, Xu)
                Yl = dpm_interpn(xx2, YY(dpm_sub2ind(size(YY), Iinu, (range(1, len(xx2)))), Xl))
                y[Ibeu] = Yl + (A1(Ibeu) - Xl) / (Xu - Xl) * (Yu - Yl)
        else:
            # if there are no grid points between boundaries
            y[Iinx] = dpm_interpn(xlim, ylim, A(Iinx))
        return y


def dpm_interpf1sbh(xx, yy, a, lim):
        # MY_INTERPF1M Computes the 1D interpolation for the given set A
        #   using the function YY(xx). USES EXTRAPOLATION
        xlu = find(xx > lim(1), 1, 'first')
        xll = find(xx <= lim(1), 1, 'last')
        xuu = find(xx >= lim(2), 1, 'first')
        xul = find(xx < lim(2), 1, 'last')

        # if a is between lower limit and regular grid
        if a <= lim(1):
            y = yy(xll)
            # if a is outside upper limit
        elif a >= lim(2):
            y = yy(xuu)
            # if a is inside limits and within regular grid
        elif (a < xx(xlu)) and a > lim(1):
            #     # if close to engine off
            #     if yy(xlu) == 1 || yy(xll) == 1
            #         [tmp ind] = min(abs([xx(xlu) lim(1)]-a))
            #         ytmp = [yy(xlu) yy(xll)]
            #         y = ytmp(ind)
            #     else
            dy = yy(xlu) - yy(xll)
            dx = xx(xlu) - lim(1)
            y = (a - lim(1)) * dy / dx + yy(xll)
        #     end
        # if a is between upper limit and regular grid
        elif (a < lim(2)) and a > xx(xul):
            dy = yy(xuu) - yy(xul)
            dx = lim(2) - xx(xul)
            y = (a - xx(xul)) * dy / dx + yy(xul)
        else:
            y = dpm_interpn(xx, yy, a)
        return y


def dpm_interpf2sbh(xx1, xx2, YY, a1, a2, lim):

        lim2[1] = dpm_interpn(xx2, lim(range(1, -1)), a2)
        lim2[2] = dpm_interpn(xx2, lim(range(2, -1)), a2)
        yy = dpm_interpn(xx2, xx1, yy, a2 * ones(size(xx1)), xx1)
        y = dpm_interpf1sbh(xx1, yy, a1, lim2)
        return y


def dpm_interpn(varargin):
        switch_val = (nargin - 1) / 2
        if switch_val == 1:
            xx1 = varargin[1]
            YY = varargin[2]
            A1 = varargin[3]
            Ars = [reshape(A1, [numel(A1), 1])]
            xx[1] = reshape(xx1, [numel(xx1), 1])
            h = max([max(diff(xx[1]))], eps)
            Ars[Ars[:, 0] < min(xx[1]), 1] = min(xx[1])
            Ars[Ars[:, 0] > max(xx[1]), 1] = max(xx[1])

            ind[:, 1, 1] = 1 + floor(round((Ars[:, 0] - xx[1](1)) / h(1) * 10 ** 8) * 10 ** (-8))
            ind[:, 1, 2] = 1 + ceil(round((Ars[:, 0] - xx[1](1)) / h(1) * 10 ** 8) * 10 ** (-8))
            yy[:, 1, 1] = YY(dpm_sub2ind(size(YY), ind[:, 1, 1]))
            yy[:, 2, 1] = YY(dpm_sub2ind(size(YY), ind[:, 1, 2]))
            da[:, 0] = (Ars[:, 0] - xx[1](ind[:, 1, 1])) / h(1)
            yi = yy[:, 1, 1] + da[:, 0] * (yy[:, 2, 1] - yy[:, 1, 1])
            y = reshape(yi, size(A1))
        elif switch_val == 2:
            xx2 = varargin[1]
            xx1 = varargin[2]
            YY = varargin[3]
            A2 = varargin[4]
            A1 = varargin[5]
            Ars = [reshape(A1, [numel(A1), 1]), reshape(A2, [numel(A2), 1])]
            xx[1] = reshape(xx1, [numel(xx1), 1])
            xx[2] = reshape(xx2, [numel(xx2), 1])
            h = max([max(diff(xx[1])), max(diff(xx[2]))], eps)

            Ars[Ars[:, 0] < min(xx[1]), 1] = min(xx[1])
            Ars[Ars[:, 0] < min(xx[2]), 2] = min(xx[2])
            Ars[Ars[:, 0] > max(xx[1]), 1] = max(xx[1])
            Ars[Ars[:, 0] > max(xx[2]), 2] = max(xx[2])

            ind[:, 1, 1] = 1 + floor(round((Ars[:, 0] - xx[1](1)) / h(1) * 10 ** 8) * 10 ** (-8))
            ind[:, 2, 1] = 1 + floor(round((Ars[:, 0] - xx[2](1)) / h(2) * 10 ** 8) * 10 ** (-8))
            ind[:, 1, 2] = 1 + ceil(round((Ars[:, 0] - xx[1](1)) / h(1) * 10 ** 8) * 10 ** (-8))
            ind[:, 2, 2] = 1 + ceil(round((Ars[:, 0] - xx[2](1)) / h(2) * 10 ** 8) * 10 ** (-8))
            yy[:, 1, 1] = YY(dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1]))
            yy[:, 2, 1] = YY(dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1]))
            yy[:, 1, 2] = YY(dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2]))
            yy[:, 2, 2] = YY(dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2]))
            da[:, 0] = (Ars[:, 0] - xx[1](ind[:, 1, 1])) / h(1)
            da[:, 0] = (Ars[:, 0] - xx[2](ind[:, 2, 1])) / h(2)
            v1[:, 0] = yy[:, 1, 1] + da[:, 0] * (yy[:, 2, 1] - yy[:, 1, 1])
            v1[:, 0] = yy[:, 1, 2] + da[:, 0] * (yy[:, 2, 2] - yy[:, 1, 2])
            yi = da[:, 0] * (v1[:, 0] - v1[:, 0]) + v1[:, 0]
            y = reshape(yi, size(A1))
        elif switch_val == 3:
            xx3 = varargin[1]
            xx2 = varargin[2]
            xx1 = varargin[3]
            yy = varargin[4]
            A3 = varargin[5]
            A2 = varargin[6]
            A1 = varargin[7]

            Ars = [reshape(A1, [numel(A1), 1]), reshape(A2, [numel(A2), 1]), reshape(A3, [numel(A3), 1])]
            xx[1] = reshape(xx1, [numel(xx1), 1])
            xx[2] = reshape(xx2, [numel(xx2), 1])
            xx[3] = reshape(xx3, [numel(xx3), 1])
            h = max([max(diff(xx[1])), max(diff(xx[2])), max(diff(xx[3]))], eps)
            Ars[Ars[:, 0] < min(xx[1]), 1] = min(xx[1])
            Ars[Ars[:, 0] < min(xx[2]), 2] = min(xx[2])
            Ars[Ars[:, 0] < min(xx[3]), 3] = min(xx[3])
            Ars[Ars[:, 0] > max(xx[1]), 1] = max(xx[1])
            Ars[Ars[:, 0] > max(xx[2]), 2] = max(xx[2])
            Ars[Ars[:, 0] > max(xx[3]), 3] = max(xx[3])

            ind[:, 1, 1] = 1 + floor(round((Ars[:, 0] - xx[1](1)) / h(1) * 10 ** 8) * 10 ** (-8))
            ind[:, 2, 1] = 1 + floor(round((Ars[:, 0] - xx[2](1)) / h(2) * 10 ** 8) * 10 ** (-8))
            ind[:, 3, 1] = 1 + floor(round((Ars[:, 0] - xx[3](1)) / h(3) * 10 ** 8) * 10 ** (-8))

            ind[:, 1, 2] = 1 + ceil(round((Ars[:, 0] - xx[1](1)) / h(1) * 10 ** 8) * 10 ** (-8))
            ind[:, 2, 2] = 1 + ceil(round((Ars[:, 0] - xx[2](1)) / h(2) * 10 ** 8) * 10 ** (-8))
            ind[:, 2, 2] = 1 + ceil(round((Ars[:, 0] - xx[3](1)) / h(3) * 10 ** 8) * 10 ** (-8))

            yy[:, 1, 1, 1] = YY(dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 3, 1]))
            yy[:, 2, 1, 1] = YY(dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 3, 1]))
            yy[:, 1, 2, 1] = YY(dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 3, 1]))
            yy[:, 2, 2, 1] = YY(dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 3, 1]))
            yy[:, 1, 1, 2] = YY(dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 2, 2]))
            yy[:, 2, 1, 2] = YY(dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 2, 2]))
            yy[:, 2, 2, 2] = YY(dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 2, 2]))
            yy[:, 1, 2, 2] = YY(dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 2, 2]))

            da[:, 0] = (Ars[:, 0] - xx[1](ind[:, 1, 1])) / h(1)
            da[:, 0] = (Ars[:, 0] - xx[2](ind[:, 2, 1])) / h(2)
            da[:, 0] = (Ars[:, 0] - xx[3](ind[:, 3, 1])) / h(3)

            v2[:, 1, 1] = yy[:, 1, 1, 1] + da[:, 0] * (yy[:, 1, 1, 2] - yy[:, 1, 1, 1])
            v2[:, 2, 1] = yy[:, 2, 1, 1] + da[:, 0] * (yy[:, 2, 1, 2] - yy[:, 2, 1, 1])
            v2[:, 1, 2] = yy[:, 1, 2, 1] + da[:, 0] * (yy[:, 2, 2, 2] - yy[:, 1, 2, 1])
            v2[:, 2, 2] = yy[:, 2, 2, 1] + da[:, 0] * (yy[:, 1, 2, 2] - yy[:, 2, 2, 1])

            v1[:, 0] = v2[:, 1, 1] + da[:, 0] * (v2[:, 1, 2] - v2[:, 1, 1])
            v1[:, 0] = v2[:, 2, 1] + da[:, 0] * (v2[:, 2, 2] - v2[:, 2, 1])
            yi = da[:, 0] * (v1[:, 0] - v1[:, 0]) + v1[:, 0]
            y = reshape(yi, size(A1))
        elif switch_val == 4:

            xx4 = varargin[1]
            xx3 = varargin[2]
            xx2 = varargin[3]
            xx1 = varargin[4]
            YY = varargin[5]
            A4 = varargin[6]
            A3 = varargin[7]
            A2 = varargin[8]
            A1 = varargin[9]

            Ars = [reshape(A1, [numel(A1), 1]), reshape(A2, [numel(A2), 1]), reshape(A3, [numel(A3), 1]),
                   reshape(A4, [numel(A4), 1])]
            xx[1] = reshape(xx1, [numel(xx1), 1])
            xx[2] = reshape(xx2, [numel(xx2), 1])
            xx[3] = reshape(xx3, [numel(xx3), 1])
            xx[4] = reshape(xx4, [numel(xx4), 1])

            h = max([max(diff(xx[1])), max(diff(xx[2])), max(diff(xx[3])), max(diff(xx[4]))], eps)

            Ars[Ars[:, 0] < min(xx[1]), 1] = min(xx[1])
            Ars[Ars[:, 0] < min(xx[2]), 2] = min(xx[2])
            Ars[Ars[:, 0] < min(xx[3]), 3] = min(xx[3])
            Ars[Ars[:, 3] < min(xx[4]), 4] = min(xx[4])

            Ars[Ars[:, 0] > max(xx[1]), 1] = max(xx[1])
            Ars[Ars[:, 0] > max(xx[2]), 2] = max(xx[2])
            Ars[Ars[:, 0] > max(xx[3]), 3] = max(xx[3])
            Ars[Ars[:, 3] > max(xx[4]), 4] = max(xx[4])

            ind[:, 0, 1] = 1 + floor(round((Ars[:, 0] - xx[1](1)) / h(1) * 10 ** 8) * 10 ** (-8))
            ind[:, 1, 1] = 1 + floor(round((Ars[:, 0] - xx[2](1)) / h(2) * 10 ** 8) * 10 ** (-8))
            ind[:, 2, 1] = 1 + floor(round((Ars[:, 0] - xx[3](1)) / h(3) * 10 ** 8) * 10 ** (-8))
            ind[:, 3, 1] = 1 + floor(round((Ars[:, 3] - xx[4](1)) / h(4) * 10 ** 8) * 10 ** (-8))

            ind[:, 0, 2] = 1 + ceil(round((Ars[:, 0] - xx[1](1)) / h(1) * 10 ** 8) * 10 ** (-8))
            ind[:, 1, 2] = 1 + ceil(round((Ars[:, 0] - xx[2](1)) / h(2) * 10 ** 8) * 10 ** (-8))
            ind[:, 2, 2] = 1 + ceil(round((Ars[:, 0] - xx[3](1)) / h(3) * 10 ** 8) * 10 ** (-8))
            ind[:, 3, 2] = 1 + ceil(round((Ars[:, 3] - xx[4](1)) / h(4) * 10 ** 8) * 10 ** (-8))

            yy[:, 1, 1, 1, 1] = YY(dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 1]))
            yy[:, 2, 1, 1, 1] = YY(dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 1]))
            yy[:, 1, 2, 1, 1] = YY(dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 1]))
            yy[:, 2, 2, 1, 1] = YY(dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 1]))
            yy[:, 1, 1, 2, 1] = YY(dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 1]))
            yy[:, 2, 1, 2, 1] = YY(dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 1]))
            yy[:, 1, 2, 2, 1] = YY(dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 1]))
            yy[:, 2, 2, 2, 1] = YY(dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 1]))
            yy[:, 1, 1, 1, 2] = YY(dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 2]))
            yy[:, 2, 1, 1, 2] = YY(dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 2]))
            yy[:, 1, 2, 1, 2] = YY(dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 2]))
            yy[:, 2, 2, 1, 2] = YY(dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 2]))
            yy[:, 1, 1, 2, 2] = YY(dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 2]))
            yy[:, 2, 1, 2, 2] = YY(dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 2]))
            yy[:, 1, 2, 2, 2] = YY(dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 2]))
            yy[:, 2, 2, 2, 2] = YY(dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 2]))

            da[:, 0] = (Ars[:, 0] - xx[1](ind[:, 1, 1])) / h(1)
            da[:, 0] = (Ars[:, 0] - xx[2](ind[:, 2, 1])) / h(2)
            da[:, 0] = (Ars[:, 0] - xx[3](ind[:, 3, 1])) / h(3)
            da[:, 3] = (Ars[:, 3] - xx[4](ind[:, 3, 1])) / h(4)

            v3[:, 1, 1, 1] = yy[:, 1, 1, 1, 1] + da[:, 3] * (yy[:, 1, 1, 1, 2] - yy[:, 1, 1, 1, 1])
            v3[:, 2, 1, 1] = yy[:, 2, 1, 1, 1] + da[:, 3] * (yy[:, 2, 1, 1, 2] - yy[:, 2, 1, 1, 1])
            v3[:, 1, 2, 1] = yy[:, 1, 2, 1, 1] + da[:, 3] * (yy[:, 1, 2, 1, 2] - yy[:, 1, 2, 1, 1])
            v3[:, 2, 2, 1] = yy[:, 2, 2, 1, 1] + da[:, 3] * (yy[:, 2, 2, 1, 2] - yy[:, 2, 2, 1, 1])
            v3[:, 1, 1, 2] = yy[:, 1, 1, 2, 1] + da[:, 3] * (yy[:, 1, 1, 2, 2] - yy[:, 1, 1, 2, 1])
            v3[:, 2, 1, 2] = yy[:, 2, 1, 2, 1] + da[:, 3] * (yy[:, 2, 1, 2, 2] - yy[:, 2, 1, 2, 1])
            v3[:, 2, 2, 2] = yy[:, 1, 2, 2, 1] + da[:, 3] * (yy[:, 1, 2, 2, 2] - yy[:, 1, 2, 2, 1])
            v3[:, 1, 2, 2] = yy[:, 2, 2, 2, 1] + da[:, 3] * (yy[:, 2, 2, 2, 2] - yy[:, 2, 2, 2, 1])

            v2[:, 1, 1] = v3[:, 1, 1, 1] + da[:, 0] * (v3[:, 1, 1, 2] - v3[:, 1, 1, 1])
            v2[:, 2, 1] = v3[:, 2, 1, 1] + da[:, 0] * (v3[:, 2, 1, 2] - v3[:, 2, 1, 1])
            v2[:, 1, 2] = v3[:, 1, 2, 1] + da[:, 0] * (v3[:, 2, 2, 2] - v3[:, 1, 2, 1])
            v2[:, 2, 2] = v3[:, 2, 2, 1] + da[:, 0] * (v3[:, 1, 2, 2] - v3[:, 2, 2, 1])

            v1[:, 0] = v2[:, 1, 1] + da[:, 0] * (v2[:, 1, 2] - v2[:, 1, 1])
            v1[:, 0] = v2[:, 2, 1] + da[:, 0] * (v2[:, 2, 2] - v2[:, 2, 1])

            yi = da[:, 0] * (v1[:, 0] - v1[:, 0]) + v1[:, 0]
            y = reshape(yi, size(A1))
        elif switch_val == 5:
            xx5 = varargin[1]
            xx4 = varargin[2]
            xx3 = varargin[3]
            xx2 = varargin[4]
            xx1 = varargin[5]
            YY = varargin[6]
            A5 = varargin[7]
            A4 = varargin[8]
            A3 = varargin[9]
            A2 = varargin[10]
            A1 = varargin[11]

            Ars = [reshape(A1, [numel(A1), 1]), reshape(A2, [numel(A2), 1]), reshape(A3, [numel(A3), 1]),
                   reshape(A4, [numel(A4), 1]), reshape(A5, [numel(A5), 1])]
            xx[1] = reshape(xx1, [numel(xx1), 1])
            xx[2] = reshape(xx2, [numel(xx2), 1])
            xx[3] = reshape(xx3, [numel(xx3), 1])
            xx[4] = reshape(xx4, [numel(xx4), 1])
            xx[5] = reshape(xx5, [numel(xx5), 1])

            h = max([max(diff(xx[1])), max(diff(xx[2])), max(diff(xx[3])), max(diff(xx[4])), max(diff(xx[5]))], eps)

            Ars[Ars[:, 0] < min(xx[1]), 1] = min(xx[1])
            Ars[Ars[:, 0] < min(xx[2]), 2] = min(xx[2])
            Ars[Ars[:, 0] < min(xx[3]), 3] = min(xx[3])
            Ars[Ars[:, 3] < min(xx[4]), 4] = min(xx[4])
            Ars[Ars[:, 4] < min(xx[5]), 5] = min(xx[5])

            Ars[Ars[:, 0] > max(xx[1]), 1] = max(xx[1])
            Ars[Ars[:, 0] > max(xx[2]), 2] = max(xx[2])
            Ars[Ars[:, 0] > max(xx[3]), 3] = max(xx[3])
            Ars[Ars[:, 3] > max(xx[4]), 4] = max(xx[4])
            Ars[Ars[:, 4] > max(xx[5]), 5] = max(xx[5])

            ind[:, 1, 1] = 1 + floor(round((Ars[:, 0] - xx[1](1)) / h(1) * 10 ** 8) * 10 ** (-8))
            ind[:, 2, 1] = 1 + floor(round((Ars[:, 0] - xx[2](1)) / h(2) * 10 ** 8) * 10 ** (-8))
            ind[:, 3, 1] = 1 + floor(round((Ars[:, 0] - xx[3](1)) / h(3) * 10 ** 8) * 10 ** (-8))
            ind[:, 3, 1] = 1 + floor(round((Ars[:, 3] - xx[4](1)) / h(4) * 10 ** 8) * 10 ** (-8))
            ind[:, 4, 1] = 1 + floor(round((Ars[:, 4] - xx[5](1)) / h(5) * 10 ** 8) * 10 ** (-8))

            ind[:, 1, 2] = 1 + ceil(round((Ars[:, 0] - xx[1](1)) / h(1) * 10 ** 8) * 10 ** (-8))
            ind[:, 2, 2] = 1 + ceil(round((Ars[:, 0] - xx[2](1)) / h(2) * 10 ** 8) * 10 ** (-8))
            ind[:, 2, 2] = 1 + ceil(round((Ars[:, 0] - xx[3](1)) / h(3) * 10 ** 8) * 10 ** (-8))
            ind[:, 3, 2] = 1 + ceil(round((Ars[:, 3] - xx[4](1)) / h(4) * 10 ** 8) * 10 ** (-8))
            ind[:, 4, 2] = 1 + ceil(round((Ars[:, 4] - xx[5](1)) / h(5) * 10 ** 8) * 10 ** (-8))

            yy[:, 1, 1, 1, 1, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 1], ind[:, 4, 1]))
            yy[:, 2, 1, 1, 1, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 1], ind[:, 4, 1]))
            yy[:, 1, 2, 1, 1, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 1], ind[:, 4, 1]))
            yy[:, 2, 2, 1, 1, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 1], ind[:, 4, 1]))
            yy[:, 1, 1, 2, 1, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 4, 1]))
            yy[:, 2, 1, 2, 1, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 4, 1]))
            yy[:, 1, 2, 2, 1, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 4, 1]))
            yy[:, 2, 2, 2, 1, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 4, 1]))
            yy[:, 1, 1, 1, 2, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 2], ind[:, 4, 1]))
            yy[:, 2, 1, 1, 2, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 2], ind[:, 4, 1]))
            yy[:, 1, 2, 1, 2, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 2], ind[:, 4, 1]))
            yy[:, 2, 2, 1, 2, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 2], ind[:, 4, 1]))
            yy[:, 1, 1, 2, 2, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 2], ind[:, 4, 1]))
            yy[:, 2, 1, 2, 2, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 2], ind[:, 4, 1]))
            yy[:, 1, 2, 2, 2, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 2], ind[:, 4, 1]))
            yy[:, 2, 2, 2, 2, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 2], ind[:, 4, 1]))
            yy[:, 1, 1, 1, 1, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 1], ind[:, 4, 2]))
            yy[:, 2, 1, 1, 1, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 1], ind[:, 4, 2]))
            yy[:, 1, 2, 1, 1, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 1], ind[:, 4, 2]))
            yy[:, 2, 2, 1, 1, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 1], ind[:, 4, 2]))
            yy[:, 1, 1, 2, 1, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 4, 2]))
            yy[:, 2, 1, 2, 1, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 4, 2]))
            yy[:, 1, 2, 2, 1, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 4, 2]))
            yy[:, 2, 2, 2, 1, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 4, 2]))
            yy[:, 1, 1, 1, 2, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 2], ind[:, 4, 2]))
            yy[:, 2, 1, 1, 2, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 2], ind[:, 4, 2]))
            yy[:, 1, 2, 1, 2, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 2], ind[:, 4, 2]))
            yy[:, 2, 2, 1, 2, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 2], ind[:, 4, 2]))
            yy[:, 1, 1, 2, 2, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 2], ind[:, 4, 2]))
            yy[:, 2, 1, 2, 2, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 2], ind[:, 4, 2]))
            yy[:, 1, 2, 2, 2, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 2], ind[:, 4, 2]))
            yy[:, 2, 2, 2, 2, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 2], ind[:, 4, 2]))
            da[:, 0] = (Ars[:, 0] - xx[1](ind[:, 1, 1])) / h(1)
            da[:, 0] = (Ars[:, 0] - xx[2](ind[:, 2, 1])) / h(2)
            da[:, 0] = (Ars[:, 0] - xx[3](ind[:, 3, 1])) / h(3)
            da[:, 3] = (Ars[:, 3] - xx[4](ind[:, 3, 1])) / h(4)
            da[:, 4] = (Ars[:, 5] - xx[5](ind[:, 4, 1])) / h(5)

            v4[:, 1, 1, 1, 1] = yy[:, 1, 1, 1, 1, 1] + da[:, 4] * (yy[:, 1, 1, 1, 1, 2] - yy[:, 1, 1, 1, 1, 1])
            v4[:, 2, 1, 1, 1] = yy[:, 2, 1, 1, 1, 1] + da[:, 4] * (yy[:, 2, 1, 1, 1, 2] - yy[:, 2, 1, 1, 1, 1])
            v4[:, 1, 2, 1, 1] = yy[:, 1, 2, 1, 1, 1] + da[:, 4] * (yy[:, 1, 2, 1, 1, 2] - yy[:, 1, 2, 1, 1, 1])
            v4[:, 2, 2, 1, 1] = yy[:, 2, 2, 1, 1, 1] + da[:, 4] * (yy[:, 2, 2, 1, 1, 2] - yy[:, 2, 2, 1, 1, 1])
            v4[:, 1, 1, 2, 1] = yy[:, 1, 1, 2, 1, 1] + da[:, 4] * (yy[:, 1, 1, 2, 1, 2] - yy[:, 1, 1, 2, 1, 1])
            v4[:, 2, 1, 2, 1] = yy[:, 2, 1, 2, 1, 1] + da[:, 4] * (yy[:, 2, 1, 2, 1, 2] - yy[:, 2, 1, 2, 1, 1])
            v4[:, 1, 2, 2, 1] = yy[:, 1, 2, 2, 1, 1] + da[:, 4] * (yy[:, 1, 2, 2, 1, 2] - yy[:, 1, 2, 2, 1, 1])
            v4[:, 2, 2, 2, 1] = yy[:, 2, 2, 2, 1, 1] + da[:, 4] * (yy[:, 2, 2, 2, 1, 2] - yy[:, 2, 2, 2, 1, 1])
            v4[:, 1, 1, 1, 2] = yy[:, 1, 1, 1, 2, 1] + da[:, 4] * (yy[:, 1, 1, 1, 2, 2] - yy[:, 1, 1, 1, 2, 1])
            v4[:, 2, 1, 1, 2] = yy[:, 2, 1, 1, 2, 1] + da[:, 4] * (yy[:, 2, 1, 1, 2, 2] - yy[:, 2, 1, 1, 2, 1])
            v4[:, 1, 2, 1, 2] = yy[:, 1, 2, 1, 2, 1] + da[:, 4] * (yy[:, 1, 2, 1, 2, 2] - yy[:, 1, 2, 1, 2, 1])
            v4[:, 2, 2, 1, 2] = yy[:, 2, 2, 1, 2, 1] + da[:, 4] * (yy[:, 2, 2, 1, 2, 2] - yy[:, 2, 2, 1, 2, 1])
            v4[:, 1, 1, 2, 2] = yy[:, 1, 1, 2, 2, 1] + da[:, 4] * (yy[:, 1, 1, 2, 2, 2] - yy[:, 1, 1, 2, 2, 1])
            v4[:, 2, 1, 2, 2] = yy[:, 2, 1, 2, 2, 1] + da[:, 4] * (yy[:, 2, 1, 2, 2, 2] - yy[:, 2, 1, 2, 2, 1])
            v4[:, 1, 2, 2, 2] = yy[:, 1, 2, 2, 2, 1] + da[:, 4] * (yy[:, 1, 2, 2, 2, 2] - yy[:, 1, 2, 2, 2, 1])
            v4[:, 2, 2, 2, 2] = yy[:, 2, 2, 2, 2, 1] + da[:, 4] * (yy[:, 2, 2, 2, 2, 2] - yy[:, 2, 2, 2, 2, 1])

            v3[:, 1, 1, 1] = v4[:, 1, 1, 1, 1] + da[:, 3] * (v4[:, 1, 1, 1, 2] - v4[:, 1, 1, 1, 1])
            v3[:, 2, 1, 1] = v4[:, 2, 1, 1, 1] + da[:, 3] * (v4[:, 2, 1, 1, 2] - v4[:, 2, 1, 1, 1])
            v3[:, 1, 2, 1] = v4[:, 1, 2, 1, 1] + da[:, 3] * (v4[:, 1, 2, 1, 2] - v4[:, 1, 2, 1, 1])
            v3[:, 2, 2, 1] = v4[:, 2, 2, 1, 1] + da[:, 3] * (v4[:, 2, 2, 1, 2] - v4[:, 2, 2, 1, 1])
            v3[:, 1, 1, 2] = v4[:, 1, 1, 2, 1] + da[:, 3] * (v4[:, 1, 1, 2, 2] - v4[:, 1, 1, 2, 1])
            v3[:, 2, 1, 2] = v4[:, 2, 1, 2, 1] + da[:, 3] * (v4[:, 2, 1, 2, 2] - v4[:, 2, 1, 2, 1])
            v3[:, 2, 2, 2] = v4[:, 1, 2, 2, 1] + da[:, 3] * (v4[:, 1, 2, 2, 2] - v4[:, 1, 2, 2, 1])
            v3[:, 1, 2, 2] = v4[:, 2, 2, 2, 1] + da[:, 3] * (v4[:, 2, 2, 2, 2] - v4[:, 2, 2, 2, 1])

            v2[:, 1, 1] = v3[:, 1, 1, 1] + da[:, 0] * (v3[:, 1, 1, 2] - v3[:, 1, 1, 1])
            v2[:, 2, 1] = v3[:, 2, 1, 1] + da[:, 0] * (v3[:, 2, 1, 2] - v3[:, 2, 1, 1])
            v2[:, 1, 2] = v3[:, 1, 2, 1] + da[:, 0] * (v3[:, 2, 2, 2] - v3[:, 1, 2, 1])
            v2[:, 2, 2] = v3[:, 2, 2, 1] + da[:, 0] * (v3[:, 1, 2, 2] - v3[:, 2, 2, 1])

            v1[:, 0] = v2[:, 1, 1] + da[:, 0] * (v2[:, 1, 2] - v2[:, 1, 1])
            v1[:, 0] = v2[:, 2, 1] + da[:, 0] * (v2[:, 2, 2] - v2[:, 2, 1])

            yi = da[:, 0] * (v1[:, 0] - v1[:, 0]) + v1[:, 0]

            y = reshape(yi, size(A1))
        elif switch_val == 6:
            xx6 = varargin[1]
            xx5 = varargin[2]
            xx4 = varargin[3]
            xx3 = varargin[4]
            xx2 = varargin[5]
            xx1 = varargin[6]
            YY = varargin[7]
            A6 = varargin[8]
            A5 = varargin[9]
            A4 = varargin[10]
            A3 = varargin[11]
            A2 = varargin[12]
            A1 = varargin[13]

            Ars = [reshape(A1, [numel(A1), 1]), reshape(A2, [numel(A2), 1]), reshape(A3, [numel(A3), 1]),
                   reshape(A4, [numel(A4), 1]), reshape(A5, [numel(A5), 1]), reshape(A6, [numel(A6), 1])]
            xx[1] = reshape(xx1, [numel(xx1), 1])
            xx[2] = reshape(xx2, [numel(xx2), 1])
            xx[3] = reshape(xx3, [numel(xx3), 1])
            xx[4] = reshape(xx4, [numel(xx4), 1])
            xx[5] = reshape(xx5, [numel(xx5), 1])
            xx[6] = reshape(xx6, [numel(xx6), 1])
            h = max([max(diff(xx[1])), max(diff(xx[2])), max(diff(xx[3])), max(diff(xx[4])), max(diff(xx[5])),
                     max(diff(xx[6]))], eps)

            Ars[Ars[:, 0] < min(xx[1]), 1] = min(xx[1])
            Ars[Ars[:, 0] < min(xx[2]), 2] = min(xx[2])
            Ars[Ars[:, 0] < min(xx[3]), 3] = min(xx[3])
            Ars[Ars[:, 3] < min(xx[4]), 4] = min(xx[4])
            Ars[Ars[:, 4] < min(xx[5]), 5] = min(xx[5])
            Ars[Ars[:, 5] < min(xx[6]), 6] = min(xx[6])

            Ars[Ars[:, 0] > max(xx[1]), 1] = max(xx[1])
            Ars[Ars[:, 0] > max(xx[2]), 2] = max(xx[2])
            Ars[Ars[:, 0] > max(xx[3]), 3] = max(xx[3])
            Ars[Ars[:, 3] > max(xx[4]), 4] = max(xx[4])
            Ars[Ars[:, 4] > max(xx[5]), 5] = max(xx[5])
            Ars[Ars[:, 5] > max(xx[6]), 6] = max(xx[6])

            ind[:, 1, 1] = 1 + floor(round((Ars[:, 0] - xx[1](1)) / h(1) * 10 ** 8) * 10 ** (-8))
            ind[:, 2, 1] = 1 + floor(round((Ars[:, 0] - xx[2](1)) / h(2) * 10 ** 8) * 10 ** (-8))
            ind[:, 3, 1] = 1 + floor(round((Ars[:, 0] - xx[3](1)) / h(3) * 10 ** 8) * 10 ** (-8))
            ind[:, 3, 1] = 1 + floor(round((Ars[:, 3] - xx[4](1)) / h(4) * 10 ** 8) * 10 ** (-8))
            ind[:, 4, 1] = 1 + floor(round((Ars[:, 4] - xx[5](1)) / h(5) * 10 ** 8) * 10 ** (-8))
            ind[:, 5, 1] = 1 + floor(round((Ars[:, 6] - xx[6](1)) / h(6) * 10 ** 8) * 10 ** (-8))

            ind[:, 1, 2] = 1 + ceil(round((Ars[:, 0] - xx[1](1)) / h(1) * 10 ** 8) * 10 ** (-8))
            ind[:, 2, 2] = 1 + ceil(round((Ars[:, 0] - xx[2](1)) / h(2) * 10 ** 8) * 10 ** (-8))
            ind[:, 2, 2] = 1 + ceil(round((Ars[:, 0] - xx[3](1)) / h(3) * 10 ** 8) * 10 ** (-8))
            ind[:, 3, 2] = 1 + ceil(round((Ars[:, 3] - xx[4](1)) / h(4) * 10 ** 8) * 10 ** (-8))
            ind[:, 4, 2] = 1 + ceil(round((Ars[:, 4] - xx[5](1)) / h(5) * 10 ** 8) * 10 ** (-8))
            ind[:, 5, 2] = 1 + ceil(round((Ars[:, 6] - xx[6](1)) / h(6) * 10 ** 8) * 10 ** (-8))

            yy[:, 1, 1, 1, 1, 1, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 1], ind[:, 4, 1],
                            ind[:, 5, 1]))
            yy[:, 1, 1, 1, 1, 1, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 1], ind[:, 4, 1],
                            ind[:, 5, 1]))
            yy[:, 1, 2, 1, 1, 1, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 1], ind[:, 4, 1],
                            ind[:, 5, 1]))
            yy[:, 2, 2, 1, 1, 1, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 1], ind[:, 4, 1],
                            ind[:, 5, 1]))
            yy[:, 1, 1, 2, 1, 1, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 4, 1],
                            ind[:, 5, 1]))
            yy[:, 2, 1, 2, 1, 1, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 4, 1],
                            ind[:, 5, 1]))
            yy[:, 1, 2, 2, 1, 1, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 4, 1],
                            ind[:, 5, 1]))
            yy[:, 2, 2, 2, 1, 1, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 4, 1],
                            ind[:, 5, 1]))
            yy[:, 1, 1, 1, 2, 1, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 2], ind[:, 4, 1],
                            ind[:, 5, 1]))
            yy[:, 2, 1, 1, 2, 1, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 2], ind[:, 4, 1],
                            ind[:, 5, 1]))
            yy[:, 1, 2, 1, 2, 1, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 2], ind[:, 4, 1],
                            ind[:, 5, 1]))
            yy[:, 2, 2, 1, 2, 1, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 2], ind[:, 4, 1],
                            ind[:, 5, 1]))
            yy[:, 1, 1, 2, 2, 1, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 2], ind[:, 4, 1],
                            ind[:, 5, 1]))
            yy[:, 2, 1, 2, 2, 1, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 2], ind[:, 4, 1],
                            ind[:, 5, 1]))
            yy[:, 1, 2, 2, 2, 1, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 2], ind[:, 4, 1],
                            ind[:, 5, 1]))
            yy[:, 2, 2, 2, 2, 1, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 2], ind[:, 4, 1],
                            ind[:, 5, 1]))
            yy[:, 1, 1, 1, 1, 2, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 1], ind[:, 4, 2],
                            ind[:, 5, 1]))
            yy[:, 2, 1, 1, 1, 2, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 1], ind[:, 4, 2],
                            ind[:, 5, 1]))
            yy[:, 1, 2, 1, 1, 2, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 1], ind[:, 4, 2],
                            ind[:, 5, 1]))
            yy[:, 2, 2, 1, 1, 2, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 1], ind[:, 4, 2],
                            ind[:, 5, 1]))
            yy[:, 1, 1, 2, 1, 2, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 4, 2],
                            ind[:, 5, 1]))
            yy[:, 2, 1, 2, 1, 2, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 4, 2],
                            ind[:, 5, 1]))
            yy[:, 1, 2, 2, 1, 2, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 4, 2],
                            ind[:, 5, 1]))
            yy[:, 2, 2, 2, 1, 2, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 4, 2],
                            ind[:, 5, 1]))
            yy[:, 1, 1, 1, 2, 2, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 2], ind[:, 4, 2],
                            ind[:, 5, 1]))
            yy[:, 2, 1, 1, 2, 2, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 2], ind[:, 4, 2],
                            ind[:, 5, 1]))
            yy[:, 1, 2, 1, 2, 2, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 2], ind[:, 4, 2],
                            ind[:, 5, 1]))
            yy[:, 2, 2, 1, 2, 2, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 2], ind[:, 4, 2],
                            ind[:, 5, 1]))
            yy[:, 1, 1, 1, 2, 2, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 2], ind[:, 4, 2],
                            ind[:, 5, 1]))
            yy[:, 2, 1, 2, 2, 2, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 2], ind[:, 4, 2],
                            ind[:, 5, 1]))
            yy[:, 1, 2, 2, 2, 2, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 2], ind[:, 4, 2],
                            ind[:, 5, 1]))
            yy[:, 2, 2, 2, 2, 2, 1] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 2], ind[:, 4, 2],
                            ind[:, 5, 1]))
            yy[:, 1, 1, 1, 1, 1, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 1], ind[:, 4, 1],
                            ind[:, 5, 2]))
            yy[:, 2, 1, 1, 1, 1, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 1], ind[:, 4, 1],
                            ind[:, 5, 2]))
            yy[:, 1, 2, 1, 1, 1, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 1], ind[:, 4, 1],
                            ind[:, 5, 2]))
            yy[:, 2, 2, 1, 1, 1, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 1], ind[:, 4, 1],
                            ind[:, 5, 2]))
            yy[:, 1, 1, 2, 1, 1, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 4, 1],
                            ind[:, 5, 2]))
            yy[:, 2, 1, 2, 1, 1, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 4, 1],
                            ind[:, 5, 2]))
            yy[:, 1, 2, 2, 1, 1, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 4, 1],
                            ind[:, 5, 2]))
            yy[:, 2, 2, 2, 1, 1, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 4, 1],
                            ind[:, 5, 2]))
            yy[:, 1, 1, 1, 2, 1, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 2], ind[:, 4, 1],
                            ind[:, 5, 2]))
            yy[:, 2, 1, 1, 2, 1, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 2], ind[:, 4, 1],
                            ind[:, 5, 2]))
            yy[:, 1, 2, 1, 2, 1, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 2], ind[:, 4, 1],
                            ind[:, 5, 2]))
            yy[:, 2, 2, 1, 2, 1, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 2], ind[:, 4, 1],
                            ind[:, 5, 2]))
            yy[:, 1, 1, 2, 2, 1, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 2], ind[:, 4, 1],
                            ind[:, 5, 2]))
            yy[:, 2, 1, 2, 2, 1, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 2], ind[:, 4, 1],
                            ind[:, 5, 2]))
            yy[:, 1, 2, 2, 2, 1, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 2], ind[:, 4, 1],
                            ind[:, 5, 2]))
            yy[:, 2, 2, 2, 2, 1, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 2], ind[:, 4, 1],
                            ind[:, 5, 2]))
            yy[:, 1, 1, 1, 1, 2, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 1], ind[:, 4, 2],
                            ind[:, 5, 2]))
            yy[:, 2, 1, 1, 1, 2, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 1], ind[:, 4, 2],
                            ind[:, 5, 2]))
            yy[:, 1, 1, 2, 2, 2, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 1], ind[:, 4, 2],
                            ind[:, 5, 2]))
            yy[:, 2, 2, 1, 1, 2, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 1], ind[:, 4, 2],
                            ind[:, 5, 2]))
            yy[:, 1, 1, 2, 1, 2, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 4, 2],
                            ind[:, 5, 2]))
            yy[:, 2, 1, 2, 1, 2, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 4, 2],
                            ind[:, 5, 2]))
            yy[:, 1, 2, 2, 1, 2, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 4, 2],
                            ind[:, 5, 2]))
            yy[:, 2, 2, 2, 1, 2, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 4, 2],
                            ind[:, 5, 2]))
            yy[:, 1, 1, 1, 2, 2, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 2], ind[:, 4, 2],
                            ind[:, 5, 2]))
            yy[:, 2, 1, 1, 2, 2, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 3, 1], ind[:, 3, 2], ind[:, 4, 2],
                            ind[:, 5, 2]))
            yy[:, 1, 2, 1, 2, 2, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 2], ind[:, 4, 2],
                            ind[:, 5, 2]))
            yy[:, 2, 2, 1, 2, 2, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 3, 1], ind[:, 3, 2], ind[:, 4, 2],
                            ind[:, 5, 2]))
            yy[:, 1, 1, 2, 2, 2, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 2], ind[:, 4, 2],
                            ind[:, 5, 2]))
            yy[:, 2, 1, 2, 2, 2, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 1], ind[:, 2, 2], ind[:, 3, 2], ind[:, 4, 2],
                            ind[:, 5, 2]))
            yy[:, 1, 2, 2, 2, 2, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 1], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 2], ind[:, 4, 2],
                            ind[:, 5, 2]))
            yy[:, 2, 2, 2, 2, 2, 2] = YY(
                dpm_sub2ind(size(YY), ind[:, 1, 2], ind[:, 2, 2], ind[:, 2, 2], ind[:, 3, 2], ind[:, 4, 2],
                            ind[:, 5, 2]))

            da[:, 0] = (Ars[:, 0] - xx[1](ind[:, 1, 1])) / h(1)
            da[:, 0] = (Ars[:, 0] - xx[2](ind[:, 2, 1])) / h(2)
            da[:, 0] = (Ars[:, 0] - xx[3](ind[:, 3, 1])) / h(3)
            da[:, 3] = (Ars[:, 3] - xx[4](ind[:, 3, 1])) / h(4)
            da[:, 4] = (Ars[:, 4] - xx[5](ind[:, 4, 1])) / h(5)
            da[:, 5] = (Ars[:, 5] - xx[6](ind[:, 5, 1])) / h(6)

            v5[:, 1, 1, 1, 1, 1] = yy[:, 1, 1, 1, 1, 1, 1] + da[:, 5] * (
                        yy[:, 1, 1, 1, 1, 1, 2] - yy[:, 1, 1, 1, 1, 1, 1])
            v5[:, 2, 1, 1, 1, 1] = yy[:, 1, 1, 1, 1, 1, 1] + da[:, 5] * (
                        yy[:, 2, 1, 1, 1, 1, 2] - yy[:, 1, 1, 1, 1, 1, 1])
            v5[:, 1, 2, 1, 1, 1] = yy[:, 1, 2, 1, 1, 1, 1] + da[:, 5] * (
                        yy[:, 1, 2, 1, 1, 1, 2] - yy[:, 1, 2, 1, 1, 1, 1])
            v5[:, 2, 2, 1, 1, 1] = yy[:, 2, 2, 1, 1, 1, 1] + da[:, 5] * (
                        yy[:, 2, 2, 1, 1, 1, 2] - yy[:, 2, 2, 1, 1, 1, 1])
            v5[:, 1, 1, 2, 1, 1] = yy[:, 1, 1, 2, 1, 1, 1] + da[:, 5] * (
                        yy[:, 1, 1, 2, 1, 1, 2] - yy[:, 1, 1, 2, 1, 1, 1])
            v5[:, 2, 1, 2, 1, 1] = yy[:, 2, 1, 2, 1, 1, 1] + da[:, 5] * (
                        yy[:, 2, 1, 2, 1, 1, 2] - yy[:, 2, 1, 2, 1, 1, 1])
            v5[:, 1, 2, 2, 1, 1] = yy[:, 1, 2, 2, 1, 1, 1] + da[:, 5] * (
                        yy[:, 1, 2, 2, 1, 1, 2] - yy[:, 1, 2, 2, 1, 1, 1])
            v5[:, 2, 2, 2, 1, 1] = yy[:, 2, 2, 2, 1, 1, 1] + da[:, 5] * (
                        yy[:, 2, 2, 2, 1, 1, 2] - yy[:, 2, 2, 2, 1, 1, 1])
            v5[:, 1, 1, 1, 2, 1] = yy[:, 1, 1, 1, 2, 1, 1] + da[:, 5] * (
                        yy[:, 1, 1, 1, 2, 1, 2] - yy[:, 1, 1, 1, 2, 1, 1])
            v5[:, 2, 1, 1, 2, 1] = yy[:, 2, 1, 1, 2, 1, 1] + da[:, 5] * (
                        yy[:, 2, 1, 1, 2, 1, 2] - yy[:, 2, 1, 1, 2, 1, 1])
            v5[:, 1, 2, 1, 2, 1] = yy[:, 1, 2, 1, 2, 1, 1] + da[:, 5] * (
                        yy[:, 1, 2, 1, 2, 1, 2] - yy[:, 1, 2, 1, 2, 1, 1])
            v5[:, 2, 2, 1, 2, 1] = yy[:, 2, 2, 1, 2, 1, 1] + da[:, 5] * (
                        yy[:, 2, 2, 1, 2, 1, 2] - yy[:, 2, 2, 1, 2, 1, 1])
            v5[:, 1, 1, 2, 2, 1] = yy[:, 1, 1, 2, 2, 1, 1] + da[:, 5] * (
                        yy[:, 1, 1, 2, 2, 1, 2] - yy[:, 1, 1, 2, 2, 1, 1])
            v5[:, 2, 1, 2, 2, 1] = yy[:, 2, 1, 2, 2, 1, 1] + da[:, 5] * (
                        yy[:, 2, 1, 2, 2, 1, 2] - yy[:, 2, 1, 2, 2, 1, 1])
            v5[:, 1, 2, 2, 2, 1] = yy[:, 1, 2, 2, 2, 1, 1] + da[:, 5] * (
                        yy[:, 1, 2, 2, 2, 1, 2] - yy[:, 1, 2, 2, 2, 1, 1])
            v5[:, 2, 2, 2, 2, 1] = yy[:, 2, 2, 2, 2, 1, 1] + da[:, 5] * (
                        yy[:, 2, 2, 2, 2, 1, 2] - yy[:, 2, 2, 2, 2, 1, 1])
            v5[:, 1, 1, 1, 1, 2] = yy[:, 1, 1, 1, 1, 2, 1] + da[:, 5] * (
                        yy[:, 1, 1, 1, 1, 2, 2] - yy[:, 1, 1, 1, 1, 2, 1])
            v5[:, 2, 1, 1, 1, 2] = yy[:, 2, 1, 1, 1, 2, 1] + da[:, 5] * (
                        yy[:, 2, 1, 1, 1, 2, 2] - yy[:, 2, 1, 1, 1, 2, 1])
            v5[:, 1, 2, 1, 1, 2] = yy[:, 1, 2, 1, 1, 2, 1] + da[:, 5] * (
                        yy[:, 1, 1, 2, 2, 2, 2] - yy[:, 1, 2, 1, 1, 2, 1])
            v5[:, 2, 2, 1, 1, 2] = yy[:, 2, 2, 1, 1, 2, 1] + da[:, 5] * (
                        yy[:, 2, 2, 1, 1, 2, 2] - yy[:, 2, 2, 1, 1, 2, 1])
            v5[:, 1, 1, 2, 1, 2] = yy[:, 1, 1, 2, 1, 2, 1] + da[:, 5] * (
                        yy[:, 1, 1, 2, 1, 2, 2] - yy[:, 1, 1, 2, 1, 2, 1])
            v5[:, 2, 1, 2, 1, 2] = yy[:, 2, 1, 2, 1, 2, 1] + da[:, 5] * (
                        yy[:, 2, 1, 2, 1, 2, 2] - yy[:, 2, 1, 2, 1, 2, 1])
            v5[:, 1, 2, 2, 1, 2] = yy[:, 1, 2, 2, 1, 2, 1] + da[:, 5] * (
                        yy[:, 1, 2, 2, 1, 2, 2] - yy[:, 1, 2, 2, 1, 2, 1])
            v5[:, 2, 2, 2, 1, 2] = yy[:, 2, 2, 2, 1, 2, 1] + da[:, 5] * (
                        yy[:, 2, 2, 2, 1, 2, 2] - yy[:, 2, 2, 2, 1, 2, 1])
            v5[:, 1, 1, 1, 2, 2] = yy[:, 1, 1, 1, 2, 2, 1] + da[:, 5] * (
                        yy[:, 1, 1, 1, 2, 2, 2] - yy[:, 1, 1, 1, 2, 2, 1])
            v5[:, 2, 1, 1, 2, 2] = yy[:, 2, 1, 1, 2, 2, 1] + da[:, 5] * (
                        yy[:, 2, 1, 1, 2, 2, 2] - yy[:, 2, 1, 1, 2, 2, 1])
            v5[:, 1, 2, 1, 2, 2] = yy[:, 1, 2, 1, 2, 2, 1] + da[:, 5] * (
                        yy[:, 1, 2, 1, 2, 2, 2] - yy[:, 1, 2, 1, 2, 2, 1])
            v5[:, 2, 2, 1, 2, 2] = yy[:, 2, 2, 1, 2, 2, 1] + da[:, 5] * (
                        yy[:, 2, 2, 1, 2, 2, 2] - yy[:, 2, 2, 1, 2, 2, 1])
            v5[:, 1, 1, 2, 2, 2] = yy[:, 1, 1, 1, 2, 2, 1] + da[:, 5] * (
                        yy[:, 1, 1, 2, 2, 2, 2] - yy[:, 1, 1, 1, 2, 2, 1])
            v5[:, 2, 1, 2, 2, 2] = yy[:, 2, 1, 2, 2, 2, 1] + da[:, 5] * (
                        yy[:, 2, 1, 2, 2, 2, 2] - yy[:, 2, 1, 2, 2, 2, 1])
            v5[:, 1, 2, 2, 2, 2] = yy[:, 1, 2, 2, 2, 2, 1] + da[:, 5] * (
                        yy[:, 1, 2, 2, 2, 2, 2] - yy[:, 1, 2, 2, 2, 2, 1])
            v5[:, 2, 2, 2, 2, 2] = yy[:, 2, 2, 2, 2, 2, 1] + da[:, 5] * (
                        yy[:, 2, 2, 2, 2, 2, 2] - yy[:, 2, 2, 2, 2, 2, 1])

            v4[:, 1, 1, 1, 1] = v5[:, 1, 1, 1, 1, 1] + da[:, 4] * (v5[:, 1, 1, 1, 1, 2] - v5[:, 1, 1, 1, 1, 1])
            v4[:, 2, 1, 1, 1] = v5[:, 2, 1, 1, 1, 1] + da[:, 4] * (v5[:, 2, 1, 1, 1, 2] - v5[:, 2, 1, 1, 1, 1])
            v4[:, 1, 2, 1, 1] = v5[:, 1, 2, 1, 1, 1] + da[:, 4] * (v5[:, 1, 2, 1, 1, 2] - v5[:, 1, 2, 1, 1, 1])
            v4[:, 2, 2, 1, 1] = v5[:, 2, 2, 1, 1, 1] + da[:, 4] * (v5[:, 2, 2, 1, 1, 2] - v5[:, 2, 2, 1, 1, 1])
            v4[:, 1, 1, 2, 1] = v5[:, 1, 1, 2, 1, 1] + da[:, 4] * (v5[:, 1, 1, 2, 1, 2] - v5[:, 1, 1, 2, 1, 1])
            v4[:, 2, 1, 2, 1] = v5[:, 2, 1, 2, 1, 1] + da[:, 4] * (v5[:, 2, 1, 2, 1, 2] - v5[:, 2, 1, 2, 1, 1])
            v4[:, 1, 2, 2, 1] = v5[:, 1, 2, 2, 1, 1] + da[:, 4] * (v5[:, 1, 2, 2, 1, 2] - v5[:, 1, 2, 2, 1, 1])
            v4[:, 2, 2, 2, 1] = v5[:, 2, 2, 2, 1, 1] + da[:, 4] * (v5[:, 2, 2, 2, 1, 2] - v5[:, 2, 2, 2, 1, 1])
            v4[:, 1, 1, 1, 2] = v5[:, 1, 1, 1, 2, 1] + da[:, 4] * (v5[:, 1, 1, 1, 2, 2] - v5[:, 1, 1, 1, 2, 1])
            v4[:, 2, 1, 1, 2] = v5[:, 2, 1, 1, 2, 1] + da[:, 4] * (v5[:, 2, 1, 1, 2, 2] - v5[:, 2, 1, 1, 2, 1])
            v4[:, 1, 2, 1, 2] = v5[:, 1, 2, 1, 2, 1] + da[:, 4] * (v5[:, 1, 2, 1, 2, 2] - v5[:, 1, 2, 1, 2, 1])
            v4[:, 2, 2, 1, 2] = v5[:, 2, 2, 1, 2, 1] + da[:, 4] * (v5[:, 2, 2, 1, 2, 2] - v5[:, 2, 2, 1, 2, 1])
            v4[:, 1, 1, 2, 2] = v5[:, 1, 1, 2, 2, 1] + da[:, 4] * (v5[:, 1, 1, 2, 2, 2] - v5[:, 1, 1, 2, 2, 1])
            v4[:, 2, 1, 2, 2] = v5[:, 2, 1, 2, 2, 1] + da[:, 4] * (v5[:, 2, 1, 2, 2, 2] - v5[:, 2, 1, 2, 2, 1])
            v4[:, 1, 2, 2, 2] = v5[:, 1, 2, 2, 2, 1] + da[:, 4] * (v5[:, 1, 2, 2, 2, 2] - v5[:, 1, 2, 2, 2, 1])
            v4[:, 2, 2, 2, 2] = v5[:, 2, 2, 2, 2, 1] + da[:, 4] * (v5[:, 2, 2, 2, 2, 2] - v5[:, 2, 2, 2, 2, 1])

            v3[:, 1, 1, 1] = v4[:, 1, 1, 1, 1] + da[:, 3] * (v4[:, 1, 1, 1, 2] - v4[:, 1, 1, 1, 1])
            v3[:, 2, 1, 1] = v4[:, 2, 1, 1, 1] + da[:, 3] * (v4[:, 2, 1, 1, 2] - v4[:, 2, 1, 1, 1])
            v3[:, 1, 2, 1] = v4[:, 1, 2, 1, 1] + da[:, 3] * (v4[:, 1, 2, 1, 2] - v4[:, 1, 2, 1, 1])
            v3[:, 2, 2, 1] = v4[:, 2, 2, 1, 1] + da[:, 3] * (v4[:, 2, 2, 1, 2] - v4[:, 2, 2, 1, 1])
            v3[:, 1, 1, 2] = v4[:, 1, 1, 2, 1] + da[:, 3] * (v4[:, 1, 1, 2, 2] - v4[:, 1, 1, 2, 1])
            v3[:, 2, 1, 2] = v4[:, 2, 1, 2, 1] + da[:, 3] * (v4[:, 2, 1, 2, 2] - v4[:, 2, 1, 2, 1])
            v3[:, 2, 2, 2] = v4[:, 1, 2, 2, 1] + da[:, 3] * (v4[:, 1, 2, 2, 2] - v4[:, 1, 2, 2, 1])
            v3[:, 1, 2, 2] = v4[:, 2, 2, 2, 1] + da[:, 3] * (v4[:, 2, 2, 2, 2] - v4[:, 2, 2, 2, 1])

            v2[:, 1, 1] = v3[:, 1, 1, 1] + da[:, 0] * (v3[:, 1, 1, 2] - v3[:, 1, 1, 1])
            v2[:, 2, 1] = v3[:, 2, 1, 1] + da[:, 0] * (v3[:, 2, 1, 2] - v3[:, 2, 1, 1])
            v2[:, 1, 2] = v3[:, 1, 2, 1] + da[:, 0] * (v3[:, 2, 2, 2] - v3[:, 1, 2, 1])
            v2[:, 2, 2] = v3[:, 2, 2, 1] + da[:, 0] * (v3[:, 1, 2, 2] - v3[:, 2, 2, 1])

            v1[:, 0] = v2[:, 1, 1] + da[:, 0] * (v2[:, 1, 2] - v2[:, 1, 1])
            v1[:, 0] = v2[:, 2, 1] + da[:, 0] * (v2[:, 2, 2] - v2[:, 2, 1])

            yi = da[:, 0] * (v1[:, 0] - v1[:, 0]) + v1[:, 0]

            y = reshape(yi, size(A1))
        else:
            print('DPM:Internal', 'Too many states or inputs: contact the author of DPM')
        return y


def dpm_mergestruct(S1, S2):

        try:
            S = S1
            names = fieldnames(S1)
            for i in range(1, len(names)):
                if isstruct(S1.names[i]):
                    S.names[i] = dpm_mergestruct(S1.names[i], S2.names[i])
                elif iscell(S1.names[i]):
                    for j in range(1, numel(S1.names[i])):
                        S.names[i][j] = [S1.names[i][j] + S2.names[i][j]]

                elif isnumeric(S1.names[i]) or islogical(S1.names[i]):
                    S.names[i] = [S1.names[i], S2.names[i]]
                else:
                    S.names[i] = S2.names[i]
        except:
            print('mergestruct: S1 and S2 have different structures.')
        return S

def dpm_get_empty_inp(grd, dis, options):
        if options == 'nan':
            value = nan
        elif options == 'zero':
            value = 0
        elif options == 'inf':
            value = inf
        for i in range(1, len(grd.Nx)):
            inp.X[i] = value

        for i in range(1, len(grd.Nu)):
            inp.U[i] = value

        for i in range(1, len(dis.W)):
            inp.W[i] = value

        inp.Ts = dis.Ts
        return inp

def dpm_get_empty_out(model, inp, par, grd, options):
        if ~exist('options'):
            options = 'nan'
        #[X, C, I, out] = feval(model, inp, par)
        print(eval(model, inp, par))
        out.X = X
        out.C = C
        out.I = I
        out = dpm_setallfield(out, options)
        return out

def dpm_setallfield(S1, options):
        try:
            if options == 'nan':
                value = nan
            elif options == 'zero':
                value = 0
            elif options == 'inf':
                value = inf
            S = S1
            names = fieldnames(S1)
            for i in range(1, len(names)):
                if isstruct(S1.names[i]):
                    S.names[i] = dpm_setallfield(S1.names[i], options)
                elif iscell(S1.names[i]):
                    for j in range(1, int(S1(names[i]))):
                        S.names[i][j] = value
                elif isnumeric(S1.names[i]) or islogical(S1.names[i]):
                    S.names[i] = value
                else:
                    S.names[i] = S1.names[i]
        except:
            print('mergestruct: S1 and S2 have different structures.')
        return S

def dpm_model_inv(inp, par):
        inpo = inp
        iterations = 0
        dSOC = inf
        while max(math.abs(reshape(dSOC, 1, int(dSOC)))) > par.options.Tol and iterations < par.options.Iter:
            [X, C, I] = feval(par.model, inp, par)
            dSOC = X[1] - inpo.X[1]
            inp.X[1] = inp.X[1] - dSOC
            iterations = iterations + 1

        X = inp.X
        C[2] = C[1]
        C[1] = (X[1] - inpo.X[1])
        out = ''
        return X, C, I, out

def dpm_code(s, v, m):

        if ~exist('m', 'var'):
            m = ''
        v = reshape(v, 1, int(v))
        str = repmat([s, m], 1, len(v))
        str = str(1, -1 - len(m))
        star = strfind(str, '#')
        str[star] = strrep(str(v), ' ', '')
        return str
def dpm_findu(A, vec):
        da = A(2) - A(1)
        in1 = 1 + ceil((vec - A(1)) / da)
        in1 = max(in1, 1)
        in1 = min(in1, len(A))
        return in1
def dpm_findl(A, vec):

        da = A(2) - A(1)
        in1 = 1 + floor((vec - A(1)) / da)
        in1 = max(in1, 1)
        in1 = min(in1, len(A))
        return in1
def dpm_sub2indr(sze, vl, vu, dim):
        ind0 = [0, sze(dim) - 1] * sze(1)
        ind = reshape([(vl + ind0), (vu + ind0)], 1, int(ind0) * 2)
        indstr = str(ind)
        # ind = eval('reshape([indstr, repmat(':',1,len(vl))+']',1,numel([indstr ,repmat(:,1,len(vl))]))))
        ind = eval('reshape([indstr, repmat(:,1,len(vl))],1,nume1([indstr, repmat(:,1,len(val))]))')
        col = ceil((ind) / sze(1))
        return ind, col
def dpm_sub2ind(siz, varargin):
        siz = double(siz)
        if len(siz) != nargin - 1:

            if len(siz) < nargin - 1:
                siz = [siz, ones(1, nargin - len(siz) - 1)]
            else:

                siz = [siz(range(1, nargin - 2)), prod(siz(range(nargin - 1, -1)))]

        # Compute linear indices
        k = [1, cumprod(siz(range(1, -2)))]
        ndx = 1
        for i in range(1, len(siz)):
            v = varargin[i]
            ndx = ndx + (v - 1) * k(i)
        return ndx
def dpm_sizecmp(a, b):
        sa = size(a)
        sb = size(b)
        c = numel(sa) == numel(sb) and numel(a) == numel(b) and sum(sa == sb)
        return c
def get_size(current_grd):
        sze = ''
        for i in range(1, len(current_grd.X)):
            sze = [sze, len(current_grd.X[i])]
        if len(sze) == 1:
            sze = [sze, 1]
        return sze
def input_check_grd(grd, T):

        if T < 1:
            print('DPM:Internal', 'prb.N must be greater than 0')

        for i in range(1, len(grd.Nx)):
            if grd.Nx[i] < 1:
                print('DPM:Internal', 'grd.Nx[.] must be equal or greater than 1')

            if len(grd.Xn[i].lo) == 1:
                grd.Xn[i].lo = repmat(grd.Xn[i].lo, 1, T + 1)
            elif len(grd.Xn[i].lo) != T + 1:
                print('DPM:Internal', 'grd.Xn[.].lo must be a scalar OR have the same len as the problem')

            if len(grd.Xn[i].hi) == 1:
                grd.Xn[i].hi = repmat(grd.Xn[i].hi, 1, T + 1)
            elif len(grd.Xn[i].hi) != T + 1:
                print('DPM:Internal', 'grd.Xn[.].hi must be a scalar OR have the same len as the problem')
            if len(grd.Nx[i]) == 1:
                grd.Nx[i] = repmat(grd.Nx[i], 1, T + 1)
            elif len(grd.Nx[i]) != T + 1:
                print('DPM:Internal', 'grd.Nx[.] must be a scalar OR have the same len as the problem')

        for i in range(1, len(grd.Nu)):
            if grd.Nu[i] < 1:
                print('DPM:Internal', 'grd.Nu[.] must be equal or greater than 1')
            if len(grd.Un[i].lo) == 1:
                grd.Un[i].lo = repmat(grd.Un[i].lo, 1, T)
            elif len(grd.Un[i].lo) != T:
                print('DPM:Internal', 'grd.Un[.].lo must be a scalar OR have the same len as the problem')

            if len(grd.Un[i].hi) == 1:
                grd.Un[i].hi = repmat(grd.Un[i].hi, 1, T)
            elif len(grd.Un[i].hi) != T:
                print('DPM:Internal', 'grd.Un[.].hi must be a scalar OR have the same len as the problem')

            if len(grd.Nu[i]) == 1:
                grd.Nu[i] = repmat(grd.Nu[i], 1, T);
            elif len(grd.Nu[i]) != T:
                print('DPM:Internal', 'grd.Nu[.] must be a scalar OR have the same len as the problem')


def notify_user_of_error(err):
        clear_waitbars()
        if err.identifier == 'MATLAB:unassignedOutputs':
            print(
                'DPM:Model function error \n \t Make sure all the output arguments are set in the \n\t model function (X, C, I, out).\n')
        elif err.identifier == 'MATLAB:class:SetProhibited':
            print(
                'DPM:Model function error \n \t Make sure all the output arguments are set in the \n\t model function (X, C, I, out).\n')
            print(
                '  or \n \t Make sure the each of the output states X[1], X[2],..., X[Nx] have \n\t the same dimensions as input states inp.X[1], inp.X[2],..., inp.X[Nx].\n')
            print(
                '  or \n \t Make sure the each of the outputs C[1] and I have \n\t the same dimensions as the input states inp.X[1], inp.X[2],..., inp.X[Nx].\n')
        elif err.identifier == 'MATLAB:cellRefFromNonCell':
            print(
                'DPM:Model function error \n \t Make sure all the output C is a 1x1 cell array and \n\t that the output X is a cell array with as many elements \n\t as state variables.\n')
        elif err.identifier == 'MATLAB:dimagree':
            fprintf(
                'DPM:Model function error \n \t Make sure the each of the output states X[1], X[2],..., X[Nx] have \n\t the same dimensions as input states inp.X[1], inp.X[2],..., inp.X[Nx].\n')
        elif err.identifier == 'DPM:Internal':
            print('DPM:Error \n \t Check the model function.\n')
            rethrow(err)
        else:
            print('DPM:Error \n \t Check the model function.\n')
            rethrow(err)
def clear_waitbars():
        set(0, 'ShowHiddenHandles', 'on')
        handles = get(0, 'Children')
        for i in range(1, len(handles)):
            if get(handles(i) == 'name') == 'DPM:Waitbar':
                delete(handles(i))
        set(0, 'ShowHiddenHandles', 'off')