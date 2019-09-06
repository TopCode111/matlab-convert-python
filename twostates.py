import math
def twostatesys(inp, par):
    et = exp(-0.5*inp.Ts)
    X[1] = et*inp.X[1] - 2*(et-1)*inp.U[1]*inp.U[2]
    X[2] = et*inp.X[2] - 2*(et-1)*inp.U[1]*(1-inp.U[2])

    # Cost
    C[1] = (inp.U[1] + 0.1*math.fabs(inp.U[2]-0.5))*inp.Ts

    # Feasibility
    I = zeros(size(X[1]))
    # constraints taken care of by dpm.m
    # Output
    out.u1 = inp.U[1]
    out.u2 = inp.U[2]
    return X,C,I,out
