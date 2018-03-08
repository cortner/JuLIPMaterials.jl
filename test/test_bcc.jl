using JuLIPMaterials
using Base.Test
using JuLIP, JuLIP.Potentials
using PyPlot
MST = JuLIPMaterials
BCC = MST.BCC

# data = Pkg.dir("JuLIP") * "/data/"
# eam_W = JuLIP.Potentials.FinnisSinclair(data*"W-pair-Wang-2014.plt", data*"W-e-dens-Wang-2014.plt")

morse = Morse(A = 3.5, r0 = rnn("W")) * C2Shift(2.3 * rnn("W"))
at = bulk("W")
set_constraint!(at, VariableCell(at))
set_calculator!(at, morse)

s = "W"
at1 = BCC.screw_111.(s, 70.1, bsign = 1, soln =:anisotropic, calc=morse)
x, y, z1 = xyz(at1)
scatter(x,y,c=z1)

set_calculator!(at1,morse)
f = norm.(forces(at1))
R = norm.(positions(at1))
loglog(R,f,".")
loglog(R,R.^-2,".")
loglog(R,R.^-1,".")

cell(at1)
BCC.lattice_constants_111("W", calc=morse)
