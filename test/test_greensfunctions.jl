

using JuLIP, JuLIP.Potentials
import MaterialsScienceTools

MST = MaterialsScienceTools
BCC = MST.BCC
EL = MST.Elasticity
GR = MST.GreensFunctions

"construct an EAM potential for W (tungsten)"
function EAMW()
    data = Pkg.dir("JuLIP") * "/data/"
    return JuLIP.Potentials.FinnisSinclair(data*"W-pair-Wang-2014.plt",
                                           data*"W-e-dens-Wang-2014.plt")
end

function unitcell()
    at0 = bulk("W", cubic=true, pbc = true)
    set_calculator!(at0, EAMW())
   #  set_constraint!(at0, VariableCell(at0))
   #  JuLIP.Solve.minimise!(at0)
   #  set_constraint!(at0, FixedCell(at0))
    return at0
end

at0 = unitcell()
C = EL.elastic_moduli(at0)
C = round(C, 9)

∷(C::Array{Float64, 3}, F::Matrix) = reshape(C, 3, 9) * F[:]

# div C ∇u = C_iajb u_j,ab
function cleforce(x, u, C)
   @show typeof(x)
    f = zeros(3)
    for j = 1:3
        ujab = ForwardDiff.hessian(y->u(y)[j], x)
        f += C[:,:,j,:] ∷ ujab
    end
    return JVecF(f)
end


for n = 1:10
   a = rand(3) - 0.5
   a = a / norm(a)
   u = x_ -> GR.GreenTensor3D(x_, C)
   x = (rand(3) - 0.5) * 10.0
   @show cleforce(x, u, C)
end
