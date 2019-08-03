
module Testing

using ForwardDiff

import JuLIPMaterials
using JuLIPMaterials: Vec3, Mat3, Ten33, Ten43,
                             MVec3, MMat3, MTen33, MTen43
using StaticArrays, LinearAlgebra

CLE = JuLIPMaterials.CLE

export randvec3, randmoduli, cleforce


function randvec3(s0=1.0, s1=2.0)
   x = rand(3) .- 0.5
   return (x / norm(x)) * (s0 + rand() * (s1-s0))
end

function randmoduli(rnd = 0.1)
   Cv = [ 3.0 1.5 1.5 0.0 0.0 0.0
          1.5 3.0 1.5 0.0 0.0 0.0
          1.5 1.5 3.0 0.0 0.0 0.0
          0.0 0.0 0.0 1.2 0.0 0.0
          0.0 0.0 0.0 0.0 1.2 0.0
          0.0 0.0 0.0 0.0 0.0 1.2 ]
   Cv += Symmetric( 2*rnd*(rand(6, 6) .- 0.5) )
   return CLE.elastic_moduli(Cv)
end

# div C ∇u = C_iajb u_j,ab
function cleforce(x::Vec3{T}, u, C) where T
    f = Vec3(0.0,0.0,0.0)
    for j = 1:3
        ujab = ForwardDiff.hessian(y->u(y)[j], x)
        Cj = reshape(C[:,:,j,:], 3, 9)
        f += Cj * ujab[:]
    end
    return f
end


end
