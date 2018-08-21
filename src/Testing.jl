
module Testing

using ForwardDiff

import JuLIPMaterials
using JuLIPMaterials: Vec3, Mat3, Ten33, Ten43,
                             MVec3, MMat3, MTen33, MTen43

CLE = JuLIPMaterials.CLE

export randvec3, randmoduli, cleforce


function randvec3(s0=1.0, s1=2.0)
   x = rand(3) - 0.5
   return (x / norm(x)) * (s0 + rand() * (s1-s0))
end

function randmoduli(rnd = 0.1)
   Cv = [ 3.0 1.5 1.5 0.0 0.0 0.0
          1.5 3.0 1.5 0.0 0.0 0.0
          1.5 1.5 3.0 0.0 0.0 0.0
          0.0 0.0 0.0 1.2 0.0 0.0
          0.0 0.0 0.0 0.0 1.2 0.0
          0.0 0.0 0.0 0.0 0.0 1.2 ]
   Cv += Symmetric( 2*rnd*(rand(6, 6)-0.5) )
   return CLE.elastic_moduli(Cv)
end

∷(C::Array{Float64, 3}, F::Matrix) = reshape(C, 3, 9) * F[:]
∷(C::Ten33{Float64}, F::Matrix) = reshape(C, 3, 9) * F[:]

# div C ∇u = C_iajb u_j,ab
function cleforce(x, u, C)
    f = zeros(3)
    for j = 1:3
        ujab = ForwardDiff.hessian(y->u(y)[j], x)
        f += C[:,:,j,:] ∷ ujab
    end
    return Vec3(f)
end


end
