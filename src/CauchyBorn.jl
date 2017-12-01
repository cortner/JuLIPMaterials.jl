

module CauchyBorn

using JuLIP
using JuLIP.Potentials: StillingerWeber

import MaterialsScienceTools.CLE: elastic_moduli

export WcbQuad


"a fully equilibrated SW potential"
function sw_eq()
    T(σ, at) = trace(stress(StillingerWeber(σ=σ), at))
    at = JuLIP.ASE.bulk("Si", pbc=true)
    r0 = 2.09474
    r1 = r0 - 0.1
    s0, s1 = T(r0, at), T(r1, at)
    while (abs(s1) > 1e-8) && abs(r0 - r1) > 1e-8
        rnew = (r0 * s1 - r1 * s0) / (s1 - s0)
        r0, r1 = r1, rnew
        s0, s1 = s1, T(rnew, at)
    end
#     @show r1
    return StillingerWeber(σ=r1)
end


function DpWcb(F, p, at = bulk("Si", pbc=true), sw = sw_eq())
    set_defm!(at, F)
    X = positions(at)
    X[2] = X[1] + p
    set_positions!(at, X)
    return -forces(sw, at)[2]
end


function DpDpWcb()
    at = bulk("Si", pbc=true)
    sw = sw_eq()
    F0 = defm(at)
    p0 = positions(at)[2] |> Vector
    h = 1e-5
    DpDpW = zeros(3, 3)
    for i = 1:3
        p0[i] += h
        DpW1 = DpWcb(F0, JVecF(p0), at, sw)
        p0[i] -= 2*h
        DpW2 = DpWcb(F0, JVecF(p0), at, sw)
        p0[i] += h
        DpDpW[:, i] = (DpW1 - DpW2) / (2*h)
    end
    return 0.5 * (DpDpW + DpDpW')
end


type WcbQuad
    DpDpW::Matrix{Float64}
    DpDpW_inv::Matrix{Float64}
    at::AbstractAtoms
    sw::StillingerWeber
    F0
end

function WcbQuad()
    DpDpW = DpDpWcb()
    at = bulk("Si", pbc=true)
    return WcbQuad(DpDpW, pinv(DpDpW), at, sw_eq(), defm(at))
end

function (W::WcbQuad)(F)
    p0 = positions(W.at)[2]
    p1 = p0 - W.DpDpW_inv * DpWcb(F, p0, W.at, W.sw)
    p2 = p1 - W.DpDpW_inv * DpWcb(F, p1, W.at, W.sw)
    return p2
end



function DW_infp(W::WcbQuad, F)
    p = W(F)
    at = W.at
    set_defm!(at, F)
    X = positions(at)
    X[2] = p
    set_positions!(at, X)
    return stress(W.sw, at)
end


function elastic_moduli(W::WcbQuad)
   F0 = W.F0 |> Matrix
   Ih = eye(3)
   h = eps()^(1/3)
   C = zeros(3,3,3,3)
   for i = 1:3, a = 1:3
      Ih[i,a] += h
      Sp = DW_infp(W, Ih * F0)
      Ih[i,a] -= 2*h
      Sm = DW_infp(W, Ih * F0)
      C[i, a, :, :] = (Sp - Sm) / (2*h)
      Ih[i,a] += h
   end
   # symmetrise it - major symmetries C_{iajb} = C_{jbia}
   for i = 1:3, a = 1:3, j=1:3, b=1:3
      t = 0.5 * (C[i,a,j,b] + C[j,b,i,a])
      C[i,a,j,b] = t
      C[j,b,i,a] = t
   end
   # minor symmetries - C_{iajb} = C_{iabj}
   for i = 1:3, a = 1:3, j=1:3, b=1:3
      t = 0.5 * (C[i,a,j,b] + C[i,a,b,j])
      C[i,a,j,b] = t
      C[i,a,b,j] = t
   end
   return C
end

end
