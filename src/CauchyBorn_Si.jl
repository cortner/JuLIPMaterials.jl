

module CauchyBornSi

using JuLIP, LinearAlgebra
using JuLIP.Potentials: StillingerWeber
import JuLIPMaterials.CLE: elastic_moduli

export WcbQuad, get_shift


function DpWcb(F, p, at, calc)
    @assert length(at) == 2
    set_cell!(at, F')
    X = positions(at)
    X[2] = X[1] + p
    set_positions!(at, X)
    return -forces(calc, at)[2]
end


function DpDpWcb(at)
    calc = calculator(at)
    F0 = cell(at)'
    p0 = positions(at)[2] |> Vector
    h = 1e-5
    DpDpW = zeros(3, 3)
    for i = 1:3
        p0[i] += h
        DpW1 = DpWcb(F0, JVecF(p0), at, calc)
        p0[i] -= 2*h
        DpW2 = DpWcb(F0, JVecF(p0), at, calc)
        p0[i] += h
        DpDpW[:, i] = (DpW1 - DpW2) / (2*h)
    end
    return 0.5 * (DpDpW + DpDpW')
end


mutable struct WcbQuad{TA, TF, TC}
    DpDpW::Matrix{Float64}
    DpDpW_inv::Matrix{Float64}
    at::TA
    calc::TC
    F0::TF
end

function WcbQuad(calc)
    at = bulk(:Si, pbc=true)
    set_calculator!(at, calc)
    DpDpW = DpDpWcb(at)
    return WcbQuad(DpDpW, pinv(DpDpW), at, calc, cell(at)')
end

# TODO: replace with get_shift
function (W::WcbQuad)(F)
    # TODO this is fishy - why is the initial position not reset?
    error("this functionality has been deprecated -> use `get_shift`")
end

function get_shift(W::WcbQuad, F)
    p0 = positions(W.at)[2]
    p1 = p0 - W.DpDpW_inv * DpWcb(F, p0, W.at, W.calc)
    p2 = p1 - W.DpDpW_inv * DpWcb(F, p1, W.at, W.calc)
    return p2
end

function DW_infp(W::WcbQuad, F)
    p = get_shift(W, F)
    at = W.at
    set_cell!(at, F')
    X = positions(at)
    X[2] = p
    set_positions!(at, X)
    return stress(W.calc, at)
end


function elastic_moduli(W::WcbQuad)
   F0 = W.F0 |> Matrix
   Ih = Matrix(1.0*I, 3, 3)
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
