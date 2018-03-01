

module CauchyBorn

using JuLIP, StaticArrays

import MaterialsScienceTools.CLE: elastic_moduli

using ..Vec3, ..Mat3
# const Ten33{T} = SArray{Tuple{3,3,3},T,3,27}
# const Ten43{T} = SArray{NTuple{4,3},T,4,81}
# const MVec3{T} = MVector{3, T}
# const MMat3{T} = MMatrix{3,3,T}
# const MTen33{T} = MArray{Tuple{3,3,3},T,3,27}
# const MTen43{T} = MArray{NTuple{4,3},T,4,81}


export Wcb

"""
`struct Wcb` : Cauchy--Born potential

Construct using
```
W = Wcb(at)
W = Wcb(at, calc)
```
This pre-computes several partial derivatives of W in the reference state.
To prevent this precomputation, use the kw-argument `precompute=false`.

Then `W` can be used to evaluate the following quantifies:

## For a Simple Lattice (unit cell contains 1 atom)

* `W(F)` : with `F` a 3 x 3 matrix is the energy / undeformed volume;
* `grad(W, F)` : with `F` a 3 x 3 matrix is the first Piola-Kirchhoff stress at
`F`, i.e., the jacobian of `W(F)` w.r.t. `F`
* `div_grad(W,F)` : finite-difference implementation of div ∂W(F)
"""
mutable struct Wcb{N, TA, TC, T}
   at::TA                # unit cell
   C0::Mat3{T}           # original cell matrix (cell(at))
   X0::Vector{Vec3{T}}   # original positions
   calc::TC              # calculator
   dpW::Vector{T}        # ------ quadratic expansion of W
   dFW::Matrix{T}        #
   dpdpW::Matrix{T}      #
   dpdpW_inv::Matrix{T}  #
   dpdFW::Array{T, 3}    #
   dFdFW::Array{T, 4}    #
   nat::Val{N}           # something to allow for efficient dispatch
end


Wcb(at::AbstractAtoms) = Wcb(at, calculator(at))

function Wcb(at::AbstractAtoms, calc::AbstractCalculator; precompute=true)
   @assert length(at) == 1
   const T = Float64
   set_calculator!(at, calc)
   if precompute && length(at) > 1
      dpdpW = dpdpWcb(at)
      dpdpW_inv = pinv(dpdpW)
   else
      dpdpW = Matrix{T}(0,0)
      dpdpW_inv = Matrix{T}(0,0)
   end
   return Wcb(at, Mat3(cell(at)), positions(at), calc,
              Vector{T}(0), Matrix{T}(0,0),
              dpdpW, dpdpW_inv, Array{T, 3}(0,0,0), Array{T, 4}(0,0,0,0),
              Val(length(at)))
end

# ================ Simple Lattice Cauchy--Born  ==============


(W::Wcb)(args...) = evaluate(W::Wcb, args...)

evaluate(W::Wcb{1}, F) = energy( set_defm!(W.at, F * W.C0') ) / det(W.C0)

grad(W::Wcb{1}, F) = (- virial( set_defm!(W.at, F * W.C0') ) * inv(F)') / det(W.C0)


function div_grad(W::Wcb{1}, F, x::Vec3{T}) where T
   h = 1e-5
   E = @SMatrix eye(3)
   return sum( (grad(W, F(x+h*E[:,i]))[:,i] - grad(W, F(x-h*E[:,i]))[:,i]) / (2*h)
               for i = 1:3 )
end


# ============= Single Species 2-Lattice =================
#
# This is a special case that

#
# # function evaluate_newton(W::Wcb, F)
# # end
#
#
# function DpWcb(F, p, at, calc)
#     @assert length(at) == 2
#     set_defm!(at, F)
#     X = positions(at)
#     X[2] = X[1] + p
#     set_positions!(at, X)
#     return -forces(calc, at)[2]
# end
#
#
# function dpdpWcb(at)
#     calc = calculator(at)
#     F0 = defm(at)
#     p0 = positions(at)[2] |> Vector
#     h = 1e-5
#     dpdpW = zeros(3, 3)
#     for i = 1:3
#         p0[i] += h
#         DpW1 = DpWcb(F0, JVecF(p0), at, calc)
#         p0[i] -= 2*h
#         DpW2 = DpWcb(F0, JVecF(p0), at, calc)
#         p0[i] += h
#         dpdpW[:, i] = (DpW1 - DpW2) / (2*h)
#     end
#     return 0.5 * (dpdpW + dpdpW')
# end
#
#
#
# # TODO: replace with get_shift
# function (W::WcbQuad)(F)
#     # TODO this is fishy - why is the initial position not reset?
#     p0 = positions(W.at)[2]
#     p1 = p0 - W.dpdpW_inv * DpWcb(F, p0, W.at, W.calc)
#     p2 = p1 - W.dpdpW_inv * DpWcb(F, p1, W.at, W.calc)
#     return p2
# end
#
#
# function DW_infp(W::WcbQuad, F)
#     p = W(F)
#     at = W.at
#     set_defm!(at, F)
#     X = positions(at)
#     X[2] = p
#     set_positions!(at, X)
#     return stress(W.calc, at)
# end
#
#
# function elastic_moduli(W::WcbQuad)
#    F0 = W.F0 |> Matrix
#    Ih = eye(3)
#    h = eps()^(1/3)
#    C = zeros(3,3,3,3)
#    for i = 1:3, a = 1:3
#       Ih[i,a] += h
#       Sp = DW_infp(W, Ih * F0)
#       Ih[i,a] -= 2*h
#       Sm = DW_infp(W, Ih * F0)
#       C[i, a, :, :] = (Sp - Sm) / (2*h)
#       Ih[i,a] += h
#    end
#    # symmetrise it - major symmetries C_{iajb} = C_{jbia}
#    for i = 1:3, a = 1:3, j=1:3, b=1:3
#       t = 0.5 * (C[i,a,j,b] + C[j,b,i,a])
#       C[i,a,j,b] = t
#       C[j,b,i,a] = t
#    end
#    # minor symmetries - C_{iajb} = C_{iabj}
#    for i = 1:3, a = 1:3, j=1:3, b=1:3
#       t = 0.5 * (C[i,a,j,b] + C[i,a,b,j])
#       C[i,a,j,b] = t
#       C[i,a,b,j] = t
#    end
#    return C
# end


# =============== GENERAL MULTI-LATTICE FUNCTIONALITY ================

# TODO
end
