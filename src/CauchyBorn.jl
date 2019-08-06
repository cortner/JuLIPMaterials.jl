

module CauchyBorn

using JuLIP, StaticArrays, Calculus, LinearAlgebra

import JuLIPMaterials.CLE: elastic_moduli

using JuLIPMaterials: ee, Vec3, Mat3
using JuLIP: AbstractAtoms, AbstractCalculator

abstract type Wcb end

export Wcb


Wcb(at::AbstractAtoms) = Wcb(at, calculator(at))

"""
Construct using
```
W = Wcb(at)
W = Wcb(at, calc)
```

## KW Arguments

* `precompute=true`: This (possibly) pre-computes several partial derivatives
of W in the reference state. To prevent this precomputation use the kw-argument
`precompute=false`.
* `normalise = :volume`: this normalises W against the volume of the unit cell.
Alternatively, to normalise against the number of atoms use `normalise = :atoms`.
Or, simply pass in a scalar `normalise = v0`.


## Usage

`W` can be used to evaluate the following quantities:

### For a Simple Lattice (unit cell contains 1 atom)

* `W(F)` : with `F` a 3 x 3 matrix is the energy / undeformed volume;
* `grad(W, F)` : with `F` a 3 x 3 matrix is the first Piola-Kirchhoff stress at
`F`, i.e., the jacobian of `W(F)` w.r.t. `F`
* `div_grad(W,F)` : finite-difference implementation of div ∂W(F)
"""
function Wcb(at::AbstractAtoms, calc::AbstractCalculator;
             normalise=:volume, T = Float64, kwargs...)
   set_calculator!(at, calc)
   # compute the normalisation factor; volume or what?
   if normalise == :volume
      v0 = det(cell(at))
   elseif normalise == :atoms
      v0 = T(length(at))
   elseif normalise isa Number
      v0 = T(normalise)
   else
      error("Wcb: unrecognised kwarg `normalise = $normalise`")
   end
   # simple lattice case
   if length(at) == 1
      return Wcb1(at, v0; kwargs...)
   elseif length(at) == 2 && length(unique(chemical_symbols(at))) == 1
      return Wcb2(at, v0; kwargs...)
   else
      error("""`Wcb`: so far, only single species 1-lattice and 2-lattice
      have been implemented. If you need a more general case, please file
      an issue at https://github.com/cortner/JuLIPMaterials.jl""")
   end
end


set_rel_defm!(W::Wcb, F) = set_cell!(W.at, (F * W.C0')')


# ================ Simple Lattice Cauchy--Born  ==============

"""
`struct Wcb1` : Simple Lattice Cauchy--Born potential, see documentation
for `Wcb`
"""
struct Wcb1{TA, TC, T} <: Wcb
   calc::TC
   at::TA        # unit cell
   C0::Mat3{T}   # original cell matrix (cell(at))
   v0::T         # volume of original cell
end

Wcb1(at::AbstractAtoms, v0) = Wcb1(calculator(at), at, Mat3(cell(at)), v0)

(W::Wcb1)(args...) = evaluate(W, args...)

evaluate(W::Wcb1, F) = energy( W.calc, set_rel_defm!(W, F) ) / W.v0

grad(W::Wcb1, F) = (- virial( W.calc, set_rel_defm!(W, F) ) * inv(F)') / W.v0

div_grad(W::Wcb1, F, x::Vec3{T}; h = 1e-5) where T =
      sum( (grad(W, F(x+h*ee(i)))[:,i] - grad(W, F(x-h*ee(i)))[:,i]) / (2*h)
            for i = 1:3 )


# ============= Single Species 2-Lattice =================

"""
`struct Wcb2` : single-species 2-lattice Cauchy--Born potential,
see documentation for `Wcb`
"""
struct Wcb2{T, TA}
   at::TA        # unit cell
   C0::Mat3{T}   # original cell matrix (cell(at))
   v0::T         # volume of original cell
   p0::Vec3{T}
   p1::Vec3{T}
   dpdpW::Mat3{T}
   dpdpW_inv::Mat3{T}
end

function Wcb2(at::AbstractAtoms; precompute = true)
   @assert length(at) == 2 && length(unique(chemical_symbols(at))) == 1
   if precompute
      dpdpW = dpdpWcb2(at)
      dpdpW_inv = inv(dpdpW)
   else
      dpdpW = zero(Mat3{T})
      dpdpW_inv = zero(Mat3{T})
   end
   X = positions(at)
   return Wcb2(at, Mat3(cell(at)), det(cell(at)), X[1], X[2], dpdpW, dpdpW_inv)
end

set_F_and_p!(W::Wcb2, F, p) = set_positions!(set_rel_defm!(W, F), [W.p0, W.p0+p])

evaluate(W::Wcb2, F, p) = energy( set_F_and_p!(W, F, p) )

grad_p(W::Wcb2, F, p) = - forces( set_F_and_p!(W, F, p) )[2]

# grad_F(W::Wcb2, F, p) = 0  # TODO

function dpdpWcb(W::Wcb2{T}) where T
   J = Calculus.jacobian(p -> grad_p(W, Matrix(I,3,3), p))
   dpdpW = J(W.p1)
   return Mat3{T}(0.5 * (dpdpW + dpdpW'))
end

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
#    Ih = Matrix(I, 3,3)
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
