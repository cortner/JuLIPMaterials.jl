
module CLE

using JuLIP: AbstractAtoms, AbstractCalculator, calculator,
             stress, cell, set_cell!, apply_defm!

using StaticArrays, LinearAlgebra

using JuLIPMaterials: Vec3, Mat3, Ten33, Ten43,
         MVec3, MMat3, MTen33, MTen43

# TODO: get rid of this?
const Tensor{T} = Array{T, 4}

"""
* `elastic_moduli(at::AbstractAtoms)`
* `elastic_moduli(calc::AbstractCalculator, at::AbstractAtoms)`
* `elastic_moduli(C::Matrix)` : convert Voigt moduli to 4th order tensor

computes the 3 x 3 x 3 x 3 elastic moduli tensor

*Notes:* this is a naive implementation that does not exploit
any symmetries at all; this means it performs 9 centered finite-differences
on the stress. The error should be in the range 1e-10
"""
elastic_moduli(at::AbstractAtoms) = elastic_moduli(calculator(at), at)

function elastic_moduli(calc::AbstractCalculator, at::AbstractAtoms)
   F0 = (cell(at) |> Matrix)'
   Ih = Matrix(1.0*I,3,3)
   h = eps()^(1/3)
   C = zeros(3,3,3,3)
   for i = 1:3, a = 1:3
      Ih[i,a] += h
      apply_defm!(at, Ih)
      Sp = stress(calc, at)
      apply_defm!(at, inv(Ih))
      Ih[i,a] -= 2*h
      apply_defm!(at, Ih)
      Sm = stress(calc, at)
      apply_defm!(at, inv(Ih))
      C[i, a, :, :] = (Sp - Sm) / (2*h)
      Ih[i,a] += h
   end
   # symmetrise it - major symmetries C_{iajb} = C_{jbia}
   for i = 1:3, a = 1:3, j=1:3, b=1:3
      C[i,a,j,b] = C[j,b,i,a] = 0.5 * (C[i,a,j,b] + C[j,b,i,a])
   end
   # minor symmetries - C_{iajb} = C_{iabj}
   for i = 1:3, a = 1:3, j=1:3, b=1:3
      C[i,a,j,b] = C[i,a,b,j] = 0.5 * (C[i,a,j,b] + C[i,a,b,j])
   end
   return C
end

"""
`voigt_moduli`: compute elastic moduli in the format of Voigt moduli.

Methods:
* `voigt_moduli(at)`
* `voigt_moduli(calc, at)`
* `voigt_moduli(C)`
"""
voigt_moduli(at::AbstractAtoms) = voigt_moduli(calculator(at), at)

voigt_moduli(calc::AbstractCalculator, at::AbstractAtoms) =
   voigt_moduli(elastic_moduli(calc, at))

const voigtinds = [1, 5, 9, 4, 7, 8]

voigt_moduli(C::Array{T,4}) where {T} =
      reshape(C, 9, 9)[voigtinds, voigtinds]


function elastic_moduli(Cv::AbstractMatrix{T}) where {T}
   @assert size(Cv) == (6,6)
   C = zeros(T, 9,9)
   C[voigtinds, voigtinds] = Cv
   C = reshape(C, 3,3,3,3)
   # now apply all the symmetries to recover C
   for i = 1:3, a = 1:3, j = 1:3, b = 1:3
      if C[i,a,j,b] != 0
         C[a,i,j,b] = C[i,a,b,j] = C[a,i,b,j] = C[j,b,i,a] =
            C[b,j,i,a] = C[j,b,a,i] = C[b,j,a,i] = C[i,a,j,b]
      end
   end
   return C
end

"""
`isotropic_moduli(λ, μ)`: compute 4th order tensor of elastic moduli
corresponding to the Lame parameters λ, μ.
"""
function isotropic_moduli(λ, μ)
   K = λ + μ * 2 / 3
   C = [ K * I[i,j] * I[k,l] + μ * (I[i,k]*I[j,l] + I[i,l]*I[j,k] - 2/3*I[i,j]*I[k,l])
         for i = 1:3, j = 1:3, k = 1:3, l = 1:3 ]
   return C
end




function zener_anisotropy_index(C::Tensor)
    Cv = voigt_moduli(C)
    A = 2*Cv[4,4]/(Cv[1,1] - Cv[1,2])
    return A
end

zener_anisotropy_index(at::AbstractAtoms) =
                  zener_anisotropy_index(elastic_moduli(at))

"""
compute the Lame parameters for an elasticity tensor C or throw
an error if the material is not isotropic.
"""
function lame_parameters(C::Tensor; aniso_threshold=1e-3)
    A = zener_anisotropy_index(C)
    @assert abs(A - 1.0) < aniso_threshold
    Cv = voigt_moduli(C)
    μ = Cv[1,2]
    λ = Cv[4,4]
    return λ, μ
end
lame_parameters(at::AbstractAtoms) = lame_parameters(elastic_moduli(at))

poisson_ratio(λ, μ) = 0.5 * λ / (λ + μ)
poisson_ratio(C::Tensor) = poisson_ratio(lame_parameters(C)...)
poisson_ratio(at::AbstractAtoms) = poisson_ratio(elastic_moduli(at))

function youngs_modulus(λ, μ)
    ν = poisson_ratio(λ, μ)
    return λ*(1 + ν)*(1 - 2ν) / ν
end

youngs_modulus(C::Tensor) = youngs_modulus(lame_parameters(C)...)
youngs_modulus(at::AbstractAtoms) = youngs_modulus(elastic_moduli(at))

"""
check whether the elasticity tensor is isotropic; return true/false
"""
function is_isotropic(C::Tensor)
   try
      lame_parameters(C)
      return true
   catch
      return false
   end
end


# """
# `module GreensFunctions`
#
# Implements some CLE Green's functions, both in analytic form or
# semi-analytic using the formulas from BBS79.
# """

include("cle_greenfunctions.jl")

include("cle_dislocations.jl")

end
