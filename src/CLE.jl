
module CLE

using Statistics: mean

using JuLIP: AbstractAtoms, AbstractCalculator, calculator,
             positions, set_positions!, volume, stress, cell, set_cell!, apply_defm!, rotation_matrix

using StaticArrays, Einsum

using JuLIPMaterials: Vec3, Mat3, Ten33, Ten43,
         MVec3, MMat3, MTen33, MTen43, ForceConstantMatrix1

export elastic_moduli, voigt_moduli, cubic_moduli, voigt_stress, voigt_strain, cauchy_stress, cauchy_strain, rotate_moduli

# TODO: get rid of this?
const Tensor{T} = Array{T, 4}

# extend `angle` to avoid going via ℂ
Base.angle(x, y) = atan(y, x)

"convert normalised vector to spherical coordinates"
spherical(x) = angle(x[1], x[2]), angle(norm(x[1:2]), x[3])

"convert spherical to euclidean coordinates on the unit sphere"
euclidean(φ, ψ) = Vec3(cos(ψ) * cos(φ), cos(ψ) * sin(φ), sin(ψ))

"given a vector x ∈ ℝ³, return `z0, z1` where `(x/norm(x),z0,z1)` form a right--handed ONB."
function onb3D(x::Vec3{T}) where {T}
   x /= norm(x)
   φ, ψ = spherical(x)
   return Vec3{T}( sin(ψ)*cos(φ), sin(ψ)*sin(φ), -cos(ψ) ),
          Vec3{T}(-sin(φ), cos(φ), zero(T) )
end



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

function elastic_moduli(calc::AbstractCalculator, at::AbstractAtoms; h=nothing)
   X0, cell0 = positions(at), cell(at)
   Ih = Matrix(1.0I(3))
   h === nothing && (h = eps()^(1/3))
   C = zeros(3,3,3,3)
   for i = 1:3, a = 1:3
      set_positions!(at, X0)
      set_cell!(at, cell0)

      Ih = Matrix(1.0I(3))
      Ih[i,a] += h
      Ih[a,i] += h
      apply_defm!(at, Ih)
      Sp = stress(calc, at)

      Ih[i,a] -= 2h
      Ih[a,i] -= 2h
      apply_defm!(at, Ih)
      Sm = stress(calc, at)

      C[i, a, :, :] = (Sp - Sm) / (2*h)
   end
   set_positions!(at, X0)
   set_cell!(at, cell0)
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

function elastic_moduli(FCM::ForceConstantMatrix1)
   ℂ = zeros(3,3,3,3)
   for i=1:length(FCM.R)
       ρ = FCM.R[i]
       Hmat = FCM.H[i]
       @einsum ℂ[a,b,c,d] = Hmat[a,c] * ρ[b] * ρ[d]
   end
   return ℂ
end

"""
`voigt_moduli`: compute elastic moduli in the format of Voigt moduli.

Methods:
* `voigt_moduli(at)`
* `voigt_moduli(calc, at)`
* `voigt_moduli(C)`
* `voigt_moduli(C11, C12, C44)`
"""
voigt_moduli(at::AbstractAtoms) = voigt_moduli(calculator(at), at)

voigt_moduli(calc::AbstractCalculator, at::AbstractAtoms) =
   voigt_moduli(elastic_moduli(calc, at))

const voigtinds = [1, 5, 9, 8, 7, 4] # xx yy zz yz xz xy

voigt_moduli(C::Array{T,4}) where {T} = reshape(C, 9, 9)[voigtinds, voigtinds]

function voigt_moduli(C11::T, C12::T, C44::T) where {T}
   z = zero(C11)
   return [C11 C12 C12 z   z    z
           C12 C11 C12 z   z    z
           C12 C12 C11 z   z    z
           z   z   z   C44 z    z
           z   z   z   z   C44  z
           z   z   z   z   z    C44]
end

"""
Convert from 3×3 stress matrix σ to Voigt 6-vector
"""
voigt_stress(σ::AbstractMatrix) = (@assert size(σ) == (3,3);
                                   reshape((σ + σ')/2, :)[voigtinds])
voigt_stress(calc::AbstractCalculator, at::AbstractAtoms) = voigt_stress(stress(calc, at))
voigt_stress(at::AbstractAtoms) = voigt_stress(stress(at))

"""
Convert from 3×3 strain matrix ϵ to Voigt 6-vector
"""
voigt_strain(ϵ::AbstractMatrix) = (@assert size(ϵ) == (3,3);
                                   reshape(ϵ + ϵ' - Diagonal(ϵ), :)[voigtinds])

# FIXME maybe there's a better name for 3x3 stress tensor than `cauchy_stress`?
"""
Convert from 6-vector Voigt stress to 3x3
"""
cauchy_stress(σ::AbstractVector) = (@assert length(σ) == 6; 
                                    [σ[1] σ[6] σ[5]
                                     σ[6] σ[2] σ[4]
                                     σ[5] σ[4] σ[3]])

"""
Convert from 6-vector Voigt stress to 3x3 matrix
"""
cauchy_strain(ϵ::AbstractVector) = (@assert length(ϵ) == 6;
                                    [ϵ[1]    ϵ[6]/2  ϵ[5]/2
                                     ϵ[6]/2  ϵ[2]    ϵ[4]/2
                                     ϵ[5]/2  ϵ[4]/2  ϵ[3]])

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
    cubic_moduli(C, [tol=1e-4])

Convert from elastic moduli (3x3x3x3) or Voigt moduli (6x6) to cubic elastic moduli.
Symmetry must be obeyed to within a tolerance of `tol`. Returns (C11, C12, C44). 
"""
function cubic_moduli(C::AbstractMatrix; tol=1e-4)
   C = copy(C)
   idxss = [[(1,1), (2,2), (3,3)],                       # C11
            [(1,2), (1,3), (2,1), (2,3), (3,1), (3,2)],  # C12 
            [(4,4), (5,5), (6,6)]]                       # C44

   cubic_C = []
   for idxs in idxss
      Cs = view(C, CartesianIndex.(idxs))
      @assert maximum(abs.(Cs .- mean(Cs))) < tol
      push!(cubic_C, mean(Cs))
      fill!(Cs, 0.0)
   end
   # check remaining elements are near to zero
   @assert maximum(abs.(C)) < tol 
   return Tuple(cubic_C)
end

cubic_moduli(c::Array{T,4}; kwargs...) where {T} = cubic_module(voigt_moduli(c); kwargs...)
cubic_moduli(calc::AbstractCalculator, at::AbstractAtoms; kwargs...) = cubic_moduli(elastic_moduli(calc, at); kwargs...)
cubic_moduli(at::AbstractAtoms; kwargs...) = cubic_moduli(elastic_moduli(at); kwargs...)

"""
Return rotated elastic moduli for a general crystal

Parameters

`C` : array
   3x3x3x3 tensor of elastic constants
`A` : array
   3x3 rotation matrix.

Returns

`C` : array
   3x3x3x3 matrix of rotated elastic constants (Voigt notation).
"""
function rotate_moduli(C::Array{T,4}, A::AbstractMatrix{T}) where {T}
   # check its a rotation matrix
   @assert A * A' ≈ I

   C_rot = zeros(eltype(C), 3, 3, 3, 3)
   @einsum C_rot[i, j, k, l] = A[i, a] * A[j, b] * A[k, c] * A[l, d] * C[a, b, c, d]
   return C_rot
end

rotate_moduli(C::AbstractMatrix{T}, A::AbstractMatrix{T}) where {T} = voigt_moduli(rotate_moduli(elastic_moduli(C), A))


"""
`isotropic_moduli(λ, μ)`: compute 4th order tensor of elastic moduli
corresponding to the Lame parameters λ, μ.
"""
function isotropic_moduli(λ, μ)
   K = λ + μ * 2 / 3
   C = @SArray [ K * I[i,j] * I[k,l] + μ * (I[i,k]*I[j,l] + I[i,l]*I[j,k] - 2/3*I[i,j]*I[k,l])
         for i = 1:3, j = 1:3, k = 1:3, l = 1:3 ]
   return C
end

"""
`isotropic_moduli2D(λ, μ)`: compute 4th order tensor of elastic moduli
(in 2D planar elasticity) corresponding to the Lame parameters λ, μ.
"""
function isotropic_moduli2D(λ, μ)
   K = λ + μ * 2 / 3
   C = @SArray [ K * I[i,j] * I[k,l] + μ * (I[i,k]*I[j,l] + I[i,l]*I[j,k] - 2/3*I[i,j]*I[k,l])
         for i = 1:2, j = 1:2, k = 1:2, l = 1:2 ]
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

include("cle_gfcorrectors3D.jl")

include("cle_dislocations.jl")

end
