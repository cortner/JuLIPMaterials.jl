
using JuLIPMaterials: Vec3, Mat3, Ten33, Ten43
using JuLIPMaterials.CLE: onb3D, euclidean, spherical

using Einsum, StaticArrays


export GreenFunction, IsoGreenFcn3D, grad

# ========== Implementation of the 3D Green's Function =============

abstract type AbstractGreenFunction{T} end


struct GreenFunction3D{T} <: AbstractGreenFunction{T}
   Nquad::Int
   C::Ten43{T}
   remove_singularity::Bool
end

"""
`GreenFunction3D`
construct a green's function type
"""
function GreenFunction(C::Ten43; Nquad = nothing, remove_singularity = true)
   if Nquad == nothing
      error("still need to implement auto-tuning, please provide Nquad")
   end
   return GreenFunction3D(Nquad, C, remove_singularity)
end


GreenFunction(C::Array{T, 4}; kwargs...) where {T} =
      GreenFunction(Ten43{T}(C); kwargs...)


GreenFunction(at::AbstractAtoms; kwargs...) =
      GreenFunction(calculator(at), at; kwargs...)

GreenFunction(calc::AbstractCalculator, at::AbstractAtoms; kwargs...) =
      GreenFunction(elastic_moduli(calc, at); kwargs...)


function (G::GreenFunction3D{T})(x) where T
   if G.remove_singularity && norm(x) < 1e-10
      return @SMatrix zeros(T, 3, 3)
   end
   return eval_green(Vec3(x), G.C, G.Nquad)
end

function grad(G::GreenFunction3D{T}, x) where T
   if G.remove_singularity && norm(x) < 1e-10
      return @SArray zeros(T, 3, 3, 3)
   end
   return grad_green(Vec3(x), G.C, G.Nquad)
end


"eval_green(x::Vec3, ℂ::Ten43, Nquad::Int)"
function eval_green(x::Vec3{TT}, ℂ::Ten43, Nquad::Int) where TT
   # allocate
   G = @SMatrix zeros(TT, 3, 3)
   zz = @MMatrix zeros(TT, 3, 3)
   # Initialise tensors.
   x̂ = x/norm(x)
   # two vectors orthogonal to x.
   x1, x2 = onb3D(x̂)
   # Integrate
   for ω in range(0.0, pi/Nquad, Nquad)
      z = cos(ω) * x1 + sin(ω) * x2
      @einsum zz[i,j] = z[α] * ℂ[i,α,j,β] * z[β]
      # Perform integration
      G += inv(zz)
   end
   # Normalise appropriately
   return G / (4*pi*norm(x)*Nquad)
end


function grad_green(x::Vec3{TT}, ℂ::Ten43, Nquad::Int) where TT
   # allocate
   DG = @MArray zeros(TT, 3, 3, 3)
   zz = @MMatrix zeros(TT, 3, 3)
   zT = @MMatrix zeros(TT, 3, 3)
   # Initialise tensors.
   x̂ = x/norm(x)
   # two vectors orthogonal to x.
   x1, x2 = onb3D(x̂)
   # Integrate
   for ω in range(0.0, pi/Nquad, Nquad)
      z = cos(ω) * x1 + sin(ω) * x2
      @einsum zz[i,j] = z[α] * ℂ[i,α,j,β] * z[β]
      @einsum zT[i,j] = z[α] * ℂ[i,α,j,β] * x̂[β]
      zzinv = inv(zz)
      F = zzinv * (zT + zT') * zzinv
      @einsum DG[i,j,k] = DG[i,j,k] + zzinv[i,j] * x̂[k] - F[i,j] * z[k]
   end
   DG ./= (-4.0 * pi * norm(x)^2 * Nquad)
   return SArray(DG)
end


struct IsoGreenFcn3D{T} <: AbstractGreenFunction{T}
   λ::T
   μ::T
   remove_singularity::Bool
end

IsoGreenFcn3D(λ, μ; remove_singularity = true) =
   IsoGreenFcn3D(λ, μ, remove_singularity)

function (G::IsoGreenFcn3D{T})(x) where T
   if G.remove_singularity && norm(x) < 1e-10
      return @SMatrix zeros(T, 3, 3)
   end
   return eval_greeniso(Vec3(x), G.λ, G.μ)
end

function grad(G::IsoGreenFcn3D{T}, x) where T
   if G.remove_singularity && norm(x) < 1e-10
      return @SArray zeros(T, 3, 3, 3)
   end
   return grad_greeniso(Vec3{T}(x), G.λ, G.μ)
end


"isotropic CLE Green's function"
function eval_greeniso(x::Vec3{T}, λ::Real, μ::Real) where T
   Id = @SMatrix eye(T, 3)
   x̂ = x/norm(x)
   return (((λ+3*μ)/(λ+2*μ)/norm(x)) * Id  +
               ((λ+μ)/(λ+2*μ)/norm(x)) * x̂ * x̂') / (8.0*π*μ)
end

"gradient of isotropic CLE Green's function"
function grad_greeniso(x::Vec3{T}, λ::Real, μ::Real) where T
   DG = @MArray zeros(T, 3, 3, 3)
   Id = @SArray eye(T, 3)
   x̂ = x / norm(x)
   for i = 1:3, j = 1:3, k = 1:3
      DG[i,j,k] = (λ+μ) * (Id[i,k] * x̂[j] + Id[j,k] * x̂[i]) - (λ+3*μ) * Id[i,j] * x̂[k] - 3*(λ+μ) * x̂[i] * x̂[j] * x̂[k]
   end
   DG ./= 8 * π * μ*(λ+2*μ) * norm(x)^2
   return DG
end
