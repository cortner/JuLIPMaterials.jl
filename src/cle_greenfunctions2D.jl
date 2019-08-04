
using JuLIPMaterials: Vec2, Mat2, Ten22, Ten42

using Einsum, StaticArrays

export GreenFunction2D, IsoGreenFcn2D, grad

# ========== Implementation of the 2D Green's Function =============

struct GreenFunction2D{T} <: AbstractGreenFunction{T}
   Nquad::Int
   C::Ten42{T}
   remove_singularity::Bool
end

# 2D rotation through π/2
Rot = [0 -1; 1  0];

"""
`GreenFunction2D`
construct a green's function type
"""
function GreenFunction(C::Ten42; Nquad = nothing, remove_singularity = true)
   if Nquad == nothing
      error("still need to implement auto-tuning, please provide Nquad")
   end
   return GreenFunction2D(Nquad, C, remove_singularity)
end


# GreenFunction{T}(C::Array{T, 4}; kwargs...) = GreenFunction(Ten42{T}(C); kwargs...)
#
# GreenFunction(at::AbstractAtoms; kwargs...) =
#       GreenFunction(calculator(at), at; kwargs...)
#
# GreenFunction(calc::AbstractCalculator, at::AbstractAtoms; kwargs...) =
#       GreenFunction(elastic_moduli(calc, at); kwargs...)


function (G::GreenFunction2D{T})(x) where T
   if G.remove_singularity && norm(x) < 1e-10
      return @SMatrix zeros(T, 2, 2)
   end
   return eval_green(Vec2(x), G.C, G.Nquad)
end

function grad(G::GreenFunction2D{T}, x) where T
   if G.remove_singularity && norm(x) < 1e-10
      return @SArray zeros(T, 2, 2, 2)
   end
   return grad_green(Vec2(x), G.C, G.Nquad)
end


"eval_green(x::Vec2, ℂ::Ten42, Nquad::Int)"
function eval_green(x::Vec2{TT}, ℂ::Ten42, Nquad::Int) where TT
   # allocate
   G = @SMatrix zeros(TT, 2, 2)
   zz = @MMatrix zeros(TT, 2, 2)
   # Initialise tensors.
   x̂ = x/norm(x)
   # two vectors orthogonal to x.
   x⟂ = Rot*x̂

   if iseven(Nquad)
      Nquad += 1
   end
   # Integrate
   for ω in range(0.0, step=pi/Nquad, length=Nquad)
      z = cos(ω) * x̂ + sin(ω) * x⟂
      @einsum zz[i,j] = z[α] * ℂ[i,α,j,β] * z[β]
      # Perform integration
      G += inv(zz)*log(dot(x,z))
   end
   # Normalise appropriately
   return G / (2*pi^2*Nquad)
end

struct IsoGreenFcn2D{T} <: AbstractGreenFunction{T}
   λ::T
   μ::T
   remove_singularity::Bool
end

IsoGreenFcn2D(λ, μ; remove_singularity = true) =
   IsoGreenFcn2D(λ, μ, remove_singularity)

function (G::IsoGreenFcn2D{T})(x) where T
   if G.remove_singularity && norm(x) < 1e-10
      return @SMatrix zeros(T, 2, 2)
   end
   return eval_greeniso(Vec2(x), G.λ, G.μ)
end

function grad(G::IsoGreenFcn2D{T}, x) where T
   if G.remove_singularity && norm(x) < 1e-10
      return @SArray zeros(T, 2, 2, 2)
   end
   return grad_greeniso(Vec2{T}(x), G.λ, G.μ)
end


"isotropic CLE Green's function"
function eval_greeniso(x::Vec2{T}, λ::Real, μ::Real) where T
   Id = one(SMatrix{2,2,T})
   return ( (λ+μ) * ( x*x'/dot(x,x)- 0.5 * Id ) -
            (λ+3*μ) * log(norm(x)) ) / (4.0*π*μ * (λ+2*μ))
end
