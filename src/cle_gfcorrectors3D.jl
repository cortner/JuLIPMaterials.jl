using JuLIPMaterials.CLE: onb3D, euclidean, spherical
using JuLIPMaterials: Vec3, Mat3, Ten33, Ten43, ForceConstantMatrix1

using Einsum, StaticArrays, ForwardDiff

export GreenFunctionCorrector

abstract type AbstractGreenFunctionCorrector{T} end

struct GreenFunctionCorrector3D{T} <: AbstractGreenFunctionCorrector{T}
   Nquad::Int
   C
   FCM::ForceConstantMatrix1{T}
   remove_singularity::Bool
end

function GreenFunctionCorrector(C, FCM::ForceConstantMatrix1; Nquad = nothing, remove_singularity = true)
   if Nquad == nothing
      error("still need to implement auto-tuning, please provide Nquad")
   end
   return GreenFunctionCorrector3D(Nquad, C, FCM, remove_singularity)
end

function (Gcorr::GreenFunctionCorrector3D{T})(x) where T
   if Gcorr.remove_singularity && norm(x) < 1e-10
      return @SMatrix zeros(T, 3, 3)
   end
   return eval_corrector(Vec3(x), Gcorr.C, Gcorr.FCM, Gcorr.Nquad)
end

# Multiplier for lattice elasticity
function _full_lattice_multiplier(k::Vec3{TT},FCM::ForceConstantMatrix1{TT}) where TT
   D = @SMatrix zeros(TT,3,3)
   for i=1:length(FCM.R)
      D += -2*sin( 0.5*dot(k,FCM.R[i]) )^2 * FCM.H[i]
   end
   return D
end

# 0--homogeneous multiplier
function _corrector_multiplier(k::Vec3{TT},ℂ,FCM) where TT
   # Construct H0
   H0 = @MMatrix zeros(TT,3,3)
   for i=1:length(FCM.R)
      H0 += dot(k, FCM.R[i])^4/24 * FCM.H[i]
   end
   kk = @MMatrix zeros(TT,3,3)
   @einsum kk[i,j] = k[α] * ℂ[i,α,j,β] * k[β]
   kkinv = inv(kk);
   return kkinv * H0 * kkinv
end

"eval_corrector(x::Vec3, ℂ::Ten43, Nquad::Int)"
function eval_corrector(x, ℂ, FCM, Nquad::Int)
   # allocate
   Gcorr = @SMatrix zeros(3, 3)
   zz = @MMatrix zeros(3, 3)
   # Initialise tensors.
   x̂ = x/norm(x)
   # two vectors orthogonal to x.
   x1, x2 = onb3D(x̂)
   # Integrate
   for ω in range(0.0, pi/Nquad, Nquad)
      z = cos(ω) * x1 + sin(ω) * x2
      u0 = t -> _corrector_multiplier( Vec3(z + t*x̂), ℂ, FCM)
      u1 = t -> ForwardDiff.derivative(u0,t)
      u2 = t -> ForwardDiff.derivative(u1,t)
      ∂²H0 = u2(0.0)
      v0 = t -> _corrector_multiplier( Vec3(z + t*z), ℂ, FCM)
      v1 = t -> ForwardDiff.derivative(v0,t)
      ∂H0z = v1(0.0)
      Gcorr += -∂H0z + ∂²H0
   end
   # Normalise appropriately
   return Gcorr / (8*pi^2*norm(x)^3*Nquad)
end
