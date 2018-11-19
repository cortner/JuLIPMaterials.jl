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

function GreenFunctionCorrector(ℂ, FCM::ForceConstantMatrix1; Nquad = nothing, remove_singularity = true)
   if Nquad == nothing
      error("still need to implement auto-tuning, please provide Nquad")
   end
   return GreenFunctionCorrector3D(Nquad, ℂ, FCM, remove_singularity)
end

function (Gcorr::GreenFunctionCorrector3D{T})(x) where T
   if Gcorr.remove_singularity && norm(x) < 1e-10
      return @SMatrix zeros(T, 3, 3)
   end
   return eval_corrector(Vec3(x), Gcorr.C, Gcorr.FCM, Gcorr.Nquad)
end

# Utility functions for computing Fourier multipliers
function _C4(k::Vec3{TT},FCM::ForceConstantMatrix1{SS}) where {TT, SS}
   H0 = zeros(TT,3,3)
   for i=1:length(FCM.R)
      ρ = FCM.R[i]
      Hmat = FCM.H[i]
      H0 += (dot(k, ρ)^4)/24*Hmat
   end
   return H0
end

function _dC4(k::Vec3{TT},FCM::ForceConstantMatrix1{SS}) where {TT, SS}
   H0 = zeros(TT,3,3,3)
   for i=1:length(FCM.R)
      ρ = FCM.R[i]
      Hmat = FCM.H[i]
      @einsum H0[a,b,c] += (dot(k, ρ)^3)/6 * Hmat[a,b] * ρ[c]
   end
   return H0
end

function _ddC4(k::Vec3{TT},FCM::ForceConstantMatrix1{SS}) where {TT, SS}
   H0 = zeros(TT,3,3,3,3)
   for i=1:length(FCM.R)
      ρ = FCM.R[i]
      Hmat = FCM.H[i]
      @einsum H0[a,b,c,d] += dot(k, ρ)^2/2 * Hmat[a,b] * ρ[c] * ρ[d]
   end
   return H0
end

function _C2(k::Vec3{TT},ℂ::Array{SS,4}) where {TT, SS}
   C2 = zeros(TT,3,3)
   @einsum C2[i,j] = k[a] * ℂ[i,a,j,b] * k[b]
   return C2;
end

function _dC2(k::Vec3{TT},ℂ::Array{SS,4}) where {TT, SS}
   dC2 = zeros(TT,3,3,3)
   @einsum dC2[i,j,b] = ℂ[i,a,j,b] * k[a] + ℂ[i,b,j,a] * k[a]
   return dC2;
end

function _C2inv(k::Vec3{TT},ℂ::Array{SS,4}) where {TT, SS}
   C2 = _C2(k,ℂ);
   return inv(C2);
end

function _dC2inv(k::Vec3{TT},ℂ::Array{SS,4}) where {TT, SS}
   dC2inv = zeros(TT,3,3,3)
   dC2 = _dC2(k,ℂ)
   C2inv = _C2inv(k,ℂ)
   @einsum dC2inv[a,b,c] = -C2inv[a,d]*dC2[d,e,c]*C2inv[e,b]
   return dC2inv;
end

function _dC2inv1(k::Vec3{TT},ℂ::Array{SS,4}) where {TT, SS}
   dC2inv = zeros(TT,3,3,3)
   C2 = _C2(k,ℂ)
   dC2 = _dC2(k,ℂ)
   for c=1:3
      dC2inv[:,:,c] = -C2 \ dC2[:,:,c] / C2
   end
   return dC2inv;
end

function _ddC2inv(k::Vec3{TT},ℂ::Array{SS,4}) where {TT, SS}
   ddC2inv = zeros(TT,3,3,3,3)
   # Get matrices
   dC2 = _dC2(k,ℂ)
   dC2inv = _dC2inv1(k,ℂ)
   C2inv = _C2inv(k,ℂ)
   @einsum ddC2inv[a,b,c,n] = -dC2inv[a,d,n]*dC2[d,e,c]*C2inv[e,b]-
                                 2*C2inv[a,d]*ℂ[d,e,c,n]*C2inv[e,b]-
                                    C2inv[a,d]*dC2[d,e,c]*dC2inv[e,b,n]
   return ddC2inv;
end

function _ddC2inv1(k::Vec3{TT},ℂ::Array{SS,4}) where {TT, SS}
   ddC2inv = zeros(TT,3,3,3,3)
   # Get matrices
   dC2 = _dC2(k,ℂ)
   dC2inv = _dC2inv1(k,ℂ)
   C2 = _C2(k,ℂ)
   for c=1:3, n=1:3
      ddC2inv[:,:,c,n] = -(dC2inv[:,:,n] * dC2[:,:,c]) / C2 -
                           2*(C2 \ ℂ[:,:,c,n] / C2) -
                           C2 \ (dC2[:,:,c] * dC2inv[:,:,n]);
   end
   return ddC2inv;
end

# Multiplier for lattice elasticity
function _full_lattice_multiplier(k::Vec3{TT},FCM::ForceConstantMatrix1{SS}) where {TT, SS}
   D = @SMatrix zeros(TT,3,3)
   for i=1:length(FCM.R)
      D += -2*sin( 0.5*dot(k,FCM.R[i]) )^2 * FCM.H[i]
   end
   return D
end

# 0--homogeneous multiplier
function _corrector_multiplier(k::Vec3{TT},ℂ::Array{SS,4},FCM::ForceConstantMatrix1{SS}) where {TT, SS}
   C2inv = _C2inv(k,ℂ)
   C4 = _C4(k,FCM)
   return C2inv * C4 * C2inv
end

function _d_corrector_multiplier(k::Vec3{TT},ℂ::Array{SS,4},FCM::ForceConstantMatrix1{SS}) where {TT, SS}
   dH2 = zeros(TT,3,3,3)
   C2inv = _C2inv(k,ℂ)
   dC2inv = _dC2inv(k,ℂ)
   C4 = _C4(k,FCM)
   dC4 = _dC4(k,FCM)
   @einsum dH2[a,b,c] = dC2inv[a,i,c]*C4[i,j]*C2inv[j,b] + C2inv[a,i]*dC4[i,j,c]*C2inv[j,b] + C2inv[a,i]*C4[i,j]*dC2inv[j,b,c]
   return dH2
end

function _dd_corrector_multiplier(k::Vec3{TT},ℂ::Array{SS,4},FCM::ForceConstantMatrix1{SS}) where {TT, SS}
   ddH2 = zeros(TT,3,3,3,3)
   C2inv = _C2inv(k,ℂ)
   dC2inv = _dC2inv1(k,ℂ)
   ddC2inv = _ddC2inv1(k,ℂ)
   C4 = _C4(k,FCM)
   dC4 = _dC4(k,FCM)
   ddC4 = _ddC4(k,FCM)
   for a=1:3, b=1:3, c=1:3, d=1:3
       ddH2[a,b,c,d] = ddC2inv[a,:,c,d]' * C4 * C2inv[:,b] +
                       dC2inv[a,:,d]' * dC4[:,:,c] * C2inv[:,b] +
                       dC2inv[a,:,d]' * C4 * dC2inv[:,b,c] +
                       dC2inv[a,:,c]' * dC4[:,:,d] * C2inv[:,b] +
                       C2inv[a,:]' * ddC4[:,:,c,d] * C2inv[:,b] +
                       C2inv[a,:]' * dC4[:,:,d] * dC2inv[:,b,c] +
                       dC2inv[a,:,c]' * C4 * dC2inv[:,b,d] +
                       C2inv[a,:]' * dC4[:,:,c] * dC2inv[:,b,d] +
                       C2inv[a,:]' * C4 * ddC2inv[:,b,c,d];
   end
   return ddH2
end

"eval_corrector(x::Vec3, ℂ::Ten43, Nquad::Int)"
function eval_corrector(x::Vec3{TT}, ℂ::Array{SS,4},FCM::ForceConstantMatrix1{SS},Nquad::Int) where {TT, SS}
   # allocate
   Gcorr = @SMatrix zeros(TT,3,3)
   zz = @MMatrix zeros(TT,3,3)
   # Initialise tensors.
   x̂ = x/norm(x)
   # two vectors orthogonal to x.
   x1, x2 = onb3D(x̂)
   # Integrate
   DH2 = zeros(TT,3,3,3)
   ∂H2 = zeros(TT,3,3)
   D²H2 = zeros(TT,3,3,3,3)
   ∂²H2 = zeros(TT,3,3)
   for ω in range(0.0, pi/Nquad, Nquad)
      z = cos(ω) * x1 + sin(ω) * x2
      D²H2 = _dd_corrector_multiplier(z,ℂ,FCM)
      for i=1:3, j=1:3
         ∂²H2[i,j] = x̂' * D²H2[i,j,:,:] * x̂
      end
      DH2 = _d_corrector_multiplier(z,ℂ,FCM)
      @einsum ∂H2[i,j] = DH2[i,j,a]*z[a]
      Gcorr += -∂H2 + ∂²H2
   end
   # Normalise appropriately
   return Gcorr / (4*pi*norm(x)^3*Nquad)
end
