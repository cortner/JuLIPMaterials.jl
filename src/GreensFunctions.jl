
"""
`module GreensFunctions`

Implements some CLE Green's functions, both in analytic form or
semi-analytic using the formulas from BBS79.
"""
module GreensFunctions

using MaterialsScienceTools: Vec3, Mat3, Ten33, Ten43

using Einsum, StaticArrays


# extend `angle` to avoid going via ℂ
Base.angle(x, y) = atan2(y, x)

"convert normalised vector to spherical coordinates"
spherical(x) = angle(x[1], x[2]), angle(norm(x[1:2]), x[3])

"convert spherical to euclidean coordinates on the unit sphere"
euclidean(φ, ψ) = Vec3(cos(ψ) * cos(φ), cos(ψ) * sin(φ), sin(ψ))

"given a vector x ∈ ℝ³, return `z0, z1` where `(x/norm(x),z0,z1)` form an ONB."
function onb{T}(x::Vec3{T})
   x /= norm(x)
   φ, ψ = spherical(x)
   return Vec3{T}(-sin(φ), cos(φ), zero(T)),
          Vec3{T}(sin(ψ)*cos(φ), sin(ψ)*sin(φ), -cos(ψ))
end

"explicit inverse to enable AD, if A is an `SMatrix` then the conversion is free"
inv3x3(A) = inv( Mat3(A) )

# ========== Implementation of the 3D Green's Function =============

struct GreenFunction3D{T}
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

GreenFunction{T}(C::Array{T, 4}; kwargs...) = GreenFunction(Ten43{T}(C); kwargs...)

function (G::GreenFunction3D{T})(x) where T
   if G.remove_singularity && norm(x) < 1e-10
      return @SMatrix zeros(T, 3, 3)
   end
   return eval_green(x, G.C, G.Nquad)
end

function grad(G::GreenFunction3D{T}, x) where T
   if G.remove_singularity && norm(x) < 1e-10
      return @SArray zeros(T, 3, 3, 3)
   end
   return grad_green(x, G.C, G.Nquad)
end


"eval_green(x::Vec3, ℂ::Ten43, Nquad::Int)"
function eval_green(x::Vec3{T}, ℂ::Ten43{T}, Nquad::Int) where T <: AbstractFloat
   # allocate
   G = @SMatrix zeros(T, 3, 3)
   zz = @MMatrix zeros(T, 3, 3)
   # Initialise tensors.
   x̂ = x/norm(x)
   # two vectors orthogonal to x.
   x1, x2 = onb(x̂)
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


function grad_green(x::Vec3{T}, ℂ::Ten43{T}, Nquad::Int) where T <: AbstractFloat
   # allocate
   DG = @MArray zeros(T, 3, 3, 3)
   zz = @MMatrix zeros(T, 3, 3)
   zT = @MMatrix zeros(T, 3, 3)
   # Initialise tensors.
   x̂ = x/norm(x)
   # two vectors orthogonal to x.
   x1, x2 = onb(x̂)
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


# ========== Implementation of the BBS79 formula for a dislocation =============

function QSB(m0, n0, Nquad)
   Q, S, B = zero(Mat3), zero(Mat3), zero(Mat3)
   for ω in range(0, pi/Nquad, Nquad)
      m = cos(ω) * m0 + sin(ω) * n0
      n = sin(ω) * m0 + cos(ω) * n0
      nn⁻¹ = inv3x3( contract2(C, n) )
      nm = contract(n, C, m)
      mm = contract2(C, m)
      Q += nn⁻¹                           # (3.6.4)
      S += nn⁻¹ * nm                      # (3.6.6)
      B += mm - nm' * nn⁻¹ * nm           # (3.6.9) and using  mn = nm'  (TODO: potential bug?)
   end
   return Q * (-1/Nquad), S * (-1/Nquad), B * (1/4/Nquad/pi)
end

"""
grad_u_bbs(x, b, ν, C) -> ∇u

the dislocation line is assumed to pass through the origin

* x : position in space
* b : burgers vector
* t : dislocation direction (tangent to dislocation line)
* C : elastic moduli 3 x 3 x 3 x 3
"""
function grad_u_bbs(x, b, t, C, Nquad = 10)
   x, b, t, C = Vec3(x), Vec3(b), Vec3(t), Ten43(C)
   t /= norm(t)
   m0, n0 = onb(t)   # some refrence ONB for computing Q, S, B
   Q, S, B = QSB(m0, n0, Nquad)
   # compute displacement gradient
   # project x to the (m0, n0) plane and compute the (m, n) vectors
   x = x - dot(x, t) * t
   r = norm(x)
   m = x / norm(x)
   n = m × t
   #  Implement (4.1.16)
   nn⁻¹ = inv3x3( contract(n, C, n) )
   nm = contract(n, C, m)
   return 1/(2*π*r) * ( (- S * b) ⊗ m + (nn⁻¹ * ((2*π*B + nm*S) * b)) ⊗ n )
end



"""
* `IsoGreenTensor3D(x,μ::Float64,λ::Float64) -> G, DG`
Returns the 3D Isotropic Green tensor and gradient at the point x, via exact formula
with Lamé parameters μ and λ.
"""
function IsoGreenTensor3D(x,μ::Float64,λ::Float64)
    # Error handling which should be redone with types.
    if ~( size(x)==(3,) ) && ~( size(x)==(3,1) )
        error("Input is not a 3D column vector");
    end

    # Construct Green tensor
    G = ((λ+3*μ)/(λ+2*μ)*eye(3)/norm(x)+(λ+μ)/(λ+2*μ)*x*x'/norm(x)^3)/(8*pi*μ);

    # Construct gradient of Green tensor
    DG = zeros(Float64,3,3,3);
    Id = eye(3);
    for i=1:3, j=1:3, k=1:3
            DG[i,j,k] = ( (λ+μ)*(Id[i,k]*x[j]+Id[j,k]*x[i]) - (λ+3*μ)*eye(3)[i,j]*x[k] ) -
                            3*(λ+μ)*x[i]*x[j]*x[k]/(norm(x)^2);
    end
    DG = DG/(8*pi*μ*(λ+2*μ)*norm(x)^3);

    return G, DG
end

end
