
"""
`module GreensFunctions`

Implements some CLE Green's functions, both in analytic form or
semi-analytic using the formulas from BBS79.
"""
module GreensFunctions

using MaterialsScienceTools: Vec3, Mat3, Ten33, Ten43



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
   return Vec3{T}(-sin(φ), cos(φ), 0.0),
          Vec3{T}(sin(ψ)*cos(φ), sin(ψ)*sin(φ), -cos(ψ))
end

"explicit inverse to enable AD, if A is an `SMatrix` then the conversion is free"
inv3x3(A) = inv( Mat3(A) )

# contract(a, C, b) -> D  where D_ij = C_injm a_n b_m
# contract(a::Vec3, C::Ten43, b::Vec3) = Mat3([dot(C[i,:,j,:] * b, a) for i=1:3, j = 1:3])
# the following is a meta-programming trick (due to @keno) which basically makes
# the compiler write out the entire expression without intermediate allocation
#   (edit: this works only on v0.6; on v0.5 not much is gained)
# @eval contract(a, C, b) = Mat3($(Expr(:tuple, (:(dot(C[$i,:,$j,:] * b, a)) for i=1:3, j=1:3)...)))

contract2(C::Ten43, a::Vec3) = constract(a, C, a)

⊗(a::Vec3, b::Vec3) = a * b'

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
* `GreenTensor3D(x,ℂ,quadpts=10) -> G, DG`
Computes the full 3D Green tensor, G, and its gradient, DG, at the point x for elasticity
tensor ℂ ∈ ℝ³ˣ³ˣ³ˣ³, based on using quadpts quadrature points on a circle.
"""
function GreenTensor3D{T}(x::AbstractVector{T}, ℂ, quadpts=10;
                           remove_singularity = true)
    # Some basic error handling which should be redone with types.
    if ~( size(x)==(3,) ) && ~( size(x)==(3,1) )
        error("Input is not a 3D column vector")
    elseif ~( size(ℂ)==(3,3,3,3) )
        error("Elasticity tensor incorrect size")
    end

    if remove_singularity && norm(x) < 1e-10
      return zeros(T, 3, 3)
   end

    # Initialise tensors.
    G = zeros(T,3,3)
    DG = zeros(T,3,3,3)
    x̂ = x/norm(x)
    F = zeros(T,3,3)

    # basis is a 3x2 matrix of vectors orthogonal to x.
    basis = hcat(onb(Vec3(x))...)

    # Integrate
    for m=0:quadpts-1
        zz = zeros(T,3,3)
        zT = zeros(T,3,3)
        z = basis*[cos(pi*m/quadpts), sin(pi*m/quadpts)]
        for i=1:3, j=1:3
            zz[i,j] = dot(z, ℂ[i,:,j,:] * z)
            zT[i,j] = dot(z, ℂ[i,:,j,:] * x̂)
        end
        zzinv = inv3x3(zz)

        # Perform integration
        G += zzinv
        F = (zz\(zT+zT'))/zz
        for i=1:3, j=1:3, k=1:3
            DG[i,j,k] += zzinv[i,j]*x̂[k]-F[i,j]*z[k]
        end
    end

    # Normalise appropriately
    G = G/(4*pi*norm(x)*quadpts)
    DG = -DG/(4*pi*norm(x)^2*quadpts)

    return G, DG
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
