
"""
implements some tools + formulas from BBS79

all internals use StaticArrays, however, the arguments passed into the
outer functions can be arbitrary arrays
"""
module BBS

using StaticArrays

typealias Vec3{T} SVector{3, T}
typealias Mat3{T} SMatrix{3,3,T}
typealias Ten33{T} SArray{(3,3,3),T,3,27}
typealias Ten43{T} SArray{(3,3,3,3),T,4,81}

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
      B += mm - nm' * nn⁻¹ * nm           # (3.6.9) and using  mn = nm'  (potential bug?)
   end
   return Q * (-1/Nquad), S * (-1/Nquad), B * (1/4/Nquad/pi)
end

"""
u_bbs(x, b, ν, C) -> u

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


end
