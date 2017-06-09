
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

# extend `angle` to avoid going via C
Base.angle(x, y) = Base.angle(x + im * y)   # atan2(y, x)

"convert normalised vector to spherical coordinates"
spherical(x) = angle(x[1], x[2]), angle(norm(x[1:2]), x[3])

"convert spherical to euclidean coordinates on the unit sphere"
euclidean(φ, ψ) = Vec3(cos(ψ) * cos(φ), cos(ψ) * sin(φ), sin(ψ))

"given a vector x ∈ R³, return `(z0, z1)` where `(x/norm(x),z0,z1)` form an ONB."
function onb{T}(x::Vec3{T})
   x /= norm(x)
   φ, ψ = spherical(x)
   return Vec3{T}(-sin(φ), cos(φ), 0.0),
          Vec3{T}(sin(ψ)*cos(φ), sin(ψ)*sin(φ), -cos(ψ))
end

"explicit inverse to enable AD, if A is an `SMatrix` then the conversion is free"
inv3x3(A) = inv( Mat3(A) )

# contract(a, C, b) -> D  where D_ij = C_injm a_n b_m
contract(a::Vec3, C::Ten43, b::Vec3) =
   Mat3([dot(C[i,:,j,:] * b, a) for i=1:3, j = 1:3])

contract2(C::Ten43, a::Vec3) = constract(a, C, a)


quad(f, Nquad) = sum(f, range(0, 2*pi/Nquad, Nquad)) * 2*pi / Nquad
z(z0, z1, ω) = cos(ω) * z0 + sin(ω) * z1


# ========== Implementation of the BBS79 formula for a dislocation =============

"""
u_bbs(x, b, ν, C) -> u

the dislocation line is assumed to pass through the origin

* x : position in space
* b : burgers vector
* t : dislocation direction (tangent to dislocation line)
* C : elastic moduli 3 x 3 x 3 x 3
"""
function u_bbs(x, b, t, C, Nquad = 10)
   x, b, t, C = Vec3(x), Vec3(b), Vec3(t), Ten43(C)
   t /= norm(t)
   m, n = onb(t)
   Q = quad( ω -> inv3x3( contract2(C, z(ω)) ) * (-1/pi)
   S = quad( ω ->
end


end
