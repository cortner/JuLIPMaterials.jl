
# TODO: this is not at all tested yet!!!!!!!

# we need this to evaluate the annoying integrand in the displacement field
using GaussQuadrature: chebyshev


# ========== Edge dislocation isotropic solid ==============

"""
`u_edge_isotropic(x, y, b, ν) -> u_x, u_y`

compute the displacement field `ux, uy` for an edge dislocation in an
isotropic linearly elastic medium, with core at (0,0),
burgers vector `b * [1.0;0.0]` and Poisson ratio `ν`

This is to be used primarily for comparison, since the exact solution will
not be the isotropic elasticity solution.
"""
function u_edge_isotropic(x, y, b, ν)
    r² = x.^2 + y.^2
    ux = b/(2*π) * ( angle.(x + im*y) + (x .* y) ./ (2*(1-ν) * r²) )
    uy = -b/(2*π) * ( (1-2*ν)/(4*(1-ν)) * log.(r²) + - 2 * y.^2 ./ (4*(1-ν) * r²) )
    return ux, uy
end


# ========== The Stroh / Hirth&Lothe Horror! =======================

include("sextic.jl")

# ========== Implementation of the BBS79 formula for a dislocation =============

function QSB(C, m0::Vec3{TT}, n0::Vec3{TT}, Nquad) where TT
   Q, S, B = zero(Mat3{TT}), zero(Mat3{TT}), zero(Mat3{TT})
   nn, nm, mm = zero(MMat3{TT}), zero(MMat3{TT}), zero(MMat3{TT})
   for ω in range(0, pi/Nquad, Nquad)
      m = cos(ω) * m0 + sin(ω) * n0
      n = sin(ω) * m0 + cos(ω) * n0
      @einsum nn[i,j] = n[α] * C[i,α,j,β] * n[β]
      @einsum nm[i,j] = n[α] * C[i,α,j,β] * m[β]
      @einsum mm[i,j] = m[α] * C[i,α,j,β] * m[β]
      nn⁻¹ = inv(nn)
      Q += nn⁻¹                           # (3.6.4)
      S += nn⁻¹ * nm                      # (3.6.6)
      B += mm - nm' * nn⁻¹ * nm           # (3.6.9) and using  mn = nm'
                                          #         (TODO: potential bug?)
   end
   return Q * (-1/Nquad), S * (-1/Nquad), B * (1/4/Nquad/pi)
end


function eval_dislocation(x::AbstractVector{TT}, b, t, C, Nquad=10) where TT
   # convert inputs (if needed)
   x, b, t, C = Vec3(x), Vec3(b), Vec3(t), Ten43(C)
   # normalise dislocation tangent direction
   t /= norm(t)
   # project x into the plane normal to t, this will not change the value of u
   x -= (t ⋅ x) * t
   # construct the ONB (t, m, n), the first vector is t,
   m = x / norm(x)   # p.145, l.7
   n = t × n
   # compute x ⤅ (r, ω)
   r = norm(x)
   m0, n0 = onb(t)      # fixed coordinate system w.r.t which we compute ω
   ω = angle( m ⋅ m0, m ⋅ n0 )
   # seems to be safe to ensure it is positive (TODO: revisit this?)
   if ω < 0.0
      ω += 2*π
   end
   # ------------ Implement components of (4.1.25) ------------
   nn, nm, mm = zero(MMat3{TT}), zero(MMat3{TT}), zero(MMat3{TT})
   Qω, Sω = zero(Mat3{TT}), zero(Mat3{TT})
   # first get the S, B tensors
   _, S, B = QSB(C, m, n, Nquad)
   # get a quadrature formula (with a little extra accuracy) + rescale
   Xquad, Wquad = chebyshev(Float64, Nquad+2)
   Xquad = (1.0 + Xquad) / 2.0 * ω    # now Xquad ranges from 0.0 to 2 ω
   Wquad = Wquad * (ω / sum(Wquad))
   for (ξ, dξ) in zip(Xquad, Wquad)
      a = cos(ξ) * m + sin(ξ) * n
      b = sin(ξ) * m + cos(ξ) * n
      @einsum nn[i,j] = a[α] * C[i,α,j,β] * a[β]
      @einsum nm[i,j] = a[α] * C[i,α,j,β] * b[β]
      nn⁻¹ = inv(nn)
      Qω += dξ * nn⁻¹
      Sω += dξ * (nn⁻¹ * nm)
   end
   # ---------- put everything together -------------  (4.1.25)
   u = (- S * b * log(r) + 4*π * Qω * B * b + Sω * S * b) / (2*π)
   return Vec3(u)
end


# """
# grad_u_bbs(x, b, ν, C) -> ∇u
#
# the dislocation line is assumed to pass through the origin
#
# * x : position in space
# * b : burgers vector
# * t : dislocation direction (tangent to dislocation line)
# * C : elastic moduli 3 x 3 x 3 x 3
# """


function grad_dislocation(x::AbstractVector{TT}, b, t, C, Nquad=10) where TT
   x, b, t, C = Vec3(x), Vec3(b), Vec3(t), Ten43(C)
   t /= norm(t)
   m0, n0 = onb(t)   # some refrence ONB for computing Q, S, B
   Q, S, B = QSB(m0, n0, Nquad)
   # compute displacement gradient
   # project x to the (m0, n0) plane and compute the (m, n) vectors
   x -= (x⋅t) * t
   r = norm(x)
   m = x / norm(x)
   n = m × t
   #  Implement (4.1.16)
   nn, nm, mm = zero(MMat3{TT}), zero(MMat3{TT}), zero(MMat3{TT})
   @einsum nn[i,j] = n[α] * C[i,α,j,β] * n[β]
   @einsum nm[i,j] = n[α] * C[i,α,j,β] * m[β]
   Du = 1/(2*π*r) * ( (- S * b) ⊗ m + (nn⁻¹ * ((2*π*B + nm*S) * b)) ⊗ n )
   return Mat3(Du)
end
