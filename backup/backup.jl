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
