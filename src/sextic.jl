

include("Elasticity_110.jl")
using MaterialsScienceTools.CLE.Elasticity_110:
      elastic_moduli,
      voigt_moduli,
      fourth_order_basis,
      sextic_roots,
      A_coefficients,
      D_coefficients,
      little_a



u_edge_fcc_110{T}(x, y, b, C::Array{T,4}; TOL=1e-4) =
         u_edge(x, y, b, voigt_moduli(C), TOL=TOL)

"""
* `u_edge{T}(x, y, b, C::Array{T,N}) -> ux, uy`

* `C` can be either 3 x 3 x 3 x 3 (`elastic_moduli`) or 6 x 6 (`voigt_moduli`)

This function computes the anisotropic CLE solution for an in-plane
edge dislocation. The elastic moduli are taken to within `TOL` accuracy (
   this is a keyword argument)
"""
function u_edge_fcc_110{T}(x, y, b, Cv::Array{T,2}, a; TOL = 1e-4)
   Cv = copy(Cv)
   test1 = Cv[1,1]
   test2 = Cv[1,2]
   test3 = Cv[6,6]
   Cv = zeros(6,6)
   Cv[1,1] = 1.0*test1
   Cv[1,2] = 1.0*test2
   Cv[6,6] = 1.0*test3
   #Cv = zeros(6,6)
   #Hard code the elasticity tensor for now
   #Cv[1,1] = 15.145 #16.57
   #Cv[1,2] = 7.669#6.39
   #Cv[6,6] = 5.635#7.96
   # >>>>>>>>> START DEBUG >>>>>>>>
   #Zero out elasticity tensor from any numerical artifacts
   Cv[2,2] = Cv[3,3] = Cv[1,1]
   Cv[1,3] = Cv[2,3] = Cv[2,1] = Cv[3,1] = Cv[3,2] = Cv[1,2]
   Cv[4,4] = Cv[5,5] = Cv[6,6]
   # <<<<<<<<< END DEBUG <<<<<<<<<

   #C = fourth_order_basis(Cv,a)
   #Cvoigt = round(Cp, 3)
   #print("We are here !")
   #print(Cvoigt)

   #Compute Anisotropic solution from Chou and Sha, J. App. Phys 42 (7) 2625
   #This uses the elasticity tensor in the usual coordinates.
   #Note that the rotated tensor agrees with the values shown on 2625
   #Same problem as with Hirth and Lothe--either formulas are incorrect or problem
   #with numeric instability.  Complex log formulas seem to work
   bar = sqrt( .5*Cv[1,1]*(Cv[1,1] +Cv[1,2] + 2.0*Cv[4,4]) )
   C = ( (bar + Cv[1,2])*(bar - Cv[1,2] - 2.0*Cv[4,4] )  )/(bar*Cv[4,4])
   lem = (bar/Cv[1,1])^(1/2)
   #print("C :")
   #print(C)

   delta2 = sqrt(-C)
   delta1 = sqrt(C+4)

   #q = (x.^4 + 2 * x.^2 .* y.^2 * lem^2 + y.^4 * lem^4) + (C*lem^2 * x.^2 .*  y.^2)

   #ux1 = (b / (4.0*pi)) * (
   #       atan( (x.*y*lem*delta1) ./ (x.^2 - lem^2*y.^2) )
   #       + (bar^2 - Cv[1,2]^2) / (bar*Cv[4,4]*delta1*delta2) * ( atanh( (x.*y*lem*delta2) ./ (x.^2 + lem^2*y.^2) ) )
   #       )
   #uy1 = (-lem*b)/(4.0*pi) * (
   #      (bar - Cv[1,2])/(2*delta1*bar)*log(q)
   #       - (bar + Cv[1,2])/(delta2*bar) *
   #               atan( (y.^2*lem^2*delta1*delta2) ./ (2.0*x.^2 + (C + 2.0)*lem^2*y.^2 ) )
   #    )

   #Now compute using Hirth and Lothe which should be valid for 110 dislocation
   #This should use K instead of Cv
   #Something seems to be either wrong with these formulas or numeric issues are arising
   #Skip to general formula using complex logs

   K = fourth_order_basis(Cv,a)
   Cvoigt = round.(K, 3)
   c̄11 = sqrt(K[1,1]*K[2,2])    # (13-106)
   lam = (K[1,1]/K[2,2])^(1/4)
     ϕ = 0.5 * acos( (K[1,2]^2 + 2*K[1,2]*K[6,6] - c̄11^2) / (2.0*c̄11*K[6,6]) )
   apple = - lam*(K[6,6]*exp(im*ϕ) + c̄11*exp(-im*ϕ))/(K[1,2]+K[6,6])
   dodo = im*b[1,1]/(2.0*c̄11*sin(2*ϕ))*(K[1,2] - c̄11*exp(2*im*ϕ))

   #Should test this against solving the full linear system
   A = Complex{Float64}[0 0 0; 0 0 0; 0 0 0]
   A = [0 1 1; 0 apple -apple; 0 0 0]
   #Set up for burgers vector in x1 direction only
   D = zeros(6,1)
   D = [ 0; 0; real(dodo); imag(dodo); -real(dodo); -imag(dodo)]
   p = sextic_roots(K)
   #Do some funny business with root switching here.  This could be what is screwing things up.
   #May have to change the roots elsewhere instead
   #p[2,1] = p[1,1]
   #p[1,1] = lam*exp(im*ϕ)
   p[3,1] = -p[2,1]
   #print("Roots from root finder: ")
   #print(p)
   #print("Roots from Hirth:  ")
   #print(lam*exp(im*ϕ))
   ux = real.( im/(2*π)*(A[1,2]*(D[3] + D[4]*im)*log.(x+p[2]*y) + A[1,3]*(D[5] + D[6]*im)*log.(x+p[3]*y)   ))

   uy = real.( im/(2*π)*(A[2,2]*(D[3] + D[4]*im)*log.(x+p[2]*y) + A[2,3]*(D[5] + D[6]*im)*log.(x+p[3]*y)   ))

   @assert isreal(ux)
   @assert isreal(uy)

   return ux, uy
end








# const _four_to_two_ = @SMatrix [1 6 5; 6 2 4; 5 4 3]
#
# four_to_two(i, j) = _four_to_two_[i,j]
#
# four_to_two_index(i, j) = error("""`four_to_two_index` has been renamed
#  `four_to_two`; if you don't like it please file an issue :)""")
#
#
# const _two_to_four_ = @SVector [(1,1), (2,2), (3,3), (3,2), (3,1), (2,1)]
#
# two_to_four(k) = _two_to_four_[k]
#
# two_to_four_index(k) = error("""`two_to_four_index` has been renamed
# `two_to_four`; if you don't like it please file an issue :)""")
#
#
# function fourth_order_basis{T}(D::Array{T,2},a)
#    C = zeros(3,3,3,3)
#    Chat = zeros(3,3,3,3)
#
#    #Convert back to Tensor notation
#    for i=1:3, j = 1:3, k = 1:3, l = 1:3
#    	m = four_to_two(i,j)
#         n = four_to_two(k,l)
#         C[i,j,k,l] = D[m,n]
#    end
#
#    #Rotate the tensor to correct orientation
#    Tr = 1/sqrt(6)*[-sqrt(3) sqrt(3) 0; -sqrt(2) -sqrt(2) sqrt(2); 1 1 2]#1/sqrt(6)*[sqrt(3) 0 -sqrt(3); sqrt(2) sqrt(2) sqrt(2); 1 -2 1] Fix from 1/25
#    Q = zeros(3,3,3,3)
#    for i=1:3, j=1:3, k=1:3, l=1:3
# 	Q[i,j,k,l] = Tr[k,i]*Tr[l,j]
#    end
#
#
#    for i=1:3, j=1:3, k=1:3, l=1:3, g=1:3, h=1:3, m=1:3, n=1:3
# 	Chat[i,j,k,l] = Chat[i,j,k,l] + Q[g,h,i,j]*C[g,h,m,n]*Q[m,n,k,l]
#    end
#
#    M = zeros(6,6)
#    #Convert the tensor back to 6 by 6
#    for i=1:6, j=1:6
#         m, n = two_to_four(i)
#         p, q = two_to_four(j)
#         M[i,j] = Chat[m,n,p,q]
#    end
#
#   return M
# end
#
#
# function A_coefficients{T}(p::Array{Complex{Float64},1},D::Array{T,2})
#
#   A = Complex{Float64}[0 0 0; 0 0 0; 0 0 0]
#   x = Complex{Float64}[0; 0 ; 0]
#   y = Complex{Float64}[0; 0 ; 0]
#   w = Complex{Float64}[0; 0 ; 0]
#   z = Complex{Float64}[0; 0 ; 0]
#
#   for i=1:3
#     x[i] = D[1,4]^2 - D[4,4]*D[5,5]-(D[4,4]^2+D[2,2]*D[5,5])*( (real(p[i]))^2 - (imag(p[i]))^2) - D[4,4]*D[2,2]*( ((real(p[i]))^2 - (imag(p[i]))^2)^2 - 4*(real(p[i]))^2*(imag(p[i]))^2  )   #real B'
#     y[i] = -2*(D[2,2]*D[5,5]+D[4,4]^2)*real(p[i])*imag(p[i])-4*D[2,2]*D[4,4]*real(p[i])*imag(p[i])*( (real(p[i]))^2 - (imag(p[i]))^2)    #image B'
#     w[i] = 2*D[2,2]*D[1,4]*real(p[i])*( (real(p[i]))^2 - (imag(p[i]))^2)+D[1,4]*(D[4,4]-D[1,2])*real(p[i])-4*D[1,4]*D[2,2]*real(p[i])*(imag(p[i]))^2   #real B''
#     z[i] = 4*D[1,4]*D[2,2]*(real(p[i]))^2*imag(p[i])+2*D[2,2]*D[1,4]*imag(p[i])*( (real(p[i]))^2 - (imag(p[i]))^2)  +D[1,4]*(D[4,4]-D[1,2])*imag(p[i])  #image B''
#   end
#
#   for i=1:3
#     A[1,i] = (x[i]*w[i]+y[i]*z[i])/(w[i]^2+z[i]^2)+ ((y[i]*w[i]-x[i]*z[i])/ (w[i]^2+z[i]^2))*im
#     A[2,i] = 2*(imag(p[i])*imag(A[1,i]) - real(A[1,i])*real(p[i]))-(D[5,5]+D[4,4]*( (real(p[i]))^2 - (imag(p[i]))^2)  )/D[4,4] - (2*D[4,4]*real(p[i])*imag(p[i]) + 2*D[1,4]*(real(p[i])*imag(A[1,i])-imag(p[i])*real(A[1,i]) )  )*(1/D[1,4])*im
#     A[3,i] = 1+0*im
#   end
#
#   return A
#
# end
#
#
# function D_coefficients{T}(p::Array{Complex{Float64},1},D::Array{T,2}, A::Array{Complex{Float64},2}, b)
#
#   alpha = zeros(6,6)
#   v = zeros(6,1)
#   v[1] = b #first three components of v are burgers vector
#   for i=1:3
#     for j=1:6
#       l = ceil(j/2)
#       l = convert(Int,l)
#       if mod(j,2) == 0
#         alpha[i,j] = -imag(A[i,l])
#       else
#         alpha[i,j] = real(A[i,l])
#       end
#     end
#   end
#
#   for j=1:3
#     k = 2*j-1
#     m = 2*j
#     l = ceil(k/2)
#     l = convert(Int,l)
#     alpha[4,k] = D[6,6]*(real(A[1,l])*real(p[l])-imag(A[1,l])*imag(p[l])+real(A[2,l]) ) + D[5,6]
#     alpha[5,k] = D[1,2]*real(A[1,l])+D[2,2]*(real(A[2,l])*real(p[l]) - imag(A[2,l])*imag(p[l])  )
#     alpha[6,k] = D[1,4]*real(A[1,l])+D[4,4]*real(p[l])
#
#     alpha[4,m] = -D[6,6]*(real(A[1,l])*imag(p[l])+imag(A[1,l])*real(p[l])+imag(A[2,l])  )
#     alpha[5,m] = -D[1,2]*imag(A[1,l])-D[2,2]*(real(A[2,l])*imag(p[l])+imag(A[2,l])*real(p[l]))
#     alpha[6,m] = -D[1,4]*imag(A[1,l])-D[4,4]*imag(p[l])
#   end
#   D = \(alpha,v)
#   return D
# end
#
# function sextic_roots{T}(D::Array{T,2})
#
# #Comput coefficients of polynomial p^6 + k_4p^4 + k_2p^2 + k_0
#   k_4 = (D[1,1]*D[2,2]*D[4,4]+D[2,2]*D[4,4]*D[5,5]+D[4,4]^3-4*D[2,2]*D[1,4]^2-D[4,4]*(D[4,4]+D[1,2])^2)/(D[2,2]*D[4,4]^2)
#   print("k4: ")
#   print(k_4)
#   k_2 = (D[1,1]*D[2,2]*D[5,5]+D[1,1]*D[4,4]^2+D[5,5]*D[4,4]^2+4*D[1,2]*D[1,4]^2-D[1,4]^2*D[4,4]-D[5,5]*(D[1,2]+D[4,4])^2)/(D[2,2]*D[4,4]^2) #I think there was a mistake in published result
#   print("k2: ")
#   print(k_2)
#   k_0 = (D[1,1]*D[4,4]*D[5,5]-D[1,4]^2*D[1,1])/(D[2,2]*D[4,4]^2)
#   print("k0: ")
#   print(k_0)
#
# #Compute the roots p^2 = r of the sextic polynomial using general solution of cubic
#   Q =  (3*k_2 - k_4^2)/9
#   R =  (9*k_2*k_4-27*k_0-2*k_4^3)/54
#   E =  Q^3 + R^2
#   print("E: ")
#   print(E)
#   S = cbrt(R+sqrt(E))
#   U = cbrt(R-sqrt(E))
#   r_1 = -1/3*k_4 + (S + U) + 0*im
#   r_2 = -1/3*k_4-1/2*(S+U)+1/2*im*sqrt(3)*(S-U)
#   r_3 = -1/3*k_4-1/2*(S+U)-1/2*im*sqrt(3)*(S-U)
# #Now compute the roots p = \pm sqrt(r) making sure to take those with positive imaginary part
#   p = Complex{Float64}[0; 0 ; 0]
#
#   p[1] = sqrt(r_1)
#   p[2] = sqrt(r_2)
#   p[3] = sqrt(r_3)
#
#   for i=1:3
#     if imag(p[i]) < 0
#       p[i] = -p[i]
#     end
#   end
#
#
#   return p
# end
