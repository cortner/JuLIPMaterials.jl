
module Elasticity

using JuLIP: AbstractAtoms, AbstractCalculator, calculator,
         stress, defm, set_defm!


typealias Tensor{T} Array{T,4}

"""
* `elastic_moduli(at::AbstractAtoms) -> C::Tensor`
* `elastic_moduli(calc::AbstractCalculator, at::AbstractAtoms) -> C::Tensor`

computes the 3 x 3 x 3 x 3 elastic moduli tensor

*Notes:* this is a naive implementation that does not exploit
any symmetries at all; this means it performs 9 centered finite-differences
on the stress. The error should be in the range 1e-10
"""
elastic_moduli(at::AbstractAtoms) = elastic_moduli(calculator(at), at)

function elastic_moduli(calc::AbstractCalculator, at::AbstractAtoms)
   F0 = defm(at) |> Matrix
   Ih = eye(3)
   h = eps()^(1/3)
   C = zeros(3,3,3,3)
   for i = 1:3, a = 1:3
      Ih[i,a] += h
      set_defm!(at, Ih * F0, updatepositions=true)
      Sp = stress(calc, at)
      Ih[i,a] -= 2*h
      set_defm!(at, Ih * F0, updatepositions=true)
      Sm = stress(calc, at)
      C[i, a, :, :] = (Sp - Sm) / (2*h)
      Ih[i,a] += h
   end
   # symmetrise it - major symmetries C_{iajb} = C_{jbia}
   for i = 1:3, a = 1:3, j=1:3, b=1:3
      t = 0.5 * (C[i,a,j,b] + C[j,b,i,a])
      C[i,a,j,b] = t
      C[j,b,i,a] = t
   end
   # minor symmetries - C_{iajb} = C_{iabj}
   for i = 1:3, a = 1:3, j=1:3, b=1:3
      t = 0.5 * (C[i,a,j,b] + C[i,a,b,j])
      C[i,a,j,b] = t
      C[i,a,b,j] = t
   end
   return C
end

voigt_moduli(at::AbstractAtoms) = voigt_moduli(calculator(at), at)

voigt_moduli(calc::AbstractCalculator, at::AbstractAtoms) =
   voigt_moduli(elastic_moduli(calc, at))

#const voigtinds = [1, 5, 9, 6, 3, 2]

const voigtinds = [1, 5, 9, 4, 7, 8]

voigt_moduli{T}(C::Array{T,4}) = reshape(C, 9, 9)[voigtinds, voigtinds]



function four_to_two_index(i, j)

if (i == 1 && j == 1)
   dumb = 1
elseif (i == 2 && j ==2)
   dumb = 2
elseif (i == 3 && j == 3)
   dumb = 3
elseif ((i == 3 && j == 2) || (i == 2 && j == 3))
   dumb = 4
elseif ((i == 3 && j == 1) || (i == 1 && j == 3))
   dumb = 5
elseif ((i == 2 && j == 1) || (i == 1 && j == 2))
   dumb = 6
end

return dumb

end



function two_to_four_index(k)

it = 0
jit = 0

if k == 1
   it = 1;
   jit = 1;
elseif k == 2
   it = 2;
   jit = 2;
elseif k == 3
   it = 3;
   jit = 3;
elseif k == 4
   it = 3;
   jit = 2;
elseif k == 5
   it = 3;
   jit = 1;
elseif k == 6
   it = 2;
   jit = 1;
end
    
return it, jit

end

function fourth_order_basis{T}(D::Array{T,2},a)
   C = zeros(3,3,3,3)
   Chat = zeros(3,3,3,3)

   #Convert back to Tensor notation
   for i=1:3, j = 1:3, k = 1:3, l = 1:3
   	m = four_to_two_index(i,j)
        n = four_to_two_index(k,l)
        C[i,j,k,l] = D[m,n]    
   end

   #Rotate the tensor to correct orientation
   Tr = 1/sqrt(6)*[sqrt(3) 0 -sqrt(3); sqrt(2) sqrt(2) sqrt(2); 1 -2 1]
   Q = zeros(3,3,3,3)
   for i=1:3, j=1:3, k=1:3, l=1:3
	Q[i,j,k,l] = Tr[k,i]*Tr[l,j]
   end


   for i=1:3, j=1:3, k=1:3, l=1:3, g=1:3, h=1:3, m=1:3, n=1:3
	Chat[i,j,k,l] = Chat[i,j,k,l] + Q[g,h,i,j]*C[g,h,m,n]*Q[m,n,k,l]
   end

   M = zeros(6,6)
   #Convert the tensor back to 6 by 6
   for i=1:6, j=1:6
        m, n = two_to_four_index(i)
        p, q = two_to_four_index(j)
        M[i,j] = Chat[m,n,p,q]
   end

   #E = eye(3)
   #X = [1/sqrt(6) 1/sqrt(3) -1/sqrt(2); -2/sqrt(6) 1/sqrt(3) 0 ; 1/sqrt(6) 1/sqrt(3) 1/sqrt(2)]
   #X = transpose(X)
   
   #Q = zeros(3,3,3,3)
   #for i=1:3, j=1:3, k=1:3, l=1:3
   #  Q[i,j,k,l] = X[k,i]*X[l,j]
   #end


   #Now change the basis:
   #for i =1:3, j = 1:3, k = 1:3, l = 1:3
   #  for p =1:3, q = 1:3, r = 1:3, s = 1:3
       #Chat[i,j,k,l] = Chat[i,j,k,l] + C[p,q,r,s]*dot(E[:,p], X[:,i])*dot(E[:,q], X[:,j])*dot(E[:,r], X[:,k])*dot(E[:,s], X[:,l])
   #    Chat[i,j,k,l] = Chat[i,j,k,l] + Q[p,q,i,j]*C[p,q,r,s]*Q[r,s,k,l]
   #  end
   #end

  return M
end


function sextic_roots{T}(D::Array{T,2})

#Comput coefficients of polynomial p^6 + k_4p^4 + k_2p^2 + k_0
  k_4 = (D[1,1]*D[2,2]*D[4,4]+D[2,2]*D[4,4]*D[5,5]+D[4,4]^3-4*D[2,2]*D[1,4]^2-D[4,4]*(D[4,4]+D[1,2])^2)/(D[2,2]*D[4,4]^2)

  k_2 = (D[1,1]*D[2,2]*D[5,5]+D[1,1]*D[4,4]^2+D[5,5]*D[4,4]^2+4*D[1,2]*D[1,4]^2-D[1,4]^2*D[4,4]-D[5,5]*(D[1,2]+D[4,4])^2)/((D[2,2]*D[4,4])^2)

  k_0 = (D[1,1]*D[4,4]*D[5,5]-D[1,4]^2*D[1,1])/(D[2,2]*D[4,4]^2)

  #Test case
  k_4 = -1
  k_2 = 1
  k_0 = -1

#Compute the roots p = r^2 of the sextic polynomial using general solution of cubic
  Q =  (3*k_2 - k_4^2)/9
  R =  (9*k_2*k_4-27*k_0-2*k_4^3)/54
  E =  Q^3 + R^2
  S = cbrt(R+sqrt(E))
  U = cbrt(R-sqrt(E))
  r_1 = -1/3*k_4 + (S + U)
  r_2 = -1/3*k_4-1/2*(S+U)+1/2*im*sqrt(3)*(S-U)
  r_3 = -1/3*k_4-1/2*(S+U)-1/2*im*sqrt(3)*(S-U)

  return r_1, r_2, r_3
end

# """
# compute the Lame parameters for an elasticity tensor C or throw
# and error if the material is not isotropic.
# """
# function lame_parameters(C::Tensor)
#    error("lame_parameters is not yet implemented")
# end
#
#
# """
# check whether the elasticity tensor is isotropic; return true/false
# """
# function is_isotropic{T}(C::Tensor)
#    try
#       lame_parameters(C)
#       return true
#    catch
#       return false
#    end
# end
#
# """
# for an isotropic elasticity tensor return the poisson ratio
# """
# function poisson_ratio(C::Tensor)
#    λ, μ = lame_parameters(C)
#    return 0.5 * λ / (λ + μ)
# end


end
