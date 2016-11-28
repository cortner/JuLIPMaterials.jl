
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

function fourth_order_basis{T}(D::Array{T,2},a)
   C = zeros(3,3,3,3)
   Chat = zeros(3,3,3,3)
   #Convert back to Tensor notation
   for k=1:3
     C[k,k,1,1] = C[1,1,k,k] = D[1,k]
     C[k,k,1,2] = C[1,2,k,k] = C[k,k,2,1] = C[2,1,k,k] = D[k,6]
     C[k,k,1,3] = C[1,3,k,k] = C[k,k,3,1] = C[3,1,k,k] = D[k,5]
     C[k,k,2,3] = C[2,3,k,k] = C[k,k,3,2] = C[3,2,k,k] = D[k,4]
     C[k,k,3,3] = C[3,3,k,k] = D[k,3]
   end
   
   C[2,3,1,2] = C[3,2,1,2] = C[2,3,2,1] =  C[1,2,2,3] = D[4,6]
   C[1,3,1,2] = C[3,1,1,2] = C[1,2,1,3] = C[1,3,2,1] = D[5,6]
   C[1,2,1,2] = C[2,1,1,2] = C[1,2,2,1] = D[6,6]
   C[2,2,2,2] = D[2,2]
   C[2,3,2,3] = C[3,2,2,3] = C[2,3,3,2] = D[4,4]
   C[2,3,1,3] = C[3,2,1,3] = C[1,3,2,3] = C[2,3,3,1] = D[4,5]
   C[1,3,1,3] = C[3,1,1,3] = C[1,3,3,1] = D[5,5]
   E = eye(3)
   X = [1/sqrt(6) 1/sqrt(3) -1/sqrt(2); -2/sqrt(6) 1/sqrt(3) 0 ; 1/sqrt(6) 1/sqrt(3) 1/sqrt(2)]
   X = transpose(X)
   
   Q = zeros(3,3,3,3)
   for i=1:3, j=1:3, k=1:3, l=1:3
     Q[i,j,k,l] = X[k,i]*X[l,j]
   end


   #Now change the basis:
   for i =1:3, j = 1:3, k = 1:3, l = 1:3
     for p =1:3, q = 1:3, r = 1:3, s = 1:3
       #Chat[i,j,k,l] = Chat[i,j,k,l] + C[p,q,r,s]*dot(E[:,p], X[:,i])*dot(E[:,q], X[:,j])*dot(E[:,r], X[:,k])*dot(E[:,s], X[:,l])
       Chat[i,j,k,l] = Chat[i,j,k,l] + Q[p,q,i,j]*C[p,q,r,s]*Q[r,s,k,l]
     end
   end 

   # symmetrise it - major symmetries C_{iajb} = C_{jbia}
   for i = 1:3, m = 1:3, j=1:3, b=1:3
      t = 0.5 * (Chat[i,m,j,b] + Chat[j,b,i,m])
      Chat[i,m,j,b] = t
      Chat[j,b,i,m] = t
   end
   # minor symmetries - C_{iajb} = C_{iabj}
   for i = 1:3, m = 1:3, j=1:3, b=1:3
      t = 0.5 * (Chat[i,m,j,b] + Chat[i,m,b,j])
      Chat[i,m,j,b] = t
      Chat[i,m,b,j] = t
   end

   Chat = reshape(Chat, 9, 9)[voigtinds, voigtinds]
   return Chat	
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
