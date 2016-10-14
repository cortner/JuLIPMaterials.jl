
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
      set_defm!(at, Ih * F0, updatepositions=false)
      Sp = stress(calc, at)
      Ih[i,a] -= 2*h
      set_defm!(at, Ih * F0, updatepositions=false)
      Sm = stress(calc, at)
      C[i, a, :, :] = (Sp - Sm) / (2*h)
      Ih[i,a] -= h
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

const voigtinds = [1, 5, 9, 6, 3, 2]

voigt_moduli{T}(C::Array{T,4}) = reshape(C, 9, 9)[voigtinds, voigtinds]




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
