
module CLE

using JuLIP: AbstractAtoms, AbstractCalculator, calculator,
             stress, defm, set_defm!

using StaticArrays

const Tensor{T} = Array{T, 4}

"""
* `elastic_moduli(at::AbstractAtoms)`
* `elastic_moduli(calc::AbstractCalculator, at::AbstractAtoms)`
* `elastic_moduli(C::Matrix)` : convert Voigt moduli to 4th order tensor

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
      C[i,a,j,b] = C[j,b,i,a] = 0.5 * (C[i,a,j,b] + C[j,b,i,a])
   end
   # minor symmetries - C_{iajb} = C_{iabj}
   for i = 1:3, a = 1:3, j=1:3, b=1:3
      C[i,a,j,b] = C[i,a,b,j] = 0.5 * (C[i,a,j,b] + C[i,a,b,j])
   end
   return C
end

"""
`voig_moduli`: compute elastic moduli in the format of Voigt moduli.

Methods:
* `voigt_moduli(at)`
* `voigt_moduli(calc, at)`
* `voigt_moduli(C)`
"""
voigt_moduli(at::AbstractAtoms) = voigt_moduli(calculator(at), at)

voigt_moduli(calc::AbstractCalculator, at::AbstractAtoms) =
   voigt_moduli(elastic_moduli(calc, at))

const voigtinds = [1, 5, 9, 4, 7, 8]

voigt_moduli{T}(C::Array{T,4}) = reshape(C, 9, 9)[voigtinds, voigtinds]


function elastic_moduli{T}(Cv::AbstractMatrix{T})
   @assert size(Cv) == (6,6)
   C = zeros(T, 9,9)
   C[voigtinds, voigtinds] = Cv
   C = reshape(C, 3,3,3,3)
   # now apply all the symmetries to recover C
   for i = 1:3, a = 1:3, j = 1:3, b = 1:3
      if C[i,a,j,b] != 0
         C[a,i,j,b] = C[i,a,b,j] = C[a,i,b,j] = C[j,b,i,a] =
            C[b,j,i,a] = C[j,b,a,i] = C[b,j,a,i] = C[i,a,j,b]
      end
   end
   return C
end

"""
`isotropic_moduli(λ, μ)`: compute 4th order tensor of elastic moduli
corresponding to the Lame parameters λ, μ.
"""
function isotropic_moduli(λ, μ)
   K = λ + μ * 2 / 3
   C = [ K * I[i,j] * I[k,l] + μ * (I[i,k]*I[j,l] + I[i,l]*I[j,k] - 2/3*I[i,j]*I[k,l])
         for i = 1:3, j = 1:3, k = 1:3, l = 1:3 ]
   return C
end


const _four_to_two_ = @SMatrix [1 6 5; 6 2 4; 5 4 3]

four_to_two(i, j) = _four_to_two_[i,j]

four_to_two_index(i, j) = error("""`four_to_two_index` has been renamed
 `four_to_two`; if you don't like it please file an issue :)""")


const _two_to_four_ = @SVector [(1,1), (2,2), (3,3), (3,2), (3,1), (2,1)]

two_to_four(k) = _two_to_four_[k]

two_to_four_index(k) = error("""`two_to_four_index` has been renamed
`two_to_four`; if you don't like it please file an issue :)""")


function fourth_order_basis{T}(D::Array{T,2},a)
   C = zeros(3,3,3,3)
   Chat = zeros(3,3,3,3)

   #Convert back to Tensor notation
   for i=1:3, j = 1:3, k = 1:3, l = 1:3
   	m = four_to_two(i,j)
        n = four_to_two(k,l)
        C[i,j,k,l] = D[m,n]
   end

   #Rotate the tensor to correct orientation
   Tr = 1/sqrt(6)*[-sqrt(3) sqrt(3) 0; -sqrt(2) -sqrt(2) sqrt(2); 1 1 2]#1/sqrt(6)*[sqrt(3) 0 -sqrt(3); sqrt(2) sqrt(2) sqrt(2); 1 -2 1] Fix from 1/25
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
        m, n = two_to_four(i)
        p, q = two_to_four(j)
        M[i,j] = Chat[m,n,p,q]
   end

  return M
end


function A_coefficients{T}(p::Array{Complex{Float64},1},D::Array{T,2})

  A = Complex{Float64}[0 0 0; 0 0 0; 0 0 0]
  x = Complex{Float64}[0; 0 ; 0]
  y = Complex{Float64}[0; 0 ; 0]
  w = Complex{Float64}[0; 0 ; 0]
  z = Complex{Float64}[0; 0 ; 0]

  for i=1:3
    x[i] = D[1,4]^2 - D[4,4]*D[5,5]-(D[4,4]^2+D[2,2]*D[5,5])*( (real(p[i]))^2 - (imag(p[i]))^2) - D[4,4]*D[2,2]*( ((real(p[i]))^2 - (imag(p[i]))^2)^2 - 4*(real(p[i]))^2*(imag(p[i]))^2  )   #real B'
    y[i] = -2*(D[2,2]*D[5,5]+D[4,4]^2)*real(p[i])*imag(p[i])-4*D[2,2]*D[4,4]*real(p[i])*imag(p[i])*( (real(p[i]))^2 - (imag(p[i]))^2)    #image B'
    w[i] = 2*D[2,2]*D[1,4]*real(p[i])*( (real(p[i]))^2 - (imag(p[i]))^2)+D[1,4]*(D[4,4]-D[1,2])*real(p[i])-4*D[1,4]*D[2,2]*real(p[i])*(imag(p[i]))^2   #real B''
    z[i] = 4*D[1,4]*D[2,2]*(real(p[i]))^2*imag(p[i])+2*D[2,2]*D[1,4]*imag(p[i])*( (real(p[i]))^2 - (imag(p[i]))^2)  +D[1,4]*(D[4,4]-D[1,2])*imag(p[i])  #image B''
  end

  for i=1:3
    A[1,i] = (x[i]*w[i]+y[i]*z[i])/(w[i]^2+z[i]^2)+ ((y[i]*w[i]-x[i]*z[i])/ (w[i]^2+z[i]^2))*im
    A[2,i] = 2*(imag(p[i])*imag(A[1,i]) - real(A[1,i])*real(p[i]))-(D[5,5]+D[4,4]*( (real(p[i]))^2 - (imag(p[i]))^2)  )/D[4,4] - (2*D[4,4]*real(p[i])*imag(p[i]) + 2*D[1,4]*(real(p[i])*imag(A[1,i])-imag(p[i])*real(A[1,i]) )  )*(1/D[1,4])*im
    A[3,i] = 1+0*im
  end

  return A

end


function D_coefficients{T}(p::Array{Complex{Float64},1},D::Array{T,2}, A::Array{Complex{Float64},2}, b)

  alpha = zeros(6,6)
  v = zeros(6,1)
  v[1] = b #first three components of v are burgers vector
  for i=1:3
    for j=1:6
      l = ceil(j/2)
      l = convert(Int,l)
      if mod(j,2) == 0
        alpha[i,j] = -imag(A[i,l])
      else
        alpha[i,j] = real(A[i,l])
      end
    end
  end

  for j=1:3
    k = 2*j-1
    m = 2*j
    l = ceil(k/2)
    l = convert(Int,l)
    alpha[4,k] = D[6,6]*(real(A[1,l])*real(p[l])-imag(A[1,l])*imag(p[l])+real(A[2,l]) ) + D[5,6]
    alpha[5,k] = D[1,2]*real(A[1,l])+D[2,2]*(real(A[2,l])*real(p[l]) - imag(A[2,l])*imag(p[l])  )
    alpha[6,k] = D[1,4]*real(A[1,l])+D[4,4]*real(p[l])

    alpha[4,m] = -D[6,6]*(real(A[1,l])*imag(p[l])+imag(A[1,l])*real(p[l])+imag(A[2,l])  )
    alpha[5,m] = -D[1,2]*imag(A[1,l])-D[2,2]*(real(A[2,l])*imag(p[l])+imag(A[2,l])*real(p[l]))
    alpha[6,m] = -D[1,4]*imag(A[1,l])-D[4,4]*imag(p[l])
  end
  D = \(alpha,v)
  return D
end

function sextic_roots{T}(D::Array{T,2})

#Comput coefficients of polynomial p^6 + k_4p^4 + k_2p^2 + k_0
  k_4 = (D[1,1]*D[2,2]*D[4,4]+D[2,2]*D[4,4]*D[5,5]+D[4,4]^3-4*D[2,2]*D[1,4]^2-D[4,4]*(D[4,4]+D[1,2])^2)/(D[2,2]*D[4,4]^2)
  print("k4: ")
  print(k_4)
  k_2 = (D[1,1]*D[2,2]*D[5,5]+D[1,1]*D[4,4]^2+D[5,5]*D[4,4]^2+4*D[1,2]*D[1,4]^2-D[1,4]^2*D[4,4]-D[5,5]*(D[1,2]+D[4,4])^2)/(D[2,2]*D[4,4]^2) #I think there was a mistake in published result
  print("k2: ")
  print(k_2)
  k_0 = (D[1,1]*D[4,4]*D[5,5]-D[1,4]^2*D[1,1])/(D[2,2]*D[4,4]^2)
  print("k0: ")
  print(k_0)

#Compute the roots p^2 = r of the sextic polynomial using general solution of cubic
  Q =  (3*k_2 - k_4^2)/9
  R =  (9*k_2*k_4-27*k_0-2*k_4^3)/54
  E =  Q^3 + R^2
  print("E: ")
  print(E)
  S = cbrt(R+sqrt(E))
  U = cbrt(R-sqrt(E))
  r_1 = -1/3*k_4 + (S + U) + 0*im
  r_2 = -1/3*k_4-1/2*(S+U)+1/2*im*sqrt(3)*(S-U)
  r_3 = -1/3*k_4-1/2*(S+U)-1/2*im*sqrt(3)*(S-U)
#Now compute the roots p = \pm sqrt(r) making sure to take those with positive imaginary part
  p = Complex{Float64}[0; 0 ; 0]

  p[1] = sqrt(r_1)
  p[2] = sqrt(r_2)
  p[3] = sqrt(r_3)

  for i=1:3
    if imag(p[i]) < 0
      p[i] = -p[i]
    end
  end


  return p
end

function zener_anisotropy_index(C::Tensor)
    Cv = voigt_moduli(C)
    A = 2*Cv[4,4]/(Cv[1,1] - Cv[1,2])
    return A
end
zener_anisotropy_index(at::AbstractAtoms) = zener_anisotropy_index(elastic_moduli(at))

"""
compute the Lame parameters for an elasticity tensor C or throw
an error if the material is not isotropic.
"""
function lame_parameters(C::Tensor; aniso_threshold=1e-3)
    A = zener_anisotropy_index(C)
    @assert abs(A - 1.0) < aniso_threshold
    Cv = voigt_moduli(C)
    μ = Cv[1,2]
    λ = Cv[4,4]
    return λ, μ
end
lame_parameters(at::AbstractAtoms) = lame_parameters(elastic_moduli(at))

poisson_ratio(λ, μ) = 0.5 * λ / (λ + μ)
poisson_ratio(C::Tensor) = poisson_ratio(lame_parameters(C)...)
poisson_ratio(at::AbstractAtoms) = poisson_ratio(elastic_moduli(at))

function youngs_modulus(λ, μ)
    ν = poisson_ratio(λ, μ)
    return λ*(1 + ν)*(1 - 2ν) / ν
end

youngs_modulus(C::Tensor) = youngs_modulus(lame_parameters(C)...)
youngs_modulus(at::AbstractAtoms) = youngs_modulus(elastic_moduli(at))

"""
check whether the elasticity tensor is isotropic; return true/false
"""
function is_isotropic(C::Tensor)
   try
      lame_parameters(C)
      return true
   catch
      return false
   end
end

end
