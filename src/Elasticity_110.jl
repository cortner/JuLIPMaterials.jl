# TODO: this probably should be removed and all the functionality moved into `sextic.jl`

module Elasticity_110

using JuLIP: AbstractAtoms, AbstractCalculator, calculator,
         stress

using JuLIPMaterials.CLE: elastic_moduli, voigt_moduli

const Tensor{T} = Array{T,4}


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





function little_a(D::Array{T,2},r,s) where {T}

   C = zeros(3,3,3,3)
   Chat = zeros(3,3,3,3)

   #Convert back to Tensor notation
   for i=1:3, j = 1:3, k = 1:3, l = 1:3
   	m = four_to_two_index(i,j)
        n = four_to_two_index(k,l)
        C[i,j,k,l] = D[m,n]
   end

   #Rotate the tensor to correct orientation
   Tr = [1/sqrt(2) -1/sqrt(2) 0; 0 0 1; 1/sqrt(2) 1/sqrt(2) 0]
   Q = zeros(3,3,3,3)
   for i=1:3, j=1:3, k=1:3, l=1:3
	Q[i,j,k,l] = Tr[k,i]*Tr[l,j]
   end


   for i=1:3, j=1:3, k=1:3, l=1:3, g=1:3, h=1:3, m=1:3, n=1:3
	Chat[i,j,k,l] = Chat[i,j,k,l] + Q[g,h,i,j]*C[g,h,m,n]*Q[m,n,k,l]
   end

   #M = zeros(3,1)
   #M(1,1) = Chat[r,1,s,1]
   #M(2,1) = Chat[r,1,s,2] + Chat[r,2,s,1]
   #M(3,1) = Chat[r,2,s,2]
   #return M(1,1), M(2,1), M(3,1)
   p0 = Chat[r,1,s,1]
   p1 = Chat[r,1,s,2] + Chat[r,2,s,1]
   p2 = Chat[r,2,s,2]
   return p0, p1, p2
end



function fourth_order_basis(D::Array{T,2},a;
            Tr = [1/sqrt(2) -1/sqrt(2) 0; 0 0 1; 1/sqrt(2) 1/sqrt(2) 0]
				) where {T}
   C = zeros(3,3,3,3)
   Chat = zeros(3,3,3,3)

   #Convert back to Tensor notation
   for i=1:3, j = 1:3, k = 1:3, l = 1:3
   	m = four_to_two_index(i,j)
        n = four_to_two_index(k,l)
        C[i,j,k,l] = D[m,n]
   end

   #Rotate the tensor to correct orientation
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

  return M
end


function A_coefficients(p::Array{Complex{Float64},1},D::Array{T,2}) where {T}

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


function D_coefficients(p::Array{Complex{Float64},1},D::Array{T,2},
						      A::Array{Complex{Float64},2}, b) where {T}

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

function sextic_roots(D::Array{T,2}) where {T}


  inter = D[1,1]*D[2,2]-2*D[1,2]*D[6,6]-D[1,2]^2
  lead = D[2,2]*D[4,4]*D[6,6]
  k_0 = D[5,5]*D[6,6]*D[1,1]/lead

  k_2 = (D[4,4]*D[1,1]*D[6,6] + D[5,5]*inter)/lead

  k_4 = (D[4,4]*inter+D[5,5]*D[2,2]*D[6,6])/lead

#Compute the roots p^2 = r of the sextic polynomial using general solution of cubic
  Q =  (3*k_2 - k_4^2)/9
  R =  (9*k_2*k_4-27*k_0-2*k_4^3)/54
  E =  Q^3 + R^2
  #print("E: ")
  #print(E)
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


end
