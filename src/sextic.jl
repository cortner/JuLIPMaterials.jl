

include("Elasticity_110.jl")
using JuLIPMaterials.CLE.Elasticity_110:
      elastic_moduli,
      voigt_moduli,
      fourth_order_basis,
      sextic_roots,
      A_coefficients,
      D_coefficients,
      little_a

using JuLIP.Potentials: fcut, fcut_d



u_edge_fcc_110(x, y, b, C::Array{T,4}; TOL=1e-4) where {T} =
         u_edge(x, y, b, voigt_moduli(C), TOL=TOL)

"""
* `u_edge(x, y, b, C) -> ux, uy`

* `C` can be either 3 x 3 x 3 x 3 (`elastic_moduli`) or 6 x 6 (`voigt_moduli`)

This function computes the anisotropic CLE solution for an in-plane
edge dislocation. The elastic moduli are taken to within `TOL` accuracy (
   this is a keyword argument)
"""
function u_edge_fcc_110(x, y, b, Cv::Array{T,2}, a; TOL = 1e-4) where {T}
   Cv = copy(Cv)
   test1 = Cv[1,1]
   test2 = Cv[1,2]
   test3 = Cv[6,6]
   # >>>>>>>>> START DEBUG >>>>>>>>
   #Zero out elasticity tensor from any numerical artifacts
   Cv = zeros(6,6)
   Cv[1,1] = 1.0*test1
   Cv[1,2] = 1.0*test2
   Cv[6,6] = 1.0*test3
   Cv[2,2] = Cv[3,3] = Cv[1,1]
   Cv[1,3] = Cv[2,3] = Cv[2,1] = Cv[3,1] = Cv[3,2] = Cv[1,2]
   Cv[4,4] = Cv[5,5] = Cv[6,6]
   # <<<<<<<<< END DEBUG <<<<<<<<<

   #Compute Anisotropic solution from Chou and Sha, J. App. Phys 42 (7) 2625
   #This uses the elasticity tensor in the usual coordinates.
   #Note that the rotated tensor agrees with the values shown on 2625
   #Same problem as with Hirth and Lothe--either formulas are incorrect or problem
   #with numeric instability.  Complex log formulas seem to work
   bar = sqrt( .5*Cv[1,1]*(Cv[1,1] +Cv[1,2] + 2.0*Cv[4,4]) )
   C = ( (bar + Cv[1,2])*(bar - Cv[1,2] - 2.0*Cv[4,4] )  )/(bar*Cv[4,4])
   lem = (bar/Cv[1,1])^(1/2)

   delta2 = sqrt(-C)
   delta1 = sqrt(C+4)

   #Now compute using Hirth and Lothe which should be valid for 110 dislocation
   #This should use K instead of Cv
   #Something seems to be either wrong with these formulas or numeric issues are arising
   #Skip to general formula using complex logs

   K = fourth_order_basis(Cv,a)
   Cvoigt = round.(K, digits=3)
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
   p[3,1] = -p[2,1]

   ux = real.( im/(2*π)*(A[1,2]*(D[3] + D[4]*im)*log.(x+p[2]*y) + A[1,3]*(D[5] + D[6]*im)*log.(x+p[3]*y)   ))

   uy = real.( im/(2*π)*(A[2,2]*(D[3] + D[4]*im)*log.(x+p[2]*y) + A[2,3]*(D[5] + D[6]*im)*log.(x+p[3]*y)   ))

   return ux, uy
end




struct EdgeCubic{T1,T2,T3,T4,T5}
    A::Matrix{T1}
    D::Vector{T2}
    p::Vector{T3}
    x0::T4
    b::T5
end

function EdgeCubic(b::Real, Cv::Array{T,2}, a::Real; x0 = zeros(3)) where {T}
    # clean up the tensor
   test1 = Cv[1,1]
   test2 = Cv[1,2]
   test3 = Cv[6,6]
   Cv = zeros(6,6)
   Cv[1,1] = test1
   Cv[1,2] = test2
   Cv[6,6] = test3
   #Zero out elasticity tensor from any numerical artifacts
   Cv[2,2] = Cv[3,3] = Cv[1,1]
   Cv[1,3] = Cv[2,3] = Cv[2,1] = Cv[3,1] = Cv[3,2] = Cv[1,2]
   Cv[4,4] = Cv[5,5] = Cv[6,6]

   # parameters needed for the Hirth/Lothe solution
   bar = sqrt( .5*Cv[1,1]*(Cv[1,1] +Cv[1,2] + 2.0*Cv[4,4]) )
   C = ( (bar + Cv[1,2])*(bar - Cv[1,2] - 2.0*Cv[4,4] )  )/(bar*Cv[4,4])
   lem = (bar/Cv[1,1])^(1/2)
   delta2 = sqrt(-C)
   delta1 = sqrt(Complex(C+4))

   # Now compute using Hirth and Lothe which should be valid for 110 dislocation
   # This should use K instead of Cv
   # Something seems to be either wrong with these formulas or numeric issues are arising
   # Skip to general formula using complex logs
   K = fourth_order_basis(Cv,a; Tr = [0 1/√2  -1/√2; 1 0 0; 0 1/√2 1/√2])
   Cvoigt = round.(K, digits=8)

   # more parameters
   c̄11 = sqrt(K[1,1]*K[2,2])    # (13-106)
   lam = (K[1,1]/K[2,2])^(1/4)
   ϕ = 0.5 * acos( (K[1,2]^2 + 2*K[1,2]*K[6,6] - c̄11^2) / (2.0*c̄11*K[6,6]) )
   apple = - lam*(K[6,6]*exp(im*ϕ) + c̄11*exp(-im*ϕ))/(K[1,2]+K[6,6])
   dodo = im*b[1,1]/(2.0*c̄11*sin(2*ϕ))*(K[1,2] - c̄11*exp(2*im*ϕ))

   #Should test this against solving the full linear system
   A = Complex{Float64}[0 1 1; 0 apple -apple; 0 0 0]
   #Set up for burgers vector in x1 direction only
   D = zeros(6,1)
   D = [ 0; 0; real(dodo); imag(dodo); -real(dodo); -imag(dodo)]
   p = sextic_roots(K)
   #Do some funny business with root switching here.  This could be what is screwing things up.
   #May have to change the roots elsewhere instead
   p[3,1] = -p[2,1]

   return EdgeCubic(A, D, p, x0[1:2], b)
end


#Create function to smoothly blend from 0 to 1
eta(Y::AbstractVector, r0=0.1, r1=1.8) = 1.0 - fcut(norm(Y), r0, r1)

grad_eta(Y::AbstractVector, r0=0.1, r1=1.8) = - fcut_d(norm(Y), r0, r1) * (Y/norm(Y))

#xi correction solver from Ehrlacher, Ortner, Shapeev
#
# TODO: we could make this robust
#
function xi_solver(Y::Vector, b::Float64; TOL = 1e-9, maxnit = 500)
    ξ1(x::Real, y::Real, b) = x - b * eta(Y) * angle(x + im * y) / (2*π)
    dξ1(x::Real, y::Real, b) = 1 + b * eta(Y) * y / (x^2 + y^2) / (2*π) + b * grad_eta(Y)[1] * angle(x + im * y)/ (2*π)
    y = Y[2]
    x = y
    for n = 1:maxnit
        f = ξ1(x, y, b) - Y[1]
        if abs(f) <= TOL; break; end
        x = x - f / dξ1(x, y, b)
    end
    if abs(ξ1(x, y, b) - Y[1]) > TOL
        warn("newton solver did not converge; returning input")
        return Y
    end
    return [x, y]
end

#return the inverse of the xi derivative
function xi_deriv_inv(Y::Vector, b::Float64)
    dξ1(x::Real, y::Real, b) = 1 + b * y / (x^2 + y^2) / (2*π) + b * grad_eta(Y)[1] * angle(x + im * y)/ (2*π)
    dξ2(x::Real, y::Real, b) = -b * x / (x^2 + y^2) / (2*π) + b * grad_eta(Y)[2] * angle(x + im * y)/ (2*π)
    D_xi_inv = (1/dξ1(Y[1],Y[2], b))*[1 -dξ2(Y[1],Y[2], b) 0;0 dξ1(Y[1],Y[2], b) 0; 0 0 0]
    return D_xi_inv
end


function evaluate(U::EdgeCubic, X::AbstractVector)
    x = X[1] - U.x0[1]
    y = X[2] - U.x0[2]
    x, y = xi_solver([x'; y'], -U.b)
    A, D, p = U.A, U.D, U.p
    ux = real( im/(2*π)*(A[1,2]*(D[3] + D[4]*im)*log(x+p[2]*y) + A[1,3]*(D[5] + D[6]*im)*log(x+p[3]*y) ))
    uy = real( im/(2*π)*(A[2,2]*(D[3] + D[4]*im)*log(x+p[2]*y) + A[2,3]*(D[5] + D[6]*im)*log(x+p[3]*y) ))
    return [ux, uy, 0.0]
end

function jacobian(U::EdgeCubic, X::AbstractVector)
    x = X[1] - U.x0[1]
    y = X[2] - U.x0[2]
    x, y = xi_solver([x'; y'], -U.b)
    D_xi_inv = xi_deriv_inv([x'; y'], -U.b)
    A, D, p = U.A, U.D, U.p
    uxx = real( im/(2*π)*(A[1,2]*(D[3] + D[4]*im)/(x+p[2]*y) + A[1,3]*(D[5] + D[6]*im)/(x+p[3]*y) ) )
    uxy = real( im/(2*π)*(A[1,2]*(D[3] + D[4]*im)*p[2]/(x+p[2]*y) + A[1,3]*(D[5] + D[6]*im)*p[3]/(x+p[3]*y) ) )
    uyx = real( im/(2*π)*(A[2,2]*(D[3] + D[4]*im)/(x+p[2]*y) + A[2,3]*(D[5] + D[6]*im)/(x+p[3]*y)   ))
    uyy = real( im/(2*π)*(A[2,2]*(D[3] + D[4]*im)*p[2]/(x+p[2]*y) + A[2,3]*(D[5] + D[6]*im)*p[3]/(x+p[3]*y)   ))
    return [uxx uxy 0.0; uyx uyy 0.0; 0.0 0.0 0.0]*D_xi_inv
end

function (U::EdgeCubic)(X::AbstractVector)
    return evaluate(U, X), jacobian(U, X)
end
