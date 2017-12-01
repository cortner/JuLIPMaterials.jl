
# TODO: this is really a module for face-centered diamond-cubic
#       and over time it should become much more general of course

module Si

using JuLIP, JuLIP.ASE, JuLIP.Potentials
import MaterialsScienceTools.CauchyBorn
using MaterialsScienceTools.CLE: elastic_moduli, voigt_moduli,
         fourth_order_basis, sextic_roots, A_coefficients, D_coefficients,
         little_a


"""
`fcc_edge_plane(s::AbstractString) -> at::ASEAtoms, b, xcore `

Generates a unit cell for an FCC crystal with orthogonal cell vectors chosen
such that the F1 direction is the burgers vector and the F3 direction the normal
to the standard edge dislocation:
   b = F1 ~ a1;  ν = F3 ~ a2-a3
The third cell vector: F2 ~ a * e1. Here, ~ means they are rotated from the
a1, a2, a3 directions. This cell contains two atoms.

Returns
* `at`: Atoms object
* `b` : Burgers vector
* `xcore` : a core-offset (to add to any lattice position)
"""
function fcc_edge_plane(s::AbstractString)
   # ensure s is actually an FCC species
   # check_fcc(s)
   # get the cubic unit cell dimension
   a = ( bulk(s, cubic=true) |> defm )[1,1]
   #print(a)
   # construct the cell matrix
   F = a*JMat( [ sqrt(2)/2 0    0;
                 0   1     0;
                 0   0    sqrt(2)/2 ] )
   X = a*[ JVec([0.0, 0.0, 0.0]),
         JVec([(1/2)*1/sqrt(2),1/2, 1/(2*sqrt(2))]), JVec([0, -1/4, 1/(2*sqrt(2))]),JVec([(1/2)*1/sqrt(2), 1/4, 0]) ]
   # construct ASEAtoms
   at = ASEAtoms(string(s,"4"))
   set_defm!(at, F)
   set_positions!(at, X)
   # compute a burgers vector in these coordinates
   b =  a*sqrt(2)/2*JVec([1.0,0.0,0.0])
   # compute a core-offset (to add to any lattice position)
   xcore = [-.7, 1.0, 0] # a*sqrt(2)/2 * JVec([1/2, -1/3, 0])  # [1/2, 1/3, 0]
   # return the information
   return at, b, xcore, a
end


project12(x) = SVec{2}([x[1],x[2]])


"""
`fcc_edge_geom(s::AbstractString, R::Float64) -> at::ASEAtoms`

generates a linear elasticity predictor configuration for
an edge dislocation in FCC, with burgers vector ∝ e₁  and dislocation
core direction ν ∝ e₃
"""
function fcc_edge_geom(s::AbstractString, R;
                       truncate=true, cle=:isotropic, ν=0.25, calc=nothing,
                       TOL=1e-4)
   # compute the correct unit cell
   atu, b, xcore, a = fcc_edge_plane(s)
   # multiply the cell to fit a ball of radius a/√2 * R inside
   L1 = ceil(Int, 2*R) + 3
   L2 = ceil(Int, 2*R/√2) + 3
   at = atu * (L1,L2, 1)
   atp = atu * (L1,L2, 6)
   # mess with the data
   # turn the Burgers vector into a scalar
   @assert b[2] == b[3] == 0.0
   b = b[1]
   # compute a dislocation core
   xcore = project12(xcore)
   X12 = project12.(positions(at))
   # compute x, y coordinates relative to the core
   x, y = mat(X12)[1,:], mat(X12)[2,:]
   xc, yc = mean(x), mean(y)
   r² = (x-xc).^2 + (y-yc).^2
   I0 = find( r² .== minimum(r²) )[1]
   xcore = X12[I0] + xcore
   x, y = x - xcore[1], y - xcore[2]
   # compute the dislocation predictor
   if cle == :isotropic
      ux, uy = u_edge_isotropic(x, y, b, ν)
   elseif cle == :anisotropic
      # TODO: this doesn't look clean; maybe we need to pass atu in the future
      # I'm not fully understanding how the function fcc_edge_plane(s) works
      set_pbc!(atu, true)
      atv = bulk("Si", cubic=true) * 4
      Cv = voigt_moduli(calc, atv)
      ux, uy = u_edge(x, y, b, Cv, a, TOL=TOL)
   else
      error("unknown `cle`")
   end
   # apply the linear elasticity displacement
   X = positions(at) |> mat
   X[1,:], X[2,:] = x + ux + xcore[1], y + uy + xcore[2]
   # if we want a circular cluster, then truncate to an approximate ball (disk)
   if truncate
      F = defm(at) # store F for later use
      X = vecs(X)  # find points within radius
      IR = find( [vecnorm(x[1:2] - xcore) for x in X] .<= R * a/√2 )
      X = X[IR]
      at = ASEAtoms("$s$(length(X))")  # generate a new atoms object
      set_defm!(at, F)                 # and insert the old cell shape
   end
   # update positions in Atoms object, set correct BCs and return
   set_positions!(at, X)
   return at, xcore
end



"""
`u_edge_isotropic(x, y, b, ν) -> u_x, u_y`

compute the displacement field `ux, uy` for an edge dislocation in an
isotropic linearly elastic medium, with core at (0,0),
burgers vector `b * [1.0;0.0]` and Poisson ratio `ν`

This is to be used primarily for comparison, since the exact solution will
not be the isotropic elasticity solution.
"""
function u_edge_isotropic(x, y, b, ν)
   x[y .< 0] += b/2
   r² = x.^2 + y.^2
   ux = b/(2*π) * ( atan(x ./ y) + (x .* y) ./ (2*(1-ν) * r²) )
   uy = -b/(2*π) *( (1-2*ν)/(4*(1-ν)) * log(r²) + (y.^2 - x.^2) ./ (4*(1-ν) * r²) )
   return ux, uy
end


u_edge{T}(x, y, b, C::Array{T,4}; TOL=1e-4) = u_edge(x, y, b, voigt_moduli(C), TOL=TOL)

"""
* `u_edge{T}(x, y, b, C::Array{T,N}) -> ux, uy`

* `C` can be either 3 x 3 x 3 x 3 (`elastic_moduli`) or 6 x 6 (`voigt_moduli`)

This function computes the anisotropic CLE solution for an in-plane
edge dislocation. The elastic moduli are taken to within `TOL` accuracy (
   this is a keyword argument)
"""
function u_edge{T}(x, y, b, Cv::Array{T,2}, a; TOL = 1e-4)
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
   #Cv[1,1] = 16.57
   #Cv[1,2] = 6.39
   #Cv[6,6] = 7.96
   # >>>>>>>>> START DEBUG >>>>>>>>
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

   bar = sqrt( .5*Cv[1,1]*(Cv[1,1] +Cv[1,2] + 2.0*Cv[4,4]) )
   C = ( (bar + Cv[1,2])*(bar - Cv[1,2] - 2.0*Cv[4,4] )  )/(bar*Cv[4,4])
   lem = (bar/Cv[1,1])^(1/2)
   #print("C :")
   #print(C)

   delta2 = sqrt(-C)
   delta1 = sqrt(C+4)

   q = (x.^4 + 2 * x.^2 .* y.^2 * lem^2 + y.^4 * lem^4) + (C*lem^2 * x.^2 .*  y.^2)

   ux1 = (b / (4.0*pi)) * (
          atan( (x.*y*lem*delta1) ./ (x.^2 - lem^2*y.^2) )
          + (bar^2 - Cv[1,2]^2) / (bar*Cv[4,4]*delta1*delta2) * ( atanh( (x.*y*lem*delta2) ./ (x.^2 + lem^2*y.^2) ) )
          )
   uy1 = (-lem*b)/(4.0*pi) * (
         (bar - Cv[1,2])/(2*delta1*bar)*log(q)
          - (bar + Cv[1,2])/(delta2*bar) *
                  atan( (y.^2*lem^2*delta1*delta2) ./ (2.0*x.^2 + (C + 2.0)*lem^2*y.^2 ) )
       )

   #Now compute using Hirth and Lothe which should be valid for 110 dislocation
   #This should use K instead of Cv

   K = fourth_order_basis(Cv,a)
   Cvoigt = round(K, 3)
   #print("We are here !")
   #print(Cvoigt)

   c̄11 = sqrt(K[1,1]*K[2,2])    # (13-106)
   lam = (K[1,1]/K[2,2])^(1/4)
     ϕ = 0.5 * acos( (K[1,2]^2 + 2*K[1,2]*K[6,6] - c̄11^2) / (2.0*c̄11*K[6,6]) )
   #print("lambda: ")
   #print(lam)
   #print(" phi : ")
   #print(ϕ)
   apple = - lam*(K[6,6]*exp(im*ϕ) + c̄11*exp(-im*ϕ))/(K[1,2]+K[6,6])
   #print(" apple: ")
   #print(apple)
   #print("exp(im phi) : ")
   #print(exp(2*im*ϕ))
   dodo = im*b[1,1]/(2.0*c̄11*sin(2*ϕ))*(K[1,2] - c̄11*exp(2*im*ϕ))
   #print(" bx : ")
   #print(b[1,1])
   #print(" dodo : ")
   #print(dodo)
   #print(" test : ")
   #print(dodo*(K[1,2] + K[2,2]*apple*lam*exp(im*ϕ)))
   q² = x.^2 + 2 * x .* y * lam * cos(ϕ) + y.^2 * lam^2
   t² = x.^2 - 2 * x .* y * lam * cos(ϕ) + y.^2 * lam^2
   ux2 = - (b / (4*π)) * (
          atan( (2*x.*y*lam*sin(ϕ)) ./ (x.^2 - lam^2*y.^2) )
          + (c̄11^2 - K[1,2]^2) / (2*c̄11*K[6,6]*sin(2*ϕ)) * (0.5 * log(q²./t²))
          )
   uy2 = (lam*b)/(4*π*c̄11*sin(2*ϕ)) * (
         (c̄11 - K[1,2]) * cos(ϕ) * (0.5 * log(q².*t²))
          - (c̄11 + K[1,2]) * sin(ϕ) *
                  atan( (y.^2*lam^2*sin(2*ϕ)) ./ (x.^2 - lam^2 * y.^2 * cos(2*ϕ)) )
       )



   #All of this is whacked up.  Let's try using a slightly more general approach
   #p0, p1, p2 = little_a(Cv,3,3)
   #print("constant: ")
   #print(p0)
   #print(" linear: ")
   #print(p1)
   #print(" quadratic: ")
   #print(p2)

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
   ux = real( im/(2*π)*(A[1,2]*(D[3] + D[4]*im)*log(x+p[2]*y) + A[1,3]*(D[5] + D[6]*im)*log(x+p[3]*y)   ))

   uy = real( im/(2*π)*(A[2,2]*(D[3] + D[4]*im)*log(x+p[2]*y) + A[2,3]*(D[5] + D[6]*im)*log(x+p[3]*y)   ))

   @assert isreal(ux)
   @assert isreal(uy)

   return ux, uy
end





"a fully equilibrated SW potential"
function sw_eq()
    T(σ, at) = trace(stress(StillingerWeber(σ=σ), at))
    at = JuLIP.ASE.bulk("Si", pbc=true)
    r0 = 2.09474
    r1 = r0 - 0.1
    s0, s1 = T(r0, at), T(r1, at)
    while (abs(s1) > 1e-8) && abs(r0 - r1) > 1e-8
        rnew = (r0 * s1 - r1 * s0) / (s1 - s0)
        r0, r1 = r1, rnew
        s0, s1 = s1, T(rnew, at)
    end
#     @show r1
    return StillingerWeber(σ=r1)
end


function si_plane(R)
    @assert isodd(R)   # TODO: why?
    atu, b, xcore, a = fcc_edge_plane("Si")   # TODO: call from FCC module
    xcore = [0.0, 0.625 * a, 0.0]
    at = atu * (R, R, 1)
    set_pbc!(at, (false, false, true))
    b = b[1]
    X = positions(at) |> mat
    X = vecs(X)
    # Old choice of xcore
    xcore1 = X[length(X) ÷ 2 + 1] + xcore
    # This choice picks the lower left and upper right atom (not site) positions
    #   TODO I don't like the 0.1
    xcore = (1/2)*(X[length(X)-2]+X[3]) +[-1, 0.1, 0]
    return at, b, xcore
end


# we now want to take the following example but apply
# the minimised shifts

# at, _, _ = si_plane(9)
# calc = sw_eq()
# x0 = JVecF([-2.0, -2.0, 0.0])
# X = [ x + 0.4 * [((x-x0)/norm(x-x0))[1:2]; 0.0] for x in positions(at)]
# set_positions!(at, X)

"""
a function that identifies multi-lattice structure in 2 layers of bulk-Si
(yes - very restrictive but will do for now!)
"""
function si_multilattice(at)
    J0 = Int[]
    J1 = Int[]
    Jdel = Int[]
    for (i, j, r, R, _) in sites(at, rnn("Si")+0.1)
        foundneig = false
        for (jj, RR) in zip(j, R)
            if (RR[1] == 0.0) && (abs(RR[2] - 1.3575) < 1e-3)
                # neighbour above >> make (i, jj) a site
                push!(J0, i)
                push!(J1, jj)
                foundneig = true
                break
            elseif (RR[1] == 0.0) && (abs(RR[2] + 1.3575) < 1e-3)
                # neighbour below >> (jj, i) is a site that will be pushed when i ↔ jj
                foundneig = true
                break
            end
        end
        if !foundneig
            # i has no neighbour above or below >> probably we just get rid of it
            push!(Jdel, i)
        end
    end
    return J0, J1, Jdel
end



function symml_displacement!(at, u)
    I0, I1, Idel = si_multilattice(at)
    @assert isempty(Idel)  # if Idel is not empty then (for now) we don't know what to do
    X = positions(at)
    W = CauchyBorn.WcbQuad()
    F0 = defm(W.at)
    p0 = W(F0)
    # transformation matrices
    Tp = [0 1/√2  -1/√2; 1 0 0; 0 1/√2 1/√2]
    Tm = diagm([1,1,-1]) * Tp

    for (i0, i1) in zip(I0, I1)   # each pair (i0, i1) corresponds to a ML lattice site
        x0, x1 = X[i0], X[i1]
        x1[3] > x0[3] ? T = Tp : T = Tm
        x̄ = 0.5 * (x0 + x1)   # centre of mass of the bond
        U, ∇U = u(x̄)          # displacement and displacement gradient
        F = T' * (I + ∇U) * T
        q = T * (W(F * F0) - p0)    # construct the shift corresponding to F = Id + ∇U
        X[i0], X[i1] = x0 + U - 0.5 * q, x1 + U + 0.5 * q
    end
    set_positions!(at, X)
    return at
end

function ml_displacement!(at, u)
    I0, I1, Idel = si_multilattice(at)
    @assert isempty(Idel)  # if Idel is not empty then (for now) we don't know what to do
    X = positions(at)
    W = CauchyBorn.WcbQuad()

    # transformation matrices
    Tp = [0 1/√2  -1/√2; 1 0 0; 0 1/√2 1/√2]
    Tm = diagm([1,1,-1]) * Tp

    F0 = defm(W.at)  # get reference information
    p0 = W(F0)

    for (i0, i1) in zip(I0, I1)   # each pair (i0, i1) corresponds to a ML lattice site
        x0, x1 = X[i0], X[i1]
        x1[3] > x0[3] ? T = Tp : T = Tm
        U, ∇U = u(x0)            # displacement and displacement gradient
        F = T' * (I + ∇U) * T
        q = T * (W(F * F0) - p0)   # construct the shift corresponding to F = Id + ∇U
        X[i0], X[i1] = x0 + U, x1 + U + q
    end
    set_positions!(at, X)
    return at
end


module Edge

import MaterialsScienceTools, ForwardDiff
using QuadGK

type EdgeCubic{T1,T2,T3,T4,T5}
    A::Matrix{T1}
    D::Vector{T2}
    p::Vector{T3}
    x0::T4
    b::T5
end

function EdgeCubic{T}(b::Real, Cv::Array{T,2}, a::Real)
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
   K = MaterialsScienceTools.CLE.Elasticity_110.fourth_order_basis(Cv,a;
                                                    Tr = [0 1/√2  -1/√2; 1 0 0; 0 1/√2 1/√2])
   Cvoigt = round.(K, 8)

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
   p = MaterialsScienceTools.CLE.Elasticity_110.sextic_roots(K)
   #Do some funny business with root switching here.  This could be what is screwing things up.
   #May have to change the roots elsewhere instead
   p[3,1] = -p[2,1]

   return EdgeCubic(A, D, p, zeros(2), b)
end


#Create function to smoothly blend from 0 to 1
function eta(Y::Vector; rHat = 10)
    v = norm(Y)/rHat
    g(x) = exp(-1/((x-x^2)))
    intX(x) = quadgk(g, 0, x)
    area = intX(1)[1]
    if v-1 > 1e-12
        eta = 1
    elseif v < 1e-12
        eta = 0
    else
        eta = intX(v)[1]/area
    end
    return eta
end

function d_eta_x(Y::Vector; rHat = 10)
    v = norm(Y)/rHat
    g(x) = exp(-1/((x-x^2)))
    intX(x) = quadgk(g, 0, x)
    area = intX(1)[1]
    if v-1 > 1e-12
        d_eta = 0
    elseif v < 1e-12
        d_eta = 0
    else
        d_eta = (g(v)/area)*(1/norm(Y))*(Y[1]/rHat)
    end
    return d_eta
end

function d_eta_y(Y::Vector; rHat = 10)
    v = norm(Y)/rHat
    g(x) = exp(-1/((x-x^2)))
    intX(x) = quadgk(g, 0, x)
    area = intX(1)[1]
    if v-1 > 1e-12
        d_eta = 0
    elseif v < 1e-12
        d_eta = 0
    else
        d_eta = (g(v)/area)*(1/norm(Y))*(Y[2]/rHat)
    end
    return d_eta
end
#xi correction solver from Ehrlacher, Ortner, Shapeev
function xi_solver(Y::Vector, b::Float64; TOL = 1e-9, maxnit = 500)
    ξ1(x::Real, y::Real, b) = x - b * eta(Y) * angle(x + im * y) / (2*π)
    dξ1(x::Real, y::Real, b) = 1 + b * eta(Y) * y / (x^2 + y^2) / (2*π) + b * d_eta_x(Y) * angle(x + im * y)/ (2*π)
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
    dξ1(x::Real, y::Real, b) = 1 + b * y / (x^2 + y^2) / (2*π) + b * d_eta_x(Y) * angle(x + im * y)/ (2*π)
    dξ2(x::Real, y::Real, b) = -b * x / (x^2 + y^2) / (2*π) + b * d_eta_y(Y) * angle(x + im * y)/ (2*π)
    D_xi_inv = (1/dξ1(Y[1],Y[2], b))*[1 -dξ2(Y[1],Y[2], b) 0;0 dξ1(Y[1],Y[2], b) 0; 0 0 0]
    return D_xi_inv
end


function evaluate(U::EdgeCubic, X::AbstractVector)
    x = X[1] - U.x0[1]
    y = X[2] - U.x0[2]
    x, y = xi_solver([x'; y'], -U.b)
    A, D, p = U.A, U.D, U.p
    ux = real( im/(2*π)*(A[1,2]*(D[3] + D[4]*im)*log(x+p[2]*y) + A[1,3]*(D[5] + D[6]*im)*log(x+p[3]*y)   ))
    uy = real( im/(2*π)*(A[2,2]*(D[3] + D[4]*im)*log(x+p[2]*y) + A[2,3]*(D[5] + D[6]*im)*log(x+p[3]*y)   ))
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

# a little test code for `module Edge`
# x0 = rand(2)
# u, ∂u = U(x0)
# for p  = 3:10
#     h = 0.1^p
#     gh = zeros(2,2)
#     for i = 1:2
#         x0[i] += h
#         uh, _ = U(x0)
#         gh[:,i] = (uh - u) / h
#         x0[i] -= h
#     end
#     println("p = $p: err = ", vecnorm(∂u - gh, Inf))
# end

end



function edge110(species::AbstractString, R::Real;
                  truncate=true, cle=:anisotropic, ν=0.25,
                  calc=nothing,
                  TOL=1e-4, zDir=1,
                  eos_correction = true)
   @assert species = "Si"

end

end
