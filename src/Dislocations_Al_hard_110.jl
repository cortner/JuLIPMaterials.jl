
module Dislocations_Al_hard_110

using JuLIP
using JuLIP.ASE
using MaterialsScienceTools.Elasticity_110: elastic_moduli, voigt_moduli, fourth_order_basis, sextic_roots, A_coefficients, D_coefficients, little_a

const Afcc = JMatF([ 0.0 1 1; 1 0 1; 1 1 0])

"""
ensure that species S actually crystallises to FCC
"""
function check_fcc(S::AbstractString)
   F = defm(bulk(S))
   @assert vecnorm(F/F[1,2] - Afcc) < 1e-12
end

const Abcc = JMatF([ -1.0 1 1; 1 -1 1; 1 1 -1])

"""
ensure that species S actually crystallises to BCC
"""
function check_bcc(S::AbstractString)
   F = defm(bulk(S))
   return vecnorm(F/F[1,2] - Abcc) < 1e-12
end


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
   check_fcc(s)
   # get the cubic unit cell dimension
   a = ( bulk(s, cubic=true) |> defm )[1,1]
   print(" a : ")
   print(a)
   # construct the cell matrix
   F = a*JMat( [ sqrt(2)/2 0    0;
                 0   1     0;
                 0   0    sqrt(2)/2 ] )
   X = a*[ JVec([0.0, 0.0, 0.0]),
         JVec([(1/2)*1/sqrt(2),1/2, 1/(2*sqrt(2))]) ]
   # construct ASEAtoms
   at = ASEAtoms(string(s,"2"))
   set_defm!(at, F)
   set_positions!(at, X)
   # compute a burgers vector in these coordinates
   b =  a*sqrt(2)/2*JVec([1.0,0.0,0.0])
   # compute a core-offset (to add to any lattice position)
   xcore = a/√2 * JVec([1/4, -sqrt(2)/4, 0])#a*sqrt(2)/2 * JVec([1/2, -1/3, 0])  # [1/2, 1/3, 0]
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
                       TOL=1e-4,zDir=1)
   # compute the correct unit cell
   atu, b, xcore, a = fcc_edge_plane(s)
   # multiply the cell to fit a ball of radius a/√2 * R inside
   L1 = ceil(Int, 2*R) + 3
   L2 = ceil(Int, 2*R/√2) + 3
   at = atu * (L1,L2, zDir)
   atp = atu * (L1,L2, zDir)
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
   print("xc: ")
   print(xc)
   print(" yc: ")
   print(yc)
   r² = (x-xc).^2 + (y-yc).^2
   tip = minimum(r²)+.0000001 
   print(" minimum index : ")
   print(find( tip .> r² .> 0 ))  
   print(" minimum : ")
   print( minimum(r²) )
   
   I0 = find(  tip .> r² .> 0 )[2*zDir]
   print(" I0 : ")
   print(I0)
   print(" X12[I0] : ")
   print(X12[I0])
   xcore = X12[I0] + xcore
   print("xcore : ")
   print(xcore)
   x, y = x - xcore[1], y - xcore[2]
   # compute the dislocation predictor
   if cle == :isotropic
      ux, uy = u_edge_isotropic(x, y, b, ν)
   elseif cle == :anisotropic
      # TODO: this doesn't look clean; maybe we need to pass atu in the future
      # I'm not fully understanding how the function fcc_edge_plane(s) works
      set_pbc!(atu, true)
      atv = bulk("Al", cubic=true) * 4
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
   set_pbc!(at, (false, false, true))
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
   #q² = x.^2 + 2 * x .* y * lam * cos(ϕ) + y.^2 * lam^2
   #t² = x.^2 - 2 * x .* y * lam * cos(ϕ) + y.^2 * lam^2
   #ux2 = - (b / (4*π)) * (
   #       atan( (2*x.*y*lam*sin(ϕ)) ./ (x.^2 - lam^2*y.^2) )
   #       + (c̄11^2 - K[1,2]^2) / (2*c̄11*K[6,6]*sin(2*ϕ)) * (0.5 * log(q²./t²))
   #       )
   #uy2 = (lam*b)/(4*π*c̄11*sin(2*ϕ)) * (
   #      (c̄11 - K[1,2]) * cos(ϕ) * (0.5 * log(q².*t²))
   #       - (c̄11 + K[1,2]) * sin(ϕ) *
   #               atan( (y.^2*lam^2*sin(2*ϕ)) ./ (x.^2 - lam^2 * y.^2 * cos(2*ϕ)) )
   #    )



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


"""
`u_general(X, b, C::Array{Float64, 4}) -> U`

implementation of the CLE displacement field for a general
straight dislocation line in direction ξ with burgers vector b and
elastic moduli C.
"""
function u_general(X, b, ξ, C::Array{Float64, 4})
   
end

end
