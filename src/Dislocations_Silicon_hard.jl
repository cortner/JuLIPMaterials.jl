
module Dislocations_Silicon_hard

using JuLIP
using JuLIP.ASE
using MaterialsScienceTools.Elasticity: elastic_moduli, voigt_moduli, fourth_order_basis, sextic_roots, A_coefficients, D_coefficients

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
   # construct the cell matrix
   F = a*JMat( [ sqrt(2)/2 0    0;
                 0   2*sqrt(3)/3     0;
                 0   0    1 ] )
   X = a*[ JVec([0.0, 0.0, 0.0]),
         JVec([sqrt(2)/4, -sqrt(3)/3, 1/(sqrt(6)*2)]), JVec([0, sqrt(3)/4, 0]),JVec([sqrt(2)/4,-sqrt(3)/3+sqrt(3)/4, 1/(sqrt(6)*2)]) ]
   # construct ASEAtoms
   at = ASEAtoms(string(s,"4"))
   set_defm!(at, F)
   set_positions!(at, X)
   # compute a burgers vector in these coordinates
   b =  a*sqrt(2)/2*JVec([1.0,0.0,0.0])
   # compute a core-offset (to add to any lattice position)
   xcore = [.5, -.5, 0]#a*sqrt(2)/2 * JVec([1/2, 1/3, 0])  # [1/2, 1/3, 0]
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
      Cv = voigt_moduli(calc, atu)
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
   uy = -b/(2*π) * ( (1-2*ν)/(4*(1-ν)) * log(r²) + (y.^2 - x.^2) ./ (4*(1-ν) * r²) )
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
   Cv = zeros(6,6)
   
   #Hard code the elasticity tensor for now
   Cv[1,1] = 16.57
   Cv[1,2] = 6.39
   Cv[6,6] = 7.96
   # >>>>>>>>> START DEBUG >>>>>>>>
   Cv[2,2] = Cv[3,3] = Cv[1,1]
   Cv[1,3] = Cv[2,3] = Cv[2,1] = Cv[3,1] = Cv[3,2] = Cv[1,2]
   Cv[4,4] = Cv[5,5] = Cv[6,6]
   # <<<<<<<<< END DEBUG <<<<<<<<<


   #Now transform Cv into the correct coordinate basis 
   C = fourth_order_basis(Cv,a)
   p = sextic_roots(C)
   
   #Should test this against solving the full linear system
   A = A_coefficients(p,C)
   #Set up for burgers vector in x1 direction only
   D = D_coefficients(p,C,A,b)
   
   ux = real( im/(2*π)*(A[1,1]*(D[1] + D[2]*im)*log(x+p[1]*y) + A[1,2]*(D[3] + D[4]*im)*log(x+p[2]*y) + A[1,3]*(D[5] + D[6]*im)*log(x+p[3]*y)   ))

   uy = real( im/(2*π)*(A[2,1]*(D[1] + D[2]*im)*log(x+p[1]*y) + A[2,2]*(D[3] + D[4]*im)*log(x+p[2]*y) + A[2,3]*(D[5] + D[6]*im)*log(x+p[3]*y)   ))


   #Now we can compute the displacements using Hirth and Lothe 13-91
   #ux = - (b / (4*π)) * (
   #       atan( (2*x.*y*λ*sin(ϕ)) ./ (x.^2 - λ^2*y.^2) )
   #       + (c̄11^2 - C[1,2]^2) / (2*c̄11*C[6,6]*sin(2*ϕ)) * (0.5 * log(q²./t²))
   #       )
   #uy = (λ*b/(4*π*c̄11*sin(2*ϕ))) * (
   #      (c̄11 - C[1,2]) * cos(ϕ) * (0.5 * log(q².*t²))
   #       - (c̄11 + C[1,2]) * sin(ϕ) *
   #                atan( (y.^2*λ^2*sin(2*ϕ)) ./ (x.^2 - λ^2 * y.^2 * cos(2*ϕ)) )
   #    )

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
