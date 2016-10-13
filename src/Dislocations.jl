
module Dislocations

using JuLIP
using JuLIP.ASE

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
   F = JMat( [a/√2   0    0;
                 0   a    0;
                 0   0    a/√2 ] )
   X = [ JVec([0.0, 0, 0]),
         JVec([a/√8, a/2, a/√8]) ]
   # construct ASEAtoms
   at = ASEAtoms(string(s,"2"), X)
   set_defm!(at, F, updatepositions=false)
   # compute a burgers vector in these coordinates
   b = a/√2 * JVec([1.0,0.0,0.0])
   # compute a core-offset (to add to any lattice position)
   xcore = a/√2 * JVec([1/2, 1/3, 0])  # [1/2, 1/3, 0]
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
function fcc_edge_geom(s::AbstractString, R; truncate=true)
   # compute the correct unit cell
   atu, b, xcore, a = fcc_edge_plane(s)
   # multiply the cell to fit a ball of radius a/√2 * R inside
   L1 = ceil(Int, 2*R) + 3
   L2 = ceil(Int, 2*R/√2) + 3
   at = atu * (L1, L2, 1)
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
   # >>>>>> TODO: need the "real" one; this is just for testing <<<<<<<
   ux, uy = u_edge_isotropic(x, y, b, 0.25)
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



end
