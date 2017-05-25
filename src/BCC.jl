
module BCC

using MaterialsScienceTools: cluster
using JuLIP, JuLIP.ASE

function lattice_constants_111(s::AbstractString)
   a0 = rnn(s)
   b0 = sqrt(8/9) * a0   # b0^2 + (a0/3)^2 = a0^2
   c0 = sqrt(3/4) * b0   # (b0/2)^2 + c0^2 = b0^2
   return a0, b0, c0
end


function cell_111(s::AbstractString)
   a0, b0, c0 = lattice_constants_111(s)
   X = [ [0.0, 0.0, 0.0] [b0, 0.0, a0/3] [2*b0, 0.0, 2*a0/3] [b0/2, c0, 2*a0/3] [3*b0/2, c0, 0.0] [5*b0/2, c0, a0/3] ] |> vecs
   F = diagm([3*b0, 2*c0, a0])
   at = ASEAtoms("$s$(length(X))")
   set_positions!(at, X)
   set_defm!(at, F)
   set_pbc!(at, true)
   return at
end


u_screw(x, y, b) = (b / (2*pi)) * angle(x + im * y)


"""
`screw_111(s::AbstractString, R::Float64; x0=:center, layers=1)`

Construct a circular cluster with a screw dislocation in the centre.
* `s`: species
* `R`: radius of cluster
* `x0`: position of screw dislocation core, relative to a lattice site
"""
function screw_111(s::AbstractString, R::Float64; x0 = :center, layers=1)::AbstractAtoms
   a0, b0, c0 = lattice_constants_111(s)
   x00 = JVecF( ([b0, 0, a0/3] + [b0/2, c0, 2*a0/3]) / 3 )
   if (x0 == :center) || (x0 == :centre)
      # center of mass of a triangle. (the z coordinate is irrelevant!)
      x0 = x00
   end
   # create a cluster
   at = cluster(cell_111(s), R, dims=(1,2))
   at = at * (1,1,layers)
   # get positions  to manipulate them
   X = positions(at) |> mat
   # reference positions
   X0 = copy(X)
   X0[1,:] -= x00[1]
   X0[2,:] -= x00[2]
   set_info!(at, :X0, copy(X))
   # get coordinates for the dislocation predictor
   x, y = X[1,:] - x0[1], X[2,:] - x0[2]
   # get the screw displacement (Burgers vector = (0, 0, a0))
   u = u_screw(x, y, a0)
   # apply to `at` and return
   # X[1, :] = X0[1,:]
   # X[2, :] = X0[2,:]
   X[3, :] += u
   set_positions!(at, X)
   return at
end

end
