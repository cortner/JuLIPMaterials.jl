
"""
`module Silicon`

Submodule of `MaterialsScienceTools` collecting functionality to set up
various types of defects in Silicon.
"""
module Silicon

using JuLIP
using JuLIP.ASE

import MaterialsScienceTools
DIS = MaterialsScienceTools.Dislocations

"""
`bulk110() -> at::ASEAtoms, b, xcore, a`

Generates a unit cell for a Si crystal which - if extended into the
x, y plane creates a 110 plane.

Returns
* `at`: Atoms object
* `b` : Burgers vector
* `xcore` : a core-offset (to add to any lattice position)
* `a` : cell dimension
"""
function bulk110()
   # get the cubic unit cell dimension
   a = ( bulk("Si", cubic=true) |> defm )[1,1]
   # construct the cell matrix and positions
   F = a * JMat( [ sqrt(2)/2 0    0;
                        0   1     0;
                        0   0    sqrt(2)/2 ] )
   X = a * [
      JVec([0.0, 0.0, 0.0]),
      JVec([(1/2)*1/sqrt(2),1/2, 1/(2*sqrt(2))]),
      JVec([0, -1/4, 1/(2*sqrt(2))]),JVec([(1/2)*1/sqrt(2), 1/4, 0])
   ]
   # construct ASEAtoms
   at = ASEAtoms("Si4")
   set_defm!(at, F)
   set_positions!(at, X)
   set_pbc!(at, (false, false, true))
   # compute a burgers vector in these coordinates
   b = a * sqrt(2)/2 * JVec([1.0, 0.0, 0.0])
   # compute a core-offset (to add to any lattice position)
   xcore = a * [-0.7, 1.0, 0.0]
   # return the information
   return at, b, xcore, a
end


"""
`edge_dipole_110() -> AbstractAtoms`

create an domain with side-lengths `L[1]`, `L[2]` cells-dimensions
with an ede dislocation dipole separated by `D` burgers vectors.

Keyword arguments:
* `D = 4` : dipole separation distance (in burgers vectors)
* `L = (3*D, 2*D)` : cell side-lengths (relative to a cubic unit cell)
"""
function edge_dipole_110(; D = 4, L = (3 * D, 2*D+1))
   @assert 1 < D <= 2 * L[1]
   @assert L[2] > 1
   # generate a single unit cell with 110 orientation
   atu, b, xcore, a = bulk110()
   # check that the Burgers vector is the same as the side-length of the cell
   F = defm(atu)
   @assert abs(b[1] - F[1,1]) < 1e-10
   # multiply the cell in x, y directions and set the correct b.c. (pbc in all directions)
   at = atu * (L[1], L[2], 1)
   set_pbc!(at, true)
   # compute the two core positions
   #   - first compute the centre of the domain (index of lower left atom in that cell)
   X = positions(at) |> mat; x, y = X[1,:], X[2,:]
   xc, yc = b[1] * floor(L[1]/2), F[2,2] * (floor(L[2]/2))
   # Ic = findmin( abs(x - xc) + abs(y - yc) )[2]
   @assert minimum(abs(x - xc) + abs(y - yc)) < 1e-10
   #   - compute cells where the cores will be
   yoffset = F[2,2] * 3/8
   xl = xc - b[1] * floor(D/2)
   yl = yc + yoffset
   xr = xc + b[1] * ceil(D/2)
   yr = yc + yoffset
   # Il = findmin( abs(x - xl) + abs(y - yc) )[2]
   @assert minimum(abs(x - xl) + abs(y - yc)) < 1e-10
   # Ir = findmin( abs(x - xr) + abs(y - yc) )[2]
   @assert minimum(abs(x - xr) + abs(y - yc)) < 1e-10

   # compute CLE displacement fields and apply to positions
   ν = 0.25
   Ul = DIS.u_edge_isotropic([x' - xl; y' - yl], b[1], ν)
   Ur = DIS.u_edge_isotropic([x' - xr; y' - yr], b[1], ν)
   Utot = Ul - Ur
   X[1:2,:] += Utot
   set_positions!(at, X)
   return at
end



end
