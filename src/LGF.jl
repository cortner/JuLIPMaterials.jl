
"""
Implements the lattice Green's function
"""
module LGFs

using JuLIP, NeighbourLists
using JuLIPMaterials.CLE
using JuLIPMaterials: ee, Vec3, Mat3, ForceConstantMatrix1
using JuLIPMaterials.CLE: GreenFunction3D

using JuLIP.Potentials: hess

import Base.length




"""
`struct LGF1` : lattice Green's function for a Bravais lattice
(single atom in the unit cell)

## Fields
* `R` : non-zero lattice vectors, in interaction range
* `H` : hessian blocks
"""
struct LGF1{T <: AbstractFloat, TI}
   DM::ForceConstantMatrix1{T}
   Gc::GreenFunction3D{T}
   at::Atoms{T, TI}      # only used to compute geometry information
end

length(G::LGF1) = length(G.R)



"""
`LGF` : construct a lattice Green's function

* `LGF(at; kwargs...)`
* `LGF(calc, at; kwargs...)`
"""
function LGF(calc::AbstractCalculator, at::Atoms; Nquad = 10, kwargs...)
   if length(at) > 1
      error("`LGF` : only the Bravais lattice case is currently implements")
   end
   return LGF1( ForceConstantMatrix1(calc, at),
                GreenFunction(calc, at; Nquad = 10, kwargs...),
                deepcopy(at) )
end


LGF(at::AbstractAtoms; kwargs...) = LGF(calculator(at), at; kwargs...)


function evaluate(G::LGF1, x)
   # what if x is not a lattice point - probably this is bad?

end


end
