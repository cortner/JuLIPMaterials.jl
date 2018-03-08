
"""
Implements the lattice Green's function
"""
module LGFs

using JuLIP, NeighbourLists
using JuLIPMaterials.CLE
using JuLIPMaterials: ee, Vec3, Mat3
using JuLIPMaterials.CLE: GreenFunction3D

using JuLIP.Potentials: hess

import Base.length


struct DynamicalMatrix1{T <: AbstractFloat}
   R::Vector{Vec3{T}}
   H::Vector{Mat3{T}}
end

DynamicalMatrix1(calc::AbstractCalculator, at::AbstractAtoms) =
      DynamicalMatrix1(force_constants(calc, at)...)

function force_constants(calc::AbstractCalculator, at::Atoms{T}; h = 1e-5) where T <: AbstractFloat
   @assert Base.length(at) == 1
   cl = cluster(bulk(chemical_symbol(at.Z[1]), cubic=true), cutoff(calc) + 1)
   set_calculator!(cl, calc)
   x = cl[1]  # x ≡ X[1]
   ∂f_∂xi = []
   for i = 1:3
      cl[1] = x + h * ee(i)
      fip = forces(cl)
      cl[1] = x - h * ee(i)
      fim = forces(cl)
      cl[1] = x
      push!(∂f_∂xi, (fip - fim) / (2*h))
   end
   # convert to dynamical matrix entries
   H = [ - [∂f_∂xi[1][n] ∂f_∂xi[2][n] ∂f_∂xi[3][n]]'  for n = 1:length(cl) ]
   @show typeof(H)
   # extract the non-zero entries
   Inz = setdiff(find( norm.(H) .> 1e-8 ), [1])
   # . . . and return
   R = [y - x for y in positions(cl)[Inz]]
   return R, H[Inz]
end

force_constants{T}(calc::PairPotential, at::Atoms{T}) =
      force_constants(SitePotential(calc), at)


"""
`struct LGF1` : lattice Green's function for a Bravais lattice
(single atom in the unit cell)

## Fields
* `R` : non-zero lattice vectors, in interaction range
* `H` : hessian blocks
"""
struct LGF1{T <: AbstractFloat}
   DM::DynamicalMatrix1{T}
   Gc::GreenFunction3D{T}
end

length(G::LGF1) = length(G.R)



"""
`LGF` : construct a lattice Green's function

* `LGF(at; kwargs...)`
* `LGF(calc, at; kwargs...)`
"""
function LGF(calc::AbstractCalculator, at::Atoms; Nquad = 10, kwargs...)
   if Base.length(at) > 1
      error("`LGF` : only the Bravais lattice case is currently implements")
   end
   return LGF1( DynamicalMatrix1(calc, at),
                GreenFunction(calc, at; Nquad = 10, kwargs...) )
end


LGF(at::AbstractAtoms; kwargs...) = LGF(calculator(at), at; kwargs...)


function evaluate(G::LGF1, x)
   # what if x is not a lattice point - probably this is bad?

end


end
