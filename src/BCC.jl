
module BCC

using ..CLE
using JuLIP
using ..Vec3, ..Mat3, ..Ten43, ..cluster

function lattice_constants_111(sym::Symbol; calc = nothing)
   # See cell_111 for description of outputs
   if calc == nothing
      a0 = rnn(s)
   else
      at0 = bulk(sym, cubic=true, pbc=true)
      variablecell!(at0)
      set_calculator!(at0, calc)
      minimise!(at0, verbose=0)
      a0 = sqrt(3)/2 * cell(at0)[1,1]
   end
   b0 = sqrt(8/9) * a0   # b0^2 + (a0/3)^2 = a0^2
   c0 = sqrt(3/4) * b0   # (b0/2)^2 + c0^2 = b0^2
   return a0, b0, c0
end

function cell_111(sym::Symbol; calc=nothing)
   # a0 is z-direction cell height
   # b0 is x-spacing between atoms in cell
   # c0 is y-spacing between atoms in cell
   # Full cell is size = 3b0 × 2c0 × a0
   a0, b0, c0 = lattice_constants_111(sym, calc=calc)
   X = [ [0.0, 0.0, 0.0] [b0, 0.0, a0/3] [2*b0, 0.0, 2*a0/3] [b0/2, c0, 2*a0/3] [3*b0/2, c0, 0.0] [5*b0/2, c0, a0/3] ] |> vecs
   F = diagm([3*b0, 2*c0, a0])
   at = Atoms(sym, X)
   set_cell!(at, F')
   set_pbc!(at, true)
   return at
end

"""
`screw_111(sym::Symbol, R::Float64; x0=:center, layers=1)`

Construct a circular cluster with a screw dislocation in the centre.
* `sym`: chemical symbol
* `R`: radius of cluster
* `x0`: position of screw dislocation core, relative to a lattice site
"""
function screw_111(sym::Symbol, R::Float64;
            x0 = :center, layers=1, soln = :anisotropic, calc = nothing,
            bsign = 1)::AbstractAtoms
   a0, b0, c0 = lattice_constants_111(sym; calc=calc)
   x00 = JVecF( ([b0, 0, a0/3] + [b0/2, c0, 2*a0/3]) / 3 )
   if (x0 == :center) || (x0 == :centre)
      # center of mass of a triangle. (the z coordinate is irrelevant!)
      x0 = x00
   end
   # create a cluster
   at = cluster(cell_111(sym, calc=calc), R, dims=(1,2))
   at = at * (1,1,layers)
   # get positions  to manipulate them
   X = positions(at) |> mat
   # reference positions
   X0 = copy(X)
   X0[1,:] -= x0[1]
   X0[2,:] -= x0[2]
   set_info!(at, :X0, copy(X))
   # get coordinates for the dislocation predictor
   x, y = X[1,:] - x0[1], X[2,:] - x0[2]

   # get the screw displacement (Burgers vector = (0, 0, a0))
   if soln == :isotropic
      disl = IsoScrewDislocation3D( bsign*a0, remove_singularity = true )
      # apply to `at` and return
      X0 = vecs(X0)
      X = vecs(X)
      X .+= disl.(X0)
   elseif soln == :anisotropic
      atu = cell_111(sym)
      set_pbc!(atu, true)
      set_calculator!(atu, calc)
      t = Vec3([0.0, 0.0, 1.0])
      b = Vec3([0.0, 0.0, a0*bsign])
      m0 = Vec3([1.0, 0.0, 0.0])
      ℂ = CLE.elastic_moduli(atu)
      ℂ = Ten43{Float64}(ℂ[:])
      @show typeof(ℂ)
      disl = CLE.Dislocation(b, t, ℂ, Nquad = 20, cut = m0, remove_singularity = true)
      X0 = vecs(X0)
      X = vecs(X)
      X .+= disl.(X0)
   else
      error("unknown `soln`")
   end

   set_positions!(at, X)
   @show cell(at)
   return at
end





end
