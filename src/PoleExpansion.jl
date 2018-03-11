
module PoleExpansion

using JuLIP, NearestNeighbors
import JuLIPMaterials
using JuLIPMaterials: findin2

⊗(x, y) = x * y'


"""
`findin(Xsm, Xlge)`

Assuming that `Xsm ⊂ Xlge`, this function
returns `Ism, Ilge`, both `Vector{Int}` such that
* Xsm[i] == Xlge[Ism[i]]
* Xlge[i] == Xsm[Ilge[i]]  if Xlge[i] ∈ Xsm; otherwise Ilge[i] == 0
"""
function findin3(Xsm, Xlge)
   # find the nearest neighbours of Xsm points in Xlge
   tree = KDTree(Xlge)
   # construct the Xsm -> Xlge mapping
   Ism = zeros(Int, length(Xsm))
   Ilge = zeros(Int, length(Xlge))
   for (n, x) in enumerate(Xsm)
      i = inrange(tree, Xsm[n], 1e-6)
      if isempty(i)
         Ism[n] = 0         # - Ism[i] == 0   if  Xsm[i] ∉ Xlge
      elseif length(i) > 1
         error("`inrange` found two neighbours")
      else
         Ism[n] = i[1]      # - Ism[i] == j   if  Xsm[i] == Xlge[j]
         Ilge[i[1]] = n     # - Ilge[j] == i  if  Xsm[i] == Xlge[j]
      end
   end
   # - if Xlge[j] ∉ Xsm then Ilge[j] == 0
   return Ism, Ilge
end



"""
* at : unrelaxed point defect configuration
* athom : homogeneous reference configuration

at the moment it is assumed both are spherical clusters
"""
function dipole_tensor(at::AbstractAtoms, athom::AbstractAtoms;
                       calc = calculator(at),
                       x̄ = mean(positions(athom)),
                       dims = find(.!pbc(at)),
                       Rin = :ball, Rdom = :ball, h = 1e-5,
                       verbose = 0,
                       precond = :exp)
   # TODO: account for dims
   # SETUP
   # remember reference positions
   X0 = positions(at)
   # compute dirichlet boundary (assume spherical cluster)
   r = [norm(x - x̄) for x in X0]
   if Rin == Rdom == :ball
      Rdom = maximum(r)
      Rin = 0.5 * Rdom
   elseif !(Rin isa Number) || !(Rdom isa Number)
      error("`Rin, Rdom` must be both :ball or both a number")
   end
   rcut = cutoff(calculator(at))
   Ifree = find(r .< Rdom - 2*rcut)
   Iclamp = find(r .> Rdom - 2*rcut)
   set_constraint!(at, FixedCell(at, free = Ifree))
   # STEP 1: minimise
   set_calculator!(at, calc)
   minimise!(at, method=:lbfgs, precond=precond, verbose=verbose)
   X = positions(at)
   # STEP 2: project the solution over to athom
   X0hom = positions(athom)
   @assert X0hom[2:end] == X0
   Xhom = copy(X0hom)
   Xhom[2:end] .= X
   # Ism, Ilge = findin(X0, X0hom)
   # @show find(Ilge .== 0)
   # @show find(Ism .== 0)
   # Xhom = copy(X0hom)
   # for n = 1:length(Ilge)
   #    if Ilge[n] != 0
   #       Xhom[n] = X[Ilge[n]]
   #    end
   # end
   # and compute the relative displacement
   Uhom = Xhom - X0hom
   U = X - X0
   # STEP 3: compute the linearised forces
   #    NB we could probably do this using the hessian or the dynamical matrix,
   #       but why not just do one finite-difference?
   Flin = ( forces(calc, set_positions!(athom, X0hom + h * Uhom)) -
            forces(calc, set_positions!(athom, X0hom - h * Uhom)) ) / (2*h)
   # STEP 4: compute and return the dipole tensor
   rhom = [ norm(x - x̄) for x in X0hom ]
   Iin = find(r .< Rin)
   # @show norm(Fhom0[Iin])
   return sum(Flin[Iin]), sum(f ⊗ (x-x̄)  for (f, x) in zip(Flin[Iin], X[Iin]))
end



function tensors2(sym, R, V;
                  verbose = 0)
   r0 = rnn(sym)
   athom = cluster(sym, R*r0 + 2*cutoff(V))
   Xhom = positions(athom)
   x̄ = Xhom[1]
   at = deleteat!(deepcopy(athom), 1)
   @show length(at)
   X0 = positions(at)
   @assert norm(x̄ - mean(X0)) < 1e-10
   r = [ norm(x - x̄) for x in X0 ]
   set_constraint!( at, FixedCell(at, free = find(r .< R*r0)) )
   set_calculator!( at, V )
   minimise!(at, method = :lbfgs, precond = FF(at, V),
             verbose=verbose, gtol = 1e-6)
   X = positions(at)
   U = X - X0

   h = 1e-5

   Uhom = [ [zero(JVecF)]; U ]
   X0hom = positions(athom)
   r = [ norm(x - x̄) for x in X0hom ]
   Iin = find(r .< 0.6 * R * r0)
   F2 = (forces(V, set_positions!(athom, Xhom + h*Uhom)) -
         forces(V, set_positions!(athom, Xhom - h*Uhom)) ) / (2*h)

   return  sum(F2[Iin]),
         sum( f * (x - x̄)'   for (f, x) in zip(F2[Iin], X0hom[Iin]) )
end

function tensors3(at, athom;
   Iin = :ball, Rin = :auto, x0 = mean(positions(athom)),
   h = 1e-5, kwargs...)

   # setup
   V = calculator(at)
   X0hom = positions(athom)
   X0 = positions(at)

   # relax the defect
   minimise!(at; kwargs...)
   X = positions(at)
   U = X - X0

   # compute the at -> athom mapping ...
   Ism, Ilge = findin3(X0, X0hom)
   # ... and the relative displacement
   Uhom = zeros(JVecF, length(athom))
   for n = 1:length(Ilge)
      if Ilge[n] != 0
         Uhom[n] = U[Ilge[n]]
      end
   end

   # compute the linearised forces in athom
   Flin = (forces(V, set_positions!(athom, X0hom + h*Uhom)) -
           forces(V, set_positions!(athom, X0hom - h*Uhom)) ) / (2*h)

   # extract the sub-domain on which to compute the tensors
   if Iin == :ball
      r = [ norm(x - x0) for x in X0hom ]
      if Rin == :auto
         Rin = 0.6 * (maximum(r) - 2*cutoff(V))
      end
      Iin = find(r .< Rin)
   end

   return  sum( Flin[Iin] ),
           sum( f * (x - x0)'   for (f, x) in zip(Flin[Iin], X0hom[Iin]) )
end


end
