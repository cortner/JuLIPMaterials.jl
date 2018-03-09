
module PoleExpansion

using JuLIP
using JuLIPMaterials: findin

function dipole_tensor(at::AbstractAtoms, athom::AbstractAtoms;
                       x̄ = mean(positions(at)),
                       dims = find(.!pbc(at)),
                       Rin = :ball)
   # SETUP
   # remember reference positions
   X0 = positions(at)
   # compute dirichlet boundary (assume spherical cluster)
   r = [norm(x - x̄) for x in X0]
   Rdom = maximum(r)
   rcut = cutoff(calculator(at))
   Ifree = find(r .< Rdom - 2*rcut)
   Iclamp = find(r .> Rdom - 2*rcut)
   set_constraint!(at, FixedCell(at, free = Ifree))
   # STEP 1: minimise
   minimise!(at)
   # STEP 2: project the solution over to athom
   X0hom = positions(athom)
   Ism, Ilge = findin(X0, X0hom)
   Xhom = copy(X0hom)
   for n = 1:length(Ilge)
      if Ilge[n] != 0
         Xhom[n] = X[Ilge[n]]
      end
   end
   # STEP 3: compute the linearised forces
   


   F = forces(at)
   if Rin == :ball  # assume the domain is a spherical cluster
      Rin = sqrt(maximum(r))
   elseif !(Rin isa Number)
      error("`dipole_tensor`: kwarg `Rin` must be a number or a recognized symbol")
   end
   Iin = find(r .< Rin)
   return sum( f ⊗ (x-x̄)  for (f, x) in zip(F[Iin], X[Iin]) )
end




end
