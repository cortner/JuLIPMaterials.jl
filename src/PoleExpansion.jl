
module PoleExpansion

using JuLIP
using JuLIPMaterials: findin

⊗(x, y) = x * y'

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

   # F1 = (forces(set_positions!(at, X0 + h*U)) -
   #       forces(set_positions!(at, X0 - h*U)) ) / (2*h)
   # Iin = find(r .< 6*r0)
   # sum(F1[Iin])
   # sum( f * (x - x̄)'   for (f, x) in zip(F1[Iin], X0[Iin]) )

   Uhom = [ [zero(JVecF)]; U ]
   X0hom = positions(athom)
   r = [ norm(x - x̄) for x in X0hom ]
   Iin = find(r .< 0.6 * R * r0)
   F2 = (forces(V, set_positions!(athom, Xhom + h*Uhom)) -
         forces(V, set_positions!(athom, Xhom - h*Uhom)) ) / (2*h)

   return  sum(F2[Iin]),
         sum( f * (x - x̄)'   for (f, x) in zip(F2[Iin], X0hom[Iin]) )
end

function tensors3(sym, R, V;
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

   # F1 = (forces(set_positions!(at, X0 + h*U)) -
   #       forces(set_positions!(at, X0 - h*U)) ) / (2*h)
   # Iin = find(r .< 6*r0)
   # sum(F1[Iin])
   # sum( f * (x - x̄)'   for (f, x) in zip(F1[Iin], X0[Iin]) )

   Uhom = [ [zero(JVecF)]; U ]
   X0hom = positions(athom)
   r = [ norm(x - x̄) for x in X0hom ]
   Iin = find(r .< 0.6 * R * r0)
   F2 = (forces(V, set_positions!(athom, Xhom + h*Uhom)) -
         forces(V, set_positions!(athom, Xhom - h*Uhom)) ) / (2*h)

   return  sum(F2[Iin]),
         sum( f * (x - x̄)'   for (f, x) in zip(F2[Iin], X0hom[Iin]) )
end


end
