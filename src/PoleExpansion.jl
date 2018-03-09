
module PoleExpansion


function dipole_tensor(at::AbstractAtoms;
                       x̄ = mean(positions(at)),
                       dims = find(.!pbc(at)),
                       Rin = :ball)
   X = positions(at)
   r = [norm(x - x̄) for x in X]
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
