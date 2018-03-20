
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
function dipole_tensor(at, athom;
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
   Flin = ( forces(V, set_positions!(athom, X0hom + h*Uhom)) -
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
