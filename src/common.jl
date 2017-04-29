
using JuLIP

export dists, cluster, strains

dists{T}(X::AbstractArray{T}, dims) = [vecnorm(x[dims]) for x in X]

dists{T}(X::AbstractArray{T}, dims::Tuple) = dists(X, [dims...])

dists{T}(X::AbstractArray{T}, y::T, dims) = [vecnorm(x[dims] - y[dims]) for x in X]

dists{T}(X::AbstractArray{T}, y::T, dims::Tuple) = dists(X, y, [dims...])

"""
`cluster(args...; kwargs...) -> at::AbstractAtoms`

Produce a circular / spherical cluster of approximately radius R. The center atom is
always at index 1 and position 0

## Methods
```
cluster(species::AbstractString, R::Real; dims=(1,2,3))
cluster(atu::AbstractAtoms, R::Real; dims = (1,2,3))
```
The second method assumes that there is only a single species.

## Parameters
* `species`: atom type
* `R` : radius
* `dims` : dimension into which the cluster is extended, typically
   `(1,2,3)` for 3D point defects and `(1,2)` for 2D dislocations, in the
   remaining dimension(s) the b.c. will be periodic.

## TODO
 * lift the restriction of single species
 * allow other shapes
"""
function cluster(atu::AbstractAtoms, R::Real; dims = (1,2,3))::AbstractAtoms
   species = JuLIP.ASE.chemical_symbols(atu)[1]
   # check that the cell is orthorombic
   Fu = defm(atu)
   X = positions(atu)
   @assert isdiag(Fu)
   @assert norm(X[1]) == 0.0   # check that the first index is the centre
   # determine by how much to multiply in each direction
   L = [1, 1, 1]
   for j in dims
      L[j] = ceil(Int, R/Fu[j,j])+3
   end
   # multiply
   at = atu * tuple(L...)
   # and make copies to make it symmetric about the origin
   X = positions(at)
   E = eye(3) |> vecs
   for j in dims
      J = eye(3); J[j, j] = -1; J = JMatF(J)
      Xcopy = [x - L[j] * Fu[j,j] * E[j] for x in X]
      X = [X; Xcopy]   # unique([X; Xreflect])
   end
   F = 2 * Fu
   @assert norm(X[1]) == 0.0   # double-check that the centre is still at 0
   # carve out a cluster with mini-buffer to account for round-off
   r = dists(X, X[1], dims)
   IR = find( r .<= R+sqrt(eps()) )
   # generate new positions
   Xcluster = X[IR]
   @assert norm(Xcluster[1]) == 0.0
   # generate the cluster
   at_cluster = ASEAtoms("$(species)$(length(Xcluster))")
   set_defm!(at_cluster, F)
   set_positions!(at_cluster, Xcluster)
   # take open boundary in all directions specified in dims, periodic in the rest
   pbcs = [ !(j ∈ dims) for j = 1:3 ]
   set_pbc!(at_cluster, tuple(pbcs...))
   # return the cluster and the index of the center atom
   return at_cluster
end


function cluster(species::AbstractString, R::Real; kwargs...)::AbstractAtoms
   # create a cubic cell
   atu = JuLIP.ASE.bulk(species, cubic=true, pbc = false)
   return cluster(atu, R; kwargs...)
end

"""
`strains(U, at; rcut = cutoff(calculator(at)))`

maximum strains maximum( du/dr over all neighbours ) at each atom.
"""
strains(U, at; rcut = cutoff(calculator(at))) =
   [ maximum(norm(u - U[i]) / s for (u, s) in zip(U[j], r))
                           for (i, j, r, _1, _2) in sites(at, rcut) ]
