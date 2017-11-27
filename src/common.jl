
using JuLIP, StaticArrays

export dists, cluster, strains


const Vec3{T} = SVector{3, T}
const Mat3{T} = SMatrix{3,3,T}
const Ten33{T} = SArray{Tuple{3,3,3},T,3,27}
const Ten43{T} = SArray{Tuple{3,3,3,3},T,4,81}


dist(x, dims) = vecnorm(x[dims])

dist{T}(x::T, y::T, dims) = dist(x - y, dims)

Base.broadcast{T}(dist, X::AbstractArray{T}, y::T, dims) =
   [ dist(x, y, dims) for x in X ]

# DEPRECATED: remove soon
dists(varargs...) = error("`dists` has been replaced with `dist.`")


"""
`cluster(args...; kwargs...) -> at::AbstractAtoms`

Produce a circular / spherical cluster of approximately radius R. The center
atom is always at index 1 and position 0

## Methods
```
cluster(species::AbstractString, R::Real; dims=[1,2,3])
cluster(atu::AbstractAtoms, R::Real; dims = [1,2,3])
```
The second method assumes that there is only a single species.

## Parameters
* `species`: atom type
* `R` : radius
* `dims` : dimension into which the cluster is extended, typically
   `(1,2,3)` for 3D point defects and `(1,2)` for 2D dislocations, in the
   remaining dimension(s) the b.c. will be periodic.
* `atu` : unitcell

## TODO
 * lift the restriction of single species
 * allow other shapes
"""
function cluster(atu::AbstractAtoms, R::Real; dims = (1,2,3))
   species = JuLIP.ASE.chemical_symbols(atu)[1]
   # check that the cell is orthorombic
   Fu = defm(atu)
   X = positions(atu)
   @assert isdiag(Fu)
   @assert norm(X[1]) == 0.0   # check that the first index is the centre
   # determine by how much to multiply in each direction
   L = [ j ∈ dims ? 2 * (ceil(Int, R/Fu[j,j])+3) : 1    for j = 1:3]
   # multiply
   at = atu * L
   # and shift + swap positions
   X = [x - (Fu * floor.(L/2)) for x in positions(at)]
   i0 = find(norm.(X) .< 1e-10)[1]
   X[1], X[i0] = X[i0], X[1]
   F = diagm([Fu[j,j]*L[j] for j = 1:3])
   # carve out a cluster with mini-buffer to account for round-off
   r = dist.(X, X[1], dims)
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

returns maximum strains :  `maximum( du/dr over all neighbours )` at each atom
"""
strains(U, at; rcut = cutoff(calculator(at))) =
   [ maximum(norm(u - U[i]) / s for (u, s) in zip(U[j], r))
                           for (i, j, r, _1, _2) in sites(at, rcut) ]
