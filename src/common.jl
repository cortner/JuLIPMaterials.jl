
using JuLIP

dists{T}(X::AbstractArray{T}, y::T, dims) = [vecnorm(x[dims] - y[dims]) for x in X]

dists{T}(X::AbstractArray{T}, y::T, dims::Tuple) = dists(X, y, [dims...])


"""
`cluster(species::AbstractString, R::Real; dims=(1,2,3)) -> at::AbstractAtoms`

Produce a circular / spherical cluster of approximately radius R. The center atom is
always at index 1 and position 0

* `species`: atom type
* `R` : radius
* `dims` : dimension into which the cluster is extended, typically
   `(1,2,3)` for 3D point defects and `(1,2)` for 2D dislocations, in the
   remaining dimension(s) the b.c. will be periodic.
"""
function cluster(species::AbstractString, R::Real; dims=(1,2,3))::AbstractAtoms
   # create a cubic cell
   atu = JuLIP.ASE.bulk(species, cubic=true, pbc = false)
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
   # and reflect to make it symmetric about the origin
   F = Matrix(defm(at))
   X = positions(at)
   for j in dims
      J = eye(3); J[j, j] = -1; J = JMatF(J)
      Xreflect = [J * x for x in X]
      X = unique([X; Xreflect])
      F[j, j] *= 2
   end
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
