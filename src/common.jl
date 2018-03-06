
using JuLIP, StaticArrays

export dists, cluster, strains


const Vec3{T} = SVector{3, T}
const Mat3{T} = SMatrix{3,3,T}
const Ten33{T} = SArray{Tuple{3,3,3},T,3,27}
const Ten43{T} = SArray{NTuple{4,3},T,4,81}
const MVec3{T} = MVector{3, T}
const MMat3{T} = MMatrix{3,3,T}
const MTen33{T} = MArray{Tuple{3,3,3},T,3,27}
const MTen43{T} = MArray{NTuple{4,3},T,4,81}


# this is all very confusing, probably better to just do things by hand
# on a case by case basis?
# dist(x, dims::AbstractVector) = vecnorm(x[dims])
#
# # little hack to allow indexing with tuples
# dist(x, dims::Tuple) = dist(x, SVector(dims))
#
# dist{T}(x::T, y::T, dims) = dist(x - y, dims)
#
# Base.broadcast{T}(dist, X::AbstractArray{T}, y::T, dims) =
#    [ dist(x, y, dims) for x in X ]
#
# # DEPRECATED: remove soon
# dists(varargs...) = error("`dists` has been replaced with `dist.`")


"""
`strains(U, at; rcut = cutoff(calculator(at)))`

returns maximum strains :  `maximum( du/dr over all neighbours )` at each atom
"""
strains(U, at; rcut = cutoff(calculator(at))) =
   [ maximum(norm(u - U[i]) / s for (u, s) in zip(U[j], r))
                           for (i, j, r, _1, _2) in sites(at, rcut) ]
