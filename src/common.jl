
using JuLIP, StaticArrays

export dists, cluster, strains


const Vec3{T} = SArray{Tuple{3},T,1,3}
const Mat3{T} = SArray{Tuple{3,3},T,2,9}
const Ten33{T} = SArray{Tuple{3,3,3},T,3,27}
const Ten43{T} = SArray{NTuple{4,3},T,4,81}
const MVec3{T} = MVector{3, T}
const MMat3{T} = MMatrix{3,3,T}
const MTen33{T} = MArray{Tuple{3,3,3},T,3,27}
const MTen43{T} = MArray{NTuple{4,3},T,4,81}

const _EE = @SMatrix eye(3)
ee(i::Integer) = _EE[:,i]


"""
`strains(U, at; rcut = cutoff(calculator(at)))`

returns maximum strains :  `maximum( du/dr over all neighbours )` at each atom
"""
strains(U, at; rcut = cutoff(calculator(at))) =
   [ maximum(norm(u - U[i]) / s for (u, s) in zip(U[j], r))
                           for (i, j, r, _1, _2) in sites(at, rcut) ]
