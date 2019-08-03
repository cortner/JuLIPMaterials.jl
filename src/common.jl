
using JuLIP, StaticArrays, NearestNeighbors

import Base: findin, *

export strains

const Vec2{T} = SArray{Tuple{2},T,1,2}
const Mat2{T} = SArray{Tuple{2,2},T,2,4}
const Ten32{T} = SArray{Tuple{2,2,2},T,3,8}
const Ten42{T} = SArray{NTuple{4,2},T,4,16}
const MVec2{T} = MVector{2,T}
const MMat2{T} = MMatrix{2,2,T}
const MTen32{T} = MArray{Tuple{2,2,2},T,3,8}
const MTen42{T} = MArray{NTuple{4,2},T,4,16}

const Vec3{T} = SArray{Tuple{3},T,1,3}
const Mat3{T} = SArray{Tuple{3,3},T,2,9}
const Ten33{T} = SArray{Tuple{3,3,3},T,3,27}
const Ten43{T} = SArray{NTuple{4,3},T,4,81}
const MVec3{T} = MVector{3, T}
const MMat3{T} = MMatrix{3,3,T}
const MTen33{T} = MArray{Tuple{3,3,3},T,3,27}
const MTen43{T} = MArray{NTuple{4,3},T,4,81}

const _EE = Mat3{Float64}([1 0 0; 0 1 0; 0 0 1])
ee(i::Integer) = _EE[:,i]


"""
`strains(U, at; rcut = cutoff(calculator(at)))`

returns maximum strains :  `maximum( du/dr over all neighbours )` at each atom
"""
strains(U, at; rcut = cutoff(calculator(at))) =
   [ maximum(norm(u - U[i]) / s for (u, s) in zip(U[j], r))
                           for (i, j, r, _1) in sites(at, rcut) ]



struct ForceConstantMatrix1{T <: AbstractFloat}
   R::Vector{Vec3{T}}
   H::Vector{Mat3{T}}
end

function *(fcm::ForceConstantMatrix1{T}, t::Tuple{<:Atoms, Vector{<: Vec3}}) where {T}
   at, U = t
   V = zeros(Vec3{T}, length(at))
   rcut = maximum(norm.(fcm.R)) * 1.01
   # WARNING: this is an extremely naive implementation which needs to be fixed
   #          as soon as possible
   for (i, j, _, R) in sites(at, rcut)
      for n = 1:length(R)
         for m = 1:length(fcm.R)
            if norm(R[n] - fcm.R[m]) < 1e-7
               V[i] += fcm.H[m] * (U[j[n]] - U[i])
               break
            end
         end
      end
   end
   return V
end


ForceConstantMatrix1(calc::AbstractCalculator, at::AbstractAtoms; kwargs...) =
      ForceConstantMatrix1(force_constants(calc, at; kwargs...)...)

function force_constants(calc::AbstractCalculator, at::Atoms{T};
                         h = 1e-5, rcut = 2.01*cutoff(calc) + 1) where T <: AbstractFloat
   @assert length(at) == 1 # assume no basis
   cl = cluster(at, rcut)
   set_calculator!(cl, calc)
   x = cl[1]  # x ≡ X[1]
   ∂f_∂xi = []
   for i = 1:3
      cl[1] = x + h * ee(i)
      fip = forces(cl)
      cl[1] = x - h * ee(i)
      fim = forces(cl)
      cl[1] = x
      push!(∂f_∂xi, (fip - fim) / (2*h))
   end
   # convert to dynamical matrix entries
   H = [ - [∂f_∂xi[1][n] ∂f_∂xi[2][n] ∂f_∂xi[3][n]]'  for n = 1:length(cl) ]
   # extract the non-zero entries
   Inz = setdiff(find( norm.(H) .> 1e-8 ), [1])
   # . . . and return
   R = [y - x for y in positions(cl)[Inz]]
   return R, H[Inz]
end

force_constants(calc::PairPotential, at::Atoms{T}) where {T} =
      force_constants(SitePotential(calc), at)


"""
`findin(Xsm, Xlge)`

Assuming that `Xsm ⊂ Xlge`, this function
returns `Ism, Ilge`, both `Vector{Int}` such that
* Xsm[i] == Xlge[Ism[i]]
* Xlge[i] == Xsm[Ilge[i]]  if Xlge[i] ∈ Xsm; otherwise Ilge[i] == 0
"""
function findin2(Xsm::Vector{SVec{T}}, Xlge::Vector{SVec{T}}) where T <: AbstractFloat
   # find the nearest neighbours of Xsm points in Xlge
   tree = KDTree(Xlge)
   # construct the Xsm -> Xlge mapping
   Ism = zeros(Int, length(Xsm))
   Ilge = zeros(Int, length(Xlge))
   for (n, x) in enumerate(Xsm)
      i = inrange(tree, Xsm, sqrt(eps(T)))
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


# """
# `relative_displacement(Xsm, Xrefsm, Xlge, Xreflge)`
#
# Assumes that Xrefsm ⊂ Xreflge;
# compute a relative displacement `U` such that
# * length(U) == length(Xlge)
# * Xsm[i] = Xlge[j] + U[j]  when  Xrefsm[i] == Xreflge[i]
# * U[j] = 0 otherwise
# """
# function relative_displacement(Xsm, Xrefsm, Xlge, Xreflge)
#    @assert length(Xsm) == length(Xrefsm)
#    @assert length(Xlge) == length(Xreflge)
#    Ism, _ = findin(Xrefsm, Xreflge)
#    @assert maximum(norm.(Xrefsm - Xreflge[Ism])) < 1e-7
#    U = Xlge - Xreflge
#    U[Ism] = Xsm - Xlge[Ism]  # = (Xsm - Xrefsm) - (Xlge - Xreflge)
#    return U
# end      # TODO: U should not be zero outside Xsm, but U = Xlge - Xlgeref
