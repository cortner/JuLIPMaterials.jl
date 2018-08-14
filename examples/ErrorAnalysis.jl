module ErrorAnalysis

using JuLIP: AbstractAtoms, positions
using NearestNeighbors

# export findin, error_energynorm, relative_displacement, sobolev_norm,
#       sobolev_error

"""
`findin(Xsm, Xlge)`
Assuming that `Xsm ⊂ Xlge`, this function
returns `Ism, Ilge`, both `Vector{Int}` such that
* Xsm[i] == Xlge[Ism[i]]
* Xlge[i] == Xsm[Ilge[i]]  if Xlge[i] ∈ Xsm; otherwise Ilge[i] == 0
"""
function findin(Xsm, Xlge)
   # find the nearest neighbours of Xsm points in Xlge
   tree = KDTree(Xlge)
   Ism, Dsm = knn(tree, Xsm, 1)  # Ism is a vector of vectors (damn)
   Ism = [i[1] for i in Ism]
   Dsm = [d[1] for d in Dsm]
   @assert maximum(Dsm) < 0.01  # ensure we found the correct points
   # next construct Ilge
   Ilge = zeros(Int, length(Xlge))
   for n = 1:length(Ism)
      Ilge[Ism[n]] = n
   end
   return Ism, Ilge
end


"""
`relative_displacement(Xsm, Xrefsm, Xlge, Xreflge)`
Assumes that Xrefsm ⊂ Xreflge;
compute a relative displacement `U` such that
* length(U) == length(Xlge)
* Xsm[i] = Xlge[j] + U[j]  when  Xrefsm[i] == Xreflge[i]
* U[j] = 0 otherwise
"""
function relative_displacement(Xsm, Xrefsm, Xlge, Xreflge)
   @assert length(Xsm) == length(Xrefsm)
   @assert length(Xlge) == length(Xreflge)
   Ism, _ = findin(Xrefsm, Xreflge)
   err_ref = maximum(norm.(Xrefsm - Xreflge[Ism]))
   if err_ref > 1e-7
      warn("the reference configurations differ by $err_ref")
      # @show norm.(Xrefsm - Xreflge[Ism])
   end
   U = Xlge - Xreflge
   U[Ism] = Xsm - Xlge[Ism]  # = (Xsm - Xrefsm) - (Xlge - Xreflge)
   return U
end      # TODO: U should not be zero outside Xsm, but U = Xlge - Xlgeref



"""
`sobolev_norm(U, Xref, p, rnn::Float64)`
"""
function sobolev_norm(U, Xref, p, rnn::Float64)
   # specify the summation function
   if p == 2
      accum = (out, t) -> out + t*t
      final = out -> sqrt(out)
   elseif p == Inf
      accum = (out, t) -> max(out, abs(t))
      final = out -> out
   elseif 1 <= p < Inf
      accum = (out, t) -> out + t^p
      final = out -> out^(1/p)
   else
      error("p must be between 1 and ∞")
   end

   tree = KDTree(Xref)
   out = 0.0
   for n = 1:length(U)
      nhd = inrange(tree, Xref[n], 1.9 * rnn)
      norm_n = maximum( norm(u - U[n])/(norm(x - Xref[n])+0.01)
                        for (u, x) in zip(U[nhd], Xref[nhd]) )
      out = accum(out, norm_n)
   end
   return final(out)
end

function sobolev_error(atsm::AbstractAtoms, atlge::AbstractAtoms, p, rnn)
   Xsm = positions(atsm)
   Xlge = positions(atlge)
   U = relative_displacement(Xsm, Xsm, Xlge, Xlge)
   return sobolev_norm(U, Xlge, p, rnn)
end

error_energynorm(atsm::AbstractAtoms, atlge::AbstractAtoms, rnn) =
      sobolev_error(atsm, atlge, 2, rnn)

function sobolev_error(Xsm, Xrefsm, Xlge, Xreflge, p, rnn; relative=false)
   U = relative_displacement(Xsm, Xrefsm, Xlge, Xreflge)
   err = sobolev_norm(U, Xreflge, p, rnn)
   if relative
      err /= sobolev_norm(Xlge - Xreflge, Xreflge, p, rnn)
   end
   return err
end

error_energynorm(Xsm, Xrefsm, Xlge, Xreflge, rnn; relative=false) =
   sobolev_error(Xsm, Xrefsm, Xlge, Xreflge, 2, rnn; relative=relative)



end
