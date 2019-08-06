
using JuLIP, JuLIPMaterials

sym = :Cu
r0 = rnn(sym)
V = lennardjones(r0 = r0)

CC1 = []
for R in [3, 4, 6, 9, 13, 20]
   athom = cluster(sym, R*r0+2*cutoff(V))              # homogeneous reference configuration
   at = deleteat!(deepcopy(athom), 1)   # create a vacancy
   x̄ = athom[1]
   r = [norm(x-x̄) for x in positions(at)]
   fixedcell!(at)
   set_free!(at, findall(r .< R *r0))
   set_calculator!(at, V)

   C0, C1 = JuLIPMaterials.PoleExpansion.dipole_tensor(at, athom;
            method = :lbfgs, precond = FF(at, V), verbose=0, gtol = 1e-4)
   println("R = $R, C0 = $C0")
   println("        C1 = $(round.(C1,2))")
   push!(CC1, C1)
end

CC1

err = [ norm(C - CC1[end]) for C in CC1[1:end-1] ]
RR = [3, 4, 6, 9, 13] * r0

using Plots
Plots.gr()
P = Plots.plot(RR, err, lw=2, ms=12, m=:o, label = "err C1",
         xaxis = (:log, [0.8*RR[1],1.2*RR[end]]), yaxis = (:log, [1e-3, 4]))
Plots.plot!(RR, 10*RR.^(-2.5), lw=1, ls = :dash, ms = 0, label = "~ R^{-2.5}")
# Plots.plot!(RR, RR.^(-2.5), lw=1, linestyle = :dash, ms = 0, label = "~ R^{-2.5}")

;



# # TEST DECAY
#
# R = 10
# athom = cluster(sym, R*r0 + 2*cutoff(V))
# Xhom = positions(athom)
# x̄ = Xhom[1]
# at = deleteat!(deepcopy(athom), 1)
# @show length(at)
# X0 = positions(at)
# @assert norm(x̄ - mean(X0)) < 1e-10
# r = [ norm(x - x̄) for x in X0 ]
# set_constraint!( at, FixedCell(at, free = findall(r .< R*r0)) )
# set_calculator!( at, V )
# minimise!(at, method = :lbfgs, precond = FF(at, V), verbose=2, gtol = 1e-6)
# X = positions(at)
# U = X - X0
#
# e = strains(U, set_positions!(at, X0); rcut = 1.5 * r0)
#
# h = 1e-5
# F1 = (forces(set_positions!(at, X0 + h*U)) -
#       forces(set_positions!(at, X0 - h*U)) ) / (2*h)
# Iin = find(r .< 6*r0)
# sum(F1[Iin])
# sum( f * (x - x̄)'   for (f, x) in zip(F1[Iin], X0[Iin]) )
#
#
# Uhom = [ [zero(JVecF)]; U ]
# X0hom = positions(athom)
# F2 = (forces(V, set_positions!(athom, Xhom + h*Uhom)) -
#       forces(V, set_positions!(athom, Xhom - h*Uhom)) ) / (2*h)
# sum(F2[Iin])
# sum( f * (x - x̄)'   for (f, x) in zip(F2[Iin], X0hom[Iin]) )
#
#
#
# using Plots
# gr()
# t = [extrema(r)...]
# P = plot(r, 1e-15+e, lw = 0, m=:o, ms=2,
#          xaxis = (:log, [0.8,1.2].*t),
#          yaxis = (:log, [1e-5, 1e-2]))
# plot!(P, t, t.^(-3), lw=1)
#
#
#
#
#
#
# R = 10
# athom = cluster(sym, R*r0 + 2*cutoff(V))
# x̄ = at[1]
# at = deleteat!(deepcopy(at), 1)
# X0 = positions(at)
# set_calculator!(at, V)
# C0, C1 = JuLIPMaterials.PoleExpansion.dipole_tensor(at, athom)
# X = positions(at)
# maximum(norm.(X - X0))
# U = X - X0
# e = strains(U, set_positions!(at, X0); rcut = 1.5 * r0)
# r = [ norm(x - x̄) for x in X0 ]
# t = [extrema(r)...]
# P = plot(r, 1e-15+e, lw = 0, m=:o, ms=2,
#          xaxis = (:log, [0.8,1.2].*t),
#          yaxis = (:log, [1e-5, 1e-2]))
# plot!(P, t, t.^(-3), lw=1)
