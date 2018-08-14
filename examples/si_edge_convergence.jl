using JuLIPMaterials, JuLIP
include("ErrorAnalysis.jl")
const ERR = ErrorAnalysis

const Si = JuLIPMaterials.Si
const sw = Si.sw_eq()
rbuf = 2 * cutoff(sw)

SS = [3, 5, 8, 12]
S_ex = 24
RR = SS * 1.1 * rnn(:Si)
R_ex = S_ex * 1.1 * rnn(:Si)

si_cell_mult(R) =
   (ceil(Int, 3 * R / cell(bulk(:Si, cubic=true))[1,1]) ÷ 2) * 2 + 1

function ref_cluster(R; kwargs...)
      L = si_cell_mult(R+rbuf)
      at, xc = Si.edge110(:Si, L; calc = sw, kwargs...)
      set_data!(at, :xc, xc)
      return at
end

function solve_edge(R, at_ref; verbose=false)
   X0 = positions(at_ref)
   xc = get_data(at_ref, :xc)
   r = [norm(x-xc) for x in X0]
   IR = find(r .<= R+rbuf)
   at = Atoms(:Si, X0[IR])
   set_cell!(at, cell(at_ref))
   set_constraint!(at, FixedCell(at, clamp = r[IR] .> R))
   set_calculator!(at, sw)
   minimise!(at; verbose=verbose)
   set_data!(at, :X0, X0[IR])
   set_data!(at, :xc, xc)
   return at
end

Base.error(at::Atoms, at_ex::Atoms) =
      ERR.error_energynorm(positions(at),    get_data(at,    :X0),
                           positions(at_ex), get_data(at_ex, :X0), rnn(:Si))

at_ref = ref_cluster(R_ex; sym = true)
at_nref = ref_cluster(R_ex; sym = false)

at_ex = solve_edge(R_ex, at_ref)
at_nex = solve_edge(R_ex, at_nref)

@show length(at_ref), length(at_ex)

err_nosym = Float64[]
err_sym = Float64[]

for R in RR
   # sym = true
   at = solve_edge(R, at_ref)
   push!(err_sym, error(at, at_ex))

   # sym = false
   at = solve_edge(R, at_nref)
   push!(err_nosym, error(at, at_nex))
end

err_sym



# at = solve_edge(RR[1], at_ref)
# X0 = get_data(at, :X0)
#
# x, y = mat(X0)[1,:], mat(X0)[2,:]
# using Plots
# plot(x, y, lw=0, m=:o)

# r_nosym = [norm(x - xc) for x in positions(at_nosym)]
# F = forces(at_nosym)
# f_nosym = norm.(F)
# J0, J1, Jdel = Si.si_multilattice(at_nosym; TOL=0.5)
# f_avg = norm.(F[J0] .+ F[J1])
# r_avg = r_nosym[J0]
#
# at_sym, xc = Si.edge110(:Si, R; calc = sw, sym=true)
# r_sym = [norm(x - xc) for x in positions(at_sym)]
# f_sym = norm.(forces(at_sym))
