
using Base.Test
using MaterialsScienceTools
using JuLIP, JuLIP.Potentials, ForwardDiff

MST = MaterialsScienceTools
CB = MST.CauchyBorn
CLE = MST.CLE
FD = ForwardDiff

println("------------------------------------------------------------")
println(" Testing Simple Lattice Cauchy--Born Implementation")
println("------------------------------------------------------------")

at = bulk("Fe")
r0 = rnn("Fe")
calc = LennardJones(σ = r0) * C2Shift(2.7*r0)
set_calculator!(at, calc)
set_constraint!(at, VariableCell(at))
# minimise!(at)
println("Constructing Wcb . . .")
W = CB.Wcb(at, calc)

println("generate a cauchy Born potential . . . ")
println("check W, grad(W..) evaluate ok . . .")
@test W(eye(3)) == energy(at) / det(defm(at))
@test CB.grad(W, eye(3)) isa AbstractMatrix

println("Finite-difference consistency test")
F = eye(3) + (rand(3,3) - 0.5) * 0.01
W0 = W(F)
dW0 = CB.grad(W, F)[:]
errors = []
for p = 2:10
   h = .1^p
   dWh = zeros(9)
   for i = 1:9
      F[i] += h
      dWh[i] = (W(F) - W0) / h
      F[i] -= h
   end
   push!(errors, norm(dWh-dW0, Inf))
   @printf(" %2d  |  %.3e \n", p, errors[end])
end
passed = minimum(errors) <= 1e-3 * maximum(errors)
if passed
   println("passed")
else
   warn("""It seems the finite-difference test has failed, which indicates
   that there is an inconsistency between the function and gradient
   evaluation. Please double-check this manually / visually. (It is
   also possible that the function being tested is poorly scaled.)""")
end

# ------ The big test: 2nd-order consistency with atomistic model -------


using Base.Test
using MaterialsScienceTools
using JuLIP, JuLIP.Potentials, ForwardDiff

MST = MaterialsScienceTools
CB = MST.CauchyBorn
CLE = MST.CLE
FD = ForwardDiff

# setup a Cauchy-Born model for bulk Fe
atu = bulk("Fe")
r0 = rnn("Fe")
calc = LennardJones(σ = r0) * C2Shift(2.7*r0)
set_calculator!(atu, calc)
set_constraint!(atu, VariableCell(atu))
minimise!(atu)
println("Constructing Wcb . . .")
W = CB.Wcb(atu, calc, normalise = :atoms)

energy(atu)

virial(atu)  |> norm
CB.grad(W, eye(3))

# a nice, smooth displacement
R = 30.1
p = -1
y = x -> x + (2 + sum(x.^2))^((p-1)/2) * x
∇y = x -> FD.jacobian(y, x)

at = MST.cluster("Fe", R)
set_pbc!(at, true)
set_calculator!(at, calc)
X0 = positions(at)
set_positions!(at, y.(X0))

r = 1 + norm.(X0)
Fat = forces(at)

Fcb = [ CB.div_grad(W, ∇y, x)  for x in X0 ]



using Plots
Plots.gr()
P = plot(r, 1e-15+norm.(Fat), lw=0, m=:o, ms=2, label = "|f_at|",
         xaxis = (:log, [1.0, 1.2*R]), yaxis = (:log, [1e-3, 1e2]) )
plot!(P, r, norm.(Fcb), lw = 0, m=:o, ms=2, label="|f_cb|")
plot!(P, r, 5_000*r.^(p-2), label = "r^{p-2}")
display(P)
