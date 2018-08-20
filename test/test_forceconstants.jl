
using Base.Test, JuLIP, JuLIPMaterials
using JuLIPMaterials: ForceConstantMatrix1

datapath = joinpath(Pkg.dir("JuLIP"), "data")
eam_Fe = EAM(datapath * "/pfe.plt",
             datapath * "/ffe.plt",
             datapath * "/F_fe.plt")

# equilibrate a unit cell
fe1 = bulk(:Fe)
set_calculator!(fe1, eam_Fe)
set_constraint!(fe1, VariableCell(fe1))
minimise!(fe1)

# get the force constants
fcm = ForceConstantMatrix1(eam_Fe, fe1, h = 1e-5)

# construct a larger computational cell and compute the hessian of eam_Fe on
# that cell
at = fe1 * 10
set_constraint!(at, FixedCell(at))
set_calculator!(at, eam_Fe)
H = JuLIP.hessian_pos(eam_Fe, at)
Hvec = JuLIP.hessian(at)
# random virtual displacement
U = rand(JVecF, length(at))
Uvec = mat(U)[:]
V = vecs(Hvec * Uvec)
# sanity check
@test V[1] ≈ sum( H[1,n] * U[n] for n = 1:length(U) )

# now multiply FCM * U
Vfcm = fcm * (at, U)
@show maximum(norm.(Vfcm - V)) / maximum(norm.(V))

# another test: do the block match?
X = positions(at)
x̄ = mean(X)
r = [norm(x-x̄) for x in X]
n0 = find( r .== minimum(r) )[1]
for n = 1:length(at)
   n == n0 && continue
   norm(H[n0,n]) < 1e-5 && continue
   R = X[n] - X[n0]
   found = false
   for m = 1:length(fcm.R)
      if norm(R - fcm.R[m]) < 1e-7
         found = true
         if norm(H[n0, m] - fcm.H[m]) < 1e-7
            println("+")
         else
            # print("-")
            println("error: ", norm(H[n0, m] - fcm.H[m]))
         end
         break
      end
   end
   if !found
      # print("x")
      println("not found: ", norm(H[n0, n]))
   end
end
