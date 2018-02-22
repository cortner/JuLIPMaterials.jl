

using JuLIP, JuLIP.Potentials
import MaterialsScienceTools

MST = MaterialsScienceTools
CLE = MST.CLE

using CLE: grad
using MST.Testing
using Einsum

println("---------------------------------------------------------------")
println(" Testing the 3D Anisotropic Dislocation Solution Implementation")
println("---------------------------------------------------------------")

# TODO:
# Test PDE is satisfied
# Test gradient matches
# Test singularity normalisation by integrating outward derivative around cylinder

# Test explicit formulae vs anistropic ones
print("Test agreement between anisotopic and isotropic edge dislocation: ")
λ = 1.0 + rand()
μ = 1.0 + rand()
Ciso = CLE.isotropic_moduli(λ, μ)
b = [1.0,0.0,0.0]
t = [0.0,0.0,1.0]
u0 = CLE.Dislocation(b, t, Ciso, Nquad = 4)
uedge = CLE.IsoEdgeDislocation3D(λ, μ, b[1])
maxerr = 0.0
maxerr_g = 0.0
for n = 1:10
   x = randvec3()
   maxerr = max( maxerr, vecnorm(u0(x) - uedge(x), Inf) )
   maxerr_g = max(maxerr_g, vecnorm(grad(u0, x) - grad(uedge, x), Inf) )
end
println("maxerr = $maxerr, maxerr_g = $maxerr_g")
@test maxerr < 1e-12
@test maxerr_g < 1e-12

λ = 1.0 + rand()
μ = 1.0 + rand()

print("Test agreement between anisotopic and isotropic edge dislocation: ")
Ciso = CLE.isotropic_moduli(λ, μ)
b = [0.0,0.0,1.0]
t = [0.0,0.0,1.0]
u0 = CLE.Dislocation(b, t, Ciso, Nquad = 4)
uedge = CLE.IsoScrewDislocation3D(λ, μ, b[3])
maxerr = 0.0
maxerr_g = 0.0
for n = 1:10
   x = randvec3()
   maxerr = max( maxerr, vecnorm(u0(x) - uedge(x), Inf) )
   maxerr_g = max(maxerr_g, vecnorm(grad(u0, x) - grad(uedge, x), Inf) )
end
println("maxerr = $maxerr, maxerr_g = $maxerr_g")
@test maxerr < 1e-12
@test maxerr_g < 1e-12

# Random elastic moduli
Crand = randmoduli()

# Generate a random dislocation
θ₁ = 4*π*rand()
ϕ₁ = π*rand()
θ₂ = 4*π*rand()
ϕ₂ = π*rand()
b = [cos(θ₁)*sin(ϕ₁),sin(θ₁)*sin(ϕ₁),cos(ϕ₁)]
t = [cos(θ₂)*sin(ϕ₂),sin(θ₂)*sin(ϕ₂),cos(ϕ₂)]

for (Disl, id, C) in [ (CLE.IsoEdgeDislocation3D(λ, μ, 1.0), "IsoEdgeDislocation3D", Ciso),
         (CLE.IsoScrewDislocation3D(λ, μ, 1.0), "IsoScrewDislocation3D", Ciso),
         (CLE.Dislocation(b,t,Crand, Nquad = 30), "Dislocation(30)", Crand) ]
   print("u = $id : test that ∇u is consistent with u: ")
   maxerr = 0.0
   for n = 1:10
      a, x = randvec3(), randvec3()
      u = x_ -> Disl(x_)
      ∂u = x_ -> grad(Disl, x_) * a
      ∂uad = x_ -> ForwardDiff.jacobian(Disl, x_) * a
      maxerr = max( maxerr, vecnorm(∂u(x) - ∂uad(x), Inf) )
   end
   println("maxerr = $maxerr")
   @test maxerr < 1e-12
end
