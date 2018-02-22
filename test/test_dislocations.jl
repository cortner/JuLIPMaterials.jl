

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

# Random Lam\'e Parameters
λ = 1.0 + rand()
μ = 1.0 + rand()
Ciso = CLE.isotropic_moduli(λ, μ)

# Test explicit formulae vs anistropic ones
print("Test agreement between anisotopic and isotropic edge dislocation: ")
b = [1.0,0.0,0.0]
t = [0.0,0.0,1.0]
u0 = CLE.Dislocation(b, t, Ciso, Nquad = 30)
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

print("Test agreement between anisotopic and isotropic screw dislocation: ")
Ciso = CLE.isotropic_moduli(λ, μ)
b = [0.0,0.0,1.0]
t = [0.0,0.0,1.0]
u0 = CLE.Dislocation(b, t, Ciso, Nquad = 40)
uscrew = CLE.IsoScrewDislocation3D(λ, μ, b[3])
maxerr = 0.0
maxerr_g = 0.0
for n = 1:10
   x = randvec3()
   maxerr = max( maxerr, vecnorm(u0(x) - uscrew(x), Inf) )
   maxerr_g = max(maxerr_g, vecnorm(grad(u0, x) - grad(uscrew, x), Inf) )
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


for (Disl, id, C) in [(CLE.IsoEdgeDislocation3D(λ, μ, 1.0), "IsoEdgeDislocation3D", Ciso),
         (CLE.IsoScrewDislocation3D(λ, μ, 1.0), "IsoScrewDislocation3D", Ciso),
         (CLE.Dislocation(b,t,Crand, Nquad = 30), "Dislocation(30)", Crand) ]
   print("u = $id: test that u satisfies the PDE: ")
   maxerr = 0.0
   for n = 1:10
      a, x = randvec3(), randvec3()
      u = x_ -> Disl(x_)
      maxerr = max( vecnorm(cleforce(x, u, C), Inf), maxerr )
   end
   println("maxerr = $maxerr")
   @test maxerr < 1e-12
end

# Test Burgers vector
Disl = CLE.IsoEdgeDislocation3D(λ, μ, 1.0)
print("u = IsoEdgeDislocation3D: test Burgers vector: ")
err = 0.0
# Integrate around loop
n = 30;
I = zeros(3)
DuTau = zeros(3)
for ω in range(0.0, pi/n, 2*n)
   x = [cos(ω),sin(ω),0.0]
   tau = [-sin(ω),cos(ω),0.0]
   I += grad(Disl,x) * tau
end
I = I*pi/n
maxerr = vecnorm(I-[1.0,0.0,0.0])
println("maxerr = $maxerr")
@test maxerr < 1e-12

Disl = CLE.IsoScrewDislocation3D(λ, μ, 1.0)
print("u = IsoScrewDislocation3D: test Burgers vector: ")
err = 0.0
# Integrate around loop
n = 30;
I = zeros(3)
DuTau = zeros(3)
for ω in range(0.0, pi/n, 2*n)
   x = [cos(ω),sin(ω),0.0]
   tau = [-sin(ω),cos(ω),0.0]
   I += grad(Disl,x) * tau
end
I = I*pi/n
maxerr = vecnorm(I-[0.0,0.0,1.0])
println("maxerr = $maxerr")
@test maxerr < 1e-12

Disl = CLE.Dislocation(b,[0.0,0.0,1.0],Crand, Nquad = 40)
print("u = Dislocation(30): test Burgers vector: ")
err = 0.0
# Integrate around loop
n = 30;
I = zeros(3)
DuTau = zeros(3)
for ω in range(0.0, pi/n, 2*n)
   x = [cos(ω),sin(ω),0.0]
   tau = [-sin(ω),cos(ω),0.0]
   I += grad(Disl,x) * tau
end
I = I*pi/n
maxerr = vecnorm(I-b)
println("maxerr = $maxerr")
@test maxerr < 1e-12

println("Convergence of u with random C (please test this visually!): ")
println(" nquad |    err    |   err_g")
println("-------|-----------|-----------")
C = Crand
xtest = [ randvec3() for n = 1:10 ]
u0 = CLE.Dislocation(b,t,C, Nquad = 40)
for nquad in [2, 4, 6, 8, 10, 12, 14]
   u = CLE.Dislocation(b,t,C, Nquad = nquad)
   err = maximum( vecnorm(u0(x) - u(x), Inf)   for x in xtest )
   err_g = maximum( vecnorm(grad(u0, x) - grad(u, x), Inf)   for x in xtest )
   @printf("   %2d  | %.3e | %.3e  \n", nquad, err, err_g)
end
