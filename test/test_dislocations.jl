

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

print("Test agreement between anisotopic and isotropic edge dislocation: ")
λ = 1.0 + rand()
μ = 1.0 + rand()
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
