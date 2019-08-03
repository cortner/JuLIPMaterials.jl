

using JuLIP, JuLIP.Potentials
using JuLIPMaterials: Vec3
import JuLIPMaterials

MST = JuLIPMaterials
CLE = MST.CLE

using CLE: grad
using MST.Testing
using Einsum
using GaussQuadrature: legendre

println("------------------------------------------------------------")
println(" Testing the 3D Anisotropic Green's Function Implementation")
println("------------------------------------------------------------")

print("check conversion between spherical and euclidean: ")
maxerr = 0.0
for n = 1:10
   x = randvec3(1.0, 1.0)
   maxerr = max(maxerr, norm(x - CLE.euclidean(CLE.spherical(x)...)))
end
println(@test maxerr < 1e-14)
println("maxerr = $maxerr")


print("Test agreement between anisotopic and isotropic Gr fcn: ")
λ = 1.0 + rand()
μ = 1.0 + rand()
Ciso = CLE.isotropic_moduli(λ, μ)
G = CLE.GreenFunction(Ciso, Nquad = 4)
Giso = CLE.IsoGreenFcn3D(λ, μ)
maxerr = 0.0
maxerr_g = 0.0
for n = 1:10
   x = randvec3()
   maxerr = max( maxerr, vecnorm(G(x) - Giso(x), Inf) )
   maxerr_g = max(maxerr_g, vecnorm(grad(G, x) - grad(Giso, x), Inf) )
end
println("maxerr = $maxerr, maxerr_g = $maxerr_g")
println(@test maxerr < 1e-12)
println(@test maxerr_g < 1e-12)

Crand = randmoduli()

for (G, id, C) in [ (CLE.IsoGreenFcn3D(λ, μ), "IsoGreenFcn3D", Ciso),
            #  The next one SHOULD fail, so it is commented out
            # (CLE.GreenFunction(Crand, Nquad = 6), "GreenFunction(6)", Crand),
           (CLE.GreenFunction(Crand, Nquad = 30), "GreenFunction(30)", Crand) ]
   print("G = $id : test that ∇G is consistent with G: ")
   maxerr = 0.0
   for n = 1:10
      a, x = randvec3(), randvec3()
      u = x_ -> (a' * G(x_))[:]
      ∂u = x_ -> reshape(a' *  reshape(grad(G, x_), 3, 9), 3, 3)
      ∂uad = x_ -> ForwardDiff.jacobian(u, x_)
      maxerr = max( maxerr, vecnorm(∂u(x) - ∂uad(x), Inf) )
   end
   println("maxerr = $maxerr")
   println(@test maxerr < 1e-12)
end

for (G, id, C) in [
            (CLE.IsoGreenFcn3D(λ, μ), "IsoGreenFcn3D", Ciso),
            (CLE.GreenFunction(Crand, Nquad = 30), "GreenFunction(30)", Crand) ]
   print("G = $id: test that G satisfies the PDE: ")
   maxerr = 0.0
   for n = 1:10
      a, x = randvec3(), randvec3()
      u = x_ -> G(x_) * a
      maxerr = max( vecnorm(cleforce(Vec3(x), u, C), Inf), maxerr )
   end
   println("maxerr = $maxerr")
   println(@test maxerr < 1e-10)
end

for (G, id, C) in [(CLE.IsoGreenFcn3D(λ, μ), "IsoGreenFcn3D", Ciso),
                   (CLE.GreenFunction(Crand, Nquad = 30), "GreenFunction(30)", Crand) ]
   print("G = $id : test normalisation of G: ")
   err = 0.0

   # Test normal derivative integral over sphere via Gaussian quadrature
   # (Could use Lebedev, but no obvious Julia package)
   n = 30;
   c, w = legendre(n)
   I = zeros(3,3)
   DGnu = zeros(3,3)
   for ω in range(0.0, pi/n, 2*n), i=1:n
      x = [sqrt(1-c[i]^2)*cos(ω),sqrt(1-c[i]^2)*sin(ω),c[i]]
      # TODO: @einsum seems to have changed behaviour or have a bug
      # @einsum DGnu[a,b]  = C[a,β,γ,δ] * CLE.grad(G,x)[b,γ,δ] * x[β]
      for a = 1:3, b = 1:3
         DGnu[a,b] = sum( C[a,β,γ,δ] * CLE.grad(G,x)[b,γ,δ] * x[β]
                          for β=1:3,γ=1:3,δ=1:3 )
      end
      I -= DGnu*w[i]
   end
   I = I*pi/n
   maxerr = norm(I-eye(3))
   println("maxerr = $maxerr")
   println(@test maxerr < 1e-12)
end


println("Convergence of G with random C (please test this visually!): ")
println(" nquad |    err    |   err_g")
println("-------|-----------|-----------")
C = Crand
xtest = [ randvec3() for n = 1:10 ]
G0 = CLE.GreenFunction(C, Nquad = 30)
for nquad in [2, 4, 6, 8, 10, 12, 14]
   G = CLE.GreenFunction(C, Nquad = nquad)
   err = maximum( vecnorm(G0(x) - G(x), Inf)   for x in xtest )
   err_g = maximum( vecnorm(grad(G0, x) - grad(G, x), Inf)   for x in xtest )
   @printf("   %2d  | %.3e | %.3e  \n", nquad, err, err_g)
end
