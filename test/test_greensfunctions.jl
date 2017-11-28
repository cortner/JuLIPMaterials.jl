

using JuLIP, JuLIP.Potentials
import MaterialsScienceTools

MST = MaterialsScienceTools
CLE = MST.CLE

using CLE: grad


function randvec3(s0=1.0, s1=2.0)
   x = rand(3) - 0.5
   return (x / norm(x)) * (s0 + rand() * (s1-s0))
end

function randmoduli(rnd = 0.1)
   Cv = [ 3.0 1.5 1.5 0.0 0.0 0.0
              1.5 3.0 1.5 0.0 0.0 0.0
              1.5 1.5 3.0 0.0 0.0 0.0
              0.0 0.0 0.0 1.2 0.0 0.0
              0.0 0.0 0.0 0.0 1.2 0.0
              0.0 0.0 0.0 0.0 0.0 1.2 ]
   Cv += Symmetric( 2*rnd*(rand(6, 6)-0.5) )
   return CLE.elastic_moduli(Cv)
end

∷(C::Array{Float64, 3}, F::Matrix) = reshape(C, 3, 9) * F[:]

# div C ∇u = C_iajb u_j,ab
function cleforce(x, u, C)
    f = zeros(3)
    for j = 1:3
        ujab = ForwardDiff.hessian(y->u(y)[j], x)
        f += C[:,:,j,:] ∷ ujab
    end
    return JVecF(f)
end

println("------------------------------------------------------------")
println(" Testing the 3D Anisotropic Green's Function Implementation")
println("------------------------------------------------------------")

print("check conversion between spherical and euclidean: ")
maxerr = 0.0
for n = 1:10
   x = randvec3(1.0, 1.0)
   maxerr = max(maxerr, norm(x - CLE.euclidean(CLE.spherical(x)...)))
end
@test maxerr < 1e-14
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
@test maxerr < 1e-12
@test maxerr_g < 1e-12

Crand = randmoduli()

for (G, id, C) in [ (CLE.IsoGreenFcn3D(λ, μ), "IsoGreenFcn3D", Ciso),
           (CLE.GreenFunction(Crand, Nquad = 6), "GreenFunction(6)", Crand),
           (CLE.GreenFunction(Crand, Nquad = 30), "GreenFunction(30)", Crand) ]
   print("G = $id : test that ∇G is consistent with G: ")
   maxerr = 0.0
   for n = 1:10
      a, x = randvec3(), randvec3()

      u = x_ -> G(x_)[1,1]
      ∂u = x_ -> grad(G, x_)[1,1,:]
      ∂uad = x_ -> ForwardDiff.gradient(u, x_)

      # u = x_ -> (a' * G(x_))[:]
      # ∂u = x_ -> reshape(a' *  reshape(grad(G, x_), 3, 9), 3, 3)
      # ∂uad = x_ -> ForwardDiff.jacobian(u, x_)

      maxerr = max( maxerr, vecnorm(∂u(x) - ∂uad(x), Inf) )
   end
   println("maxerr = $maxerr")
   @test maxerr < 1e-12
end

for (G, id, C) in [
            (CLE.IsoGreenFcn3D(λ, μ), "IsoGreenFcn3D", Ciso),
            (CLE.GreenFunction(Crand, Nquad = 30), "GreenFunction(30)", Crand) ]
   print("G = $id: test that G satisfies the PDE: ")
   maxerr = 0.0
   for n = 1:10
      a, x = randvec3(), randvec3()
      u = x_ -> G(x_) * a
      maxerr = max( vecnorm(cleforce(x, u, C), Inf), maxerr )
   end
   println("maxerr = $maxerr")
   @test maxerr < 1e-12
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
