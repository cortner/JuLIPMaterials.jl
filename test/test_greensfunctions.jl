

using JuLIP, JuLIP.Potentials
import MaterialsScienceTools

MST = MaterialsScienceTools
CLE = MST.CLE
GR = MST.GreensFunctions


Cvoigt = [ 3.0 1.5 1.5 0.0 0.0 0.0
           1.5 3.0 1.5 0.0 0.0 0.0
           1.5 1.5 3.0 0.0 0.0 0.0
           0.0 0.0 0.0 1.2 0.0 0.0
           0.0 0.0 0.0 0.0 1.2 0.0
           0.0 0.0 0.0 0.0 0.0 1.2 ]

C = CLE.elastic_moduli(Cvoigt)

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

print("check that conversion between spherical and euclidean is accurate: ")
maxerr = 0.0
for n = 1:10
   x = rand(3) - 0.5
   x /= norm(x)
   maxerr = max(maxerr, norm(x - GR.euclidean(GR.spherical(x)...)))
end
@test maxerr < 1e-14
println("maxerr = $maxerr")


print("Test agreement between anisotopic and isotropic Gr fcn: ")
λ = 1.0 + rand()
μ = 1.0 + rand()
C = CLE.isotropic_moduli(λ, μ)
maxerr = 0.0
for n = 1:10
   x = (rand(3) - 0.5) * 10.0
   maxerr = max( maxerr,
      vecnorm(GR.GreenTensor3D(x, C)[1] - GR.IsoGreenTensor3D(x, μ, λ)[1], Inf) )
end
println("maxerr = $maxerr")
@test maxerr < 1e-12

print("test that ∇G is consistent with G: ")
maxerr = 0.0
for n = 1:10
   a = rand(3) - 0.5
   a = a / norm(a)
   u = x_ -> (a' * GR.GreenTensor3D(x_, C)[1])[:]
   ∂u = x_ -> reshape(a' * reshape(GR.GreenTensor3D(x_, C)[2], 3, 9), 3, 3)
   x = (rand(3) - 0.5) * 10.0
   maxerr = max( maxerr,
      vecnorm(∂u(x) - ForwardDiff.jacobian(u, x), Inf) * norm(x)^2 )
end
println("maxerr = $maxerr")
@test maxerr < 1e-12


print("test that G satisfies the PDE: ")
maxerr = 0.0
for n = 1:10
   a = rand(3) - 0.5
   a = a / norm(a)
   u = x_ -> GR.GreenTensor3D(x_, C, 10)[1] * a
   x = (rand(3) - 0.5) * 10.0
   maxerr = max( norm(cleforce(x, u, C) * norm(x)^3, Inf), maxerr )
end
println("maxerr = $maxerr")
@test maxerr < 1e-12


Cvoigt = [ 3.0 1.5 1.5 0.0 0.0 0.0
           1.5 3.0 1.5 0.0 0.0 0.0
           1.5 1.5 3.0 0.0 0.0 0.0
           0.0 0.0 0.0 1.2 0.0 0.0
           0.0 0.0 0.0 0.0 1.2 0.0
           0.0 0.0 0.0 0.0 0.0 1.2 ]

Cvoigt = Cvoigt + 0.1 * rand(Cvoigt)
Cvoigt = 0.5 * (Cvoigt + Cvoigt')
C = CLE.elastic_moduli(Cvoigt)


println("Convergence of G with random C: ")
maxerr = 0.0
x = (rand(3) - 0.5) * 3.0
G0 = GR.GreenTensor3D(x, C, 20)[1]
for nquad in (1, 2, 4, 6, 8, 10)
   println("nquad = $nquad => err = ",
         vecnorm(G0 - GR.GreenTensor3D(x, C, nquad)[1]) )
end
