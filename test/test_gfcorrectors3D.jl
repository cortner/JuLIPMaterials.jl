using Base.Test, JuLIP, JuLIP.Potentials
using JuLIPMaterials.CLE, JuLIPMaterials.Testing
using JuLIPMaterials: Vec3, Mat3, ForceConstantMatrix1
using Einsum
using GaussQuadrature: legendre

CLE = JuLIPMaterials.CLE
using CLE: elastic_moduli

# Set up a reference atomistic calculator
datapath = joinpath(Pkg.dir("JuLIP"), "data")
eam_Fe = EAM(datapath * "/pfe.plt",
             datapath * "/ffe.plt",
             datapath * "/F_fe.plt")

# equilibrate a unit cell
fe1 = bulk(:Fe)
set_constraint!(fe1, VariableCell(fe1))
set_calculator!(fe1, eam_Fe)
minimise!(fe1)
# get the force constants
fcm = ForceConstantMatrix1(eam_Fe, fe1, h = 1e-5)
# symmetrise (hack for now)
for i=1:length(fcm.H)
    fcm.H[i] = 0.5*(fcm.H[i]+fcm.H[i]')
end
# get the elasticity tensor
ℂ = elastic_moduli(fcm);

# # Set up a simple cubic example.
# R1 = [ Vec3(1.0,0.0,0.0), Vec3(-1.0,0.0,0.0), Vec3(0.0,1.0,0.0), Vec3(0.0,-1.0,0.0), Vec3(0.0,0.0,1.0), Vec3(0.0,0.0,-1.0) ];
# R2 = [];
# for i=-1.0:1.0, j=-1.0:1.0, k=-1.0:1.0
#     if i*j*k == 0 & ~(Vec3(i,j,k) in R1)
#         push!(R2,Vec3(i,j,k));
#     end
# end
# R = [R1[:]; R2[:]];
# R = Array{StaticArrays.SArray{Tuple{3},Float64,1,3},1}(R)
#
# a = randn()^2; b = randn()^2;
# H1 = Mat3(a*(3*eye(3)-ones(3,3)) + b*eye(3))
# c = 0.5*randn()^2; d = 0.5*randn()^2;
# H2 = Mat3(c*(3*eye(3)-ones(3,3)) + d*eye(3))
#
# H = [];
# for i=1:length(R1)
#     push!(H,H1);
# end
# for i=1:length(R2)
#     push!(H,H2);
# end
# H = Array{StaticArrays.SArray{Tuple{3,3},Float64,2,9},1}(H)
#
# fcm = ForceConstantMatrix1(R,H);
# # get the elasticity tensor
# ℂ = elastic_moduli(fcm);

println("----------------------------------------------------------")
println(" Testing the 3D Green's Function Corrector Implementation")
println("----------------------------------------------------------")

Gf = GreenFunction(ℂ, Nquad = 8)
Gcorr = GreenFunctionCorrector(ℂ, fcm, Nquad = 8)

println("Test that Gcorr satisfies the PDE div(C2 ∇Gcorr) = C4[∇]G ")
maxerr = 0.0
# for i=1:10

    a, x = randvec3(), randvec3()
    # Compute LHS
    v = x_ -> Gcorr(x_)*a
    f1 = cleforce(Vec3(x), v, ℂ)
    println(f1);
    # Compute RHS
    f0 = zeros(3);
    for i=1:length(fcm.R)
        u0 = t -> Gf(x+ t * fcm.R[i])*a
        u1 = t -> ForwardDiff.derivative(u0,t)
        u2 = t -> ForwardDiff.derivative(u1,t)
        u3 = t -> ForwardDiff.derivative(u2,t)
        u4 = t -> ForwardDiff.derivative(u3,t)
        f0 += fcm.H[i]*u4(0.0)/24
    end
    println(f0);
    # Compare
    maxerr = max( norm(f0-f1, Inf), maxerr )
# end
println("maxerr = $maxerr")
@test maxerr < 1e-9


# for (G, id, C) in [(CLE.IsoGreenFcn3D(λ, μ), "IsoGreenFcn3D", Ciso),
#                    (CLE.GreenFunction(Crand, Nquad = 30), "GreenFunction(30)", Crand) ]
#    print("G = $id : test normalisation of G: ")
#    err = 0.0
#
#    # Test normal derivative integral over sphere via Gaussian quadrature
#    # (Could use Lebedev, but no obvious Julia package)
#    n = 30;
#    c, w = legendre(n)
#    I = zeros(3,3)
#    DGnu = zeros(3,3)
#    for ω in range(0.0, pi/n, 2*n), i=1:n
#       x = Vec3(sqrt(1-c[i]^2)*cos(ω),sqrt(1-c[i]^2)*sin(ω),c[i])
#       DG = CLE.grad(G,x)
#       @einsum DGnu[a,b] = C[a,β,γ,δ] * DG[b,γ,δ] * x[β]
#       I -= DGnu * w[i]
#    end
#    I = I*pi/n
#    maxerr = norm(I-eye(3))
#    println("maxerr = $maxerr")
#    @test maxerr < 1e-12
# end
#
#
# println("Convergence of Gcorr with random C (please test this visually!): ")
# println(" nquad |    err    |   err_g")
# println("-------|-----------|-----------")
# C = Crand
# xtest = [ randvec3() for n = 1:10 ]
# G0 = CLE.GreenFunction(C, Nquad = 64)
# for nquad in [2, 4, 8, 16, 32]
#    G = CLE.GreenFunction(C, Nquad = nquad)
#    err = maximum( vecnorm(G0(x) - G(x), Inf)   for x in xtest )
#    err_g = maximum( vecnorm(grad(G0, x) - grad(G, x), Inf)   for x in xtest )
#    @printf("   %2d  | %.3e | %.3e  \n", nquad, err, err_g)
# end
