using Base.Test, JuLIP, JuLIP.Potentials, Einsum
using JuLIPMaterials.CLE, JuLIPMaterials.Testing
using JuLIPMaterials: Vec3, Mat3, ForceConstantMatrix1
using GaussQuadrature: legendre
using JuLIPMaterials.CLE: _C2, _dC2, _C2inv, _dC2inv, _ddC2inv, _C4, _dC4, _ddC4, _corrector_multiplier, _d_corrector_multiplier, _dd_corrector_multiplier

CLE = JuLIPMaterials.CLE
using CLE: elastic_moduli

# Set up a simple reference atomistic calculator
at = bulk(:Cu)
r0 = rnn(:Cu)
lj = lennardjones(r0=r0, rcut=[1.3*r0, 1.7*r0])
set_calculator!(at, lj)
set_constraint!(at, VariableCell(at))
minimise!(at)
# Get force constants
fcm = ForceConstantMatrix1(lj, at, h = 1e-5)
# Get moduli
ℂ = elastic_moduli(at)

println("----------------------------------------------------------")
println(" Testing the 3D Green's Function Corrector Implementation")
println("----------------------------------------------------------")

println("Test multiplier functions: dC2")
maxerr = 0.0
for i=1:10
    a, x = Vec3(randvec3()), Vec3(randvec3())
    ∇C2 = ForwardDiff.derivative(t->_C2(x+t*a,ℂ),0.0)
    dC2 = _dC2(x,ℂ)
    dC2a = zeros(3,3)
    @einsum dC2a[i,j] = dC2[i,j,k]*a[k]
    maxerr = max(maxerr, norm(dC2a-∇C2));
end
println("maxerr = $maxerr")
@test maxerr < 1e-9

println("Test multiplier functions: dC2inv")
maxerr = 0.0
for i=1:10
    a, x = Vec3(randvec3()), Vec3(randvec3())
    ∇C2inv = ForwardDiff.derivative(t->_C2inv(x+t*a,ℂ),0.0)
    dC2inv = _dC2inv(x,ℂ)
    dC2inva = zeros(3,3)
    @einsum dC2inva[i,j] = dC2inv[i,j,k]*a[k]
    maxerr = max(maxerr, norm(dC2inva-∇C2inv));
end
println("maxerr = $maxerr")
@test maxerr < 1e-9

println("Test multiplier functions: ddC2inv")
maxerr = 0.0
for i=1:10
    a, x = Vec3(randvec3()), Vec3(randvec3())
    ∇²C2inv = ForwardDiff.derivative(s->ForwardDiff.derivative(t->_C2inv(x+t*a,ℂ),s),0.0)
    ddC2inv = _ddC2inv(x,ℂ)
    ddC2invaa = zeros(3,3)
    @einsum ddC2invaa[i,j] = ddC2inv[i,j,k,l]*a[k]*a[l]
    maxerr = max(maxerr, norm(ddC2invaa-∇²C2inv));
end
println("maxerr = $maxerr")
@test maxerr < 1e-5

println("Test multiplier functions: dC4")
maxerr = 0.0
for i=1:10
    a, x = Vec3(randvec3()), Vec3(randvec3())
    ∇C4 = ForwardDiff.derivative(t->_C4(x+t*a,fcm),0.0)
    dC4 = _dC4(x,fcm)
    dC4a = zeros(3,3)
    @einsum dC4a[i,j] = dC4[i,j,k]*a[k]
    maxerr = max(maxerr, norm(dC4a-∇C4));
end
println("maxerr = $maxerr")
@test maxerr < 1e-9

println("Test multiplier functions: ddC4")
maxerr = 0.0
for i=1:10
    a, x = Vec3(randvec3()), Vec3(randvec3())
    ∇²C4 = ForwardDiff.derivative(s -> ForwardDiff.derivative(t->_C4(x+t*a,fcm),s),0.0)
    ddC4 = _ddC4(x,fcm)
    ddC4aa = zeros(3,3)
    @einsum ddC4aa[i,j] = ddC4[i,j,k,l]*a[k]*a[l]
    maxerr = max(maxerr, norm(ddC4aa-∇²C4));
end
println("maxerr = $maxerr")
@test maxerr < 1e-5

println("Test multiplier functions: d_corrector_multiplier")
maxerr = 0.0
for i=1:10
    a, x = Vec3(randvec3()), Vec3(randvec3())
    ∇H4 = ForwardDiff.derivative(t->_corrector_multiplier(x+t*a,ℂ,fcm),0.0)
    dH4 = _d_corrector_multiplier(x,ℂ,fcm)
    dH4a = zeros(3,3)
    @einsum dH4a[i,j] = dH4[i,j,k]*a[k]
    maxerr = max(maxerr, norm(dH4a-∇H4));
end
println("maxerr = $maxerr")
@test maxerr < 1e-9

println("Test multiplier functions: dd_corrector_multiplier")
maxerr = 0.0
for i=1:10
    a, x = Vec3(randvec3()), Vec3(randvec3())
    ∇²H4 = ForwardDiff.derivative(s->ForwardDiff.derivative(t->_corrector_multiplier(x+t*a,ℂ,fcm),s),0.0)
    ddH4 = _dd_corrector_multiplier(x,ℂ,fcm)
    ddH4aa = zeros(3,3)
    @einsum ddH4aa[i,j] = ddH4[i,j,k,l]*a[k]*a[l]
    maxerr = max(maxerr, norm(ddH4aa-∇²H4));
end
println("maxerr = $maxerr")
@test maxerr < 1e-5

# # Set up Green's function and corrector
# G0 = GreenFunction(ℂ, Nquad = 32)
# Gcorr = GreenFunctionCorrector(ℂ, fcm, Nquad = 32)
#
# println("Test that Gcorr satisfies the PDE div(C2 ∇Gcorr) = C4[∇]G0 ")
# maxerr = 0.0
# # for i=1:10
#
#     a, x = randvec3(), randvec3()
#     # Compute LHS
#     δ = 1e-3;
#     e1 = Vec3(1.0,0.0,0.0)
#     e2 = Vec3(0.0,1.0,0.0)
#     e3 = Vec3(0.0,0.0,1.0)
#     D²Gcorr = zeros(Vec3,3,3)
#     D²Gcorr[1,1] = (Gcorr(x+δ*e1)-2*Gcorr(x)+Gcorr(x-δ*e1))*a/δ^2
#     D²Gcorr[1,2] = (Gcorr(x+δ*(e1+e2)/2)-2*Gcorr(x)+Gcorr(x-δ*(e1+e2)/2))*a/δ^2
#     D²Gcorr[1,3] = (Gcorr(x+δ*(e1+e3)/2)-2*Gcorr(x)+Gcorr(x-δ*(e1+e3)/2))*a/δ^2
#     D²Gcorr[2,1] = D²Gcorr[2,1]
#     D²Gcorr[2,2] = (Gcorr(x+δ*e2)-2*Gcorr(x)+Gcorr(x-δ*e2))*a/δ^2
#     D²Gcorr[2,3] = (Gcorr(x+δ*(e2+e3)/2)-2*Gcorr(x)+Gcorr(x-δ*(e2+e3)/2))*a/δ^2
#     D²Gcorr[3,1] = D²Gcorr[1,3]
#     D²Gcorr[3,2] = D²Gcorr[2,3]
#     D²Gcorr[3,3] = (Gcorr(x+δ*e3)-2*Gcorr(x)+Gcorr(x-δ*e3))*a/δ^2
#     LHS = zeros(3)
#     for i=1:3, j=1:3
#         LHS += ℂ[:,i,:,j]*D²Gcorr[i,j]
#     end
#     # Compute RHS
#     f0 = zeros(3);
#
#     for i=1:length(fcm.R)
#         u0 = t -> G0(x + t * fcm.R[i])*a
#         u1 = t -> ForwardDiff.derivative(u0,t)
#         u2 = t -> ForwardDiff.derivative(u1,t)
#         u3 = t -> ForwardDiff.derivative(u2,t)
#         u4 = t -> ForwardDiff.derivative(u3,t)
#         f0 += fcm.H[i]*u4(0.0)/24
#     end
#     println(f0);
#     # Compare
#     maxerr = max( norm(f0-f1, Inf), maxerr )
# # end
# println("maxerr = $maxerr")
# @test maxerr < 1e-9


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
