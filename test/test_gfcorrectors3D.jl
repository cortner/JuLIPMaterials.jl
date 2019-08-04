using JuLIPMaterials, ForwardDiff
using Test, JuLIP, JuLIP.Potentials, Einsum
using JuLIPMaterials.CLE, JuLIPMaterials.Testing
using JuLIPMaterials: Vec3, Mat3, ForceConstantMatrix1
using GaussQuadrature: legendre
using JuLIPMaterials.CLE: _C2, _dC2, _C2inv, _dC2inv, _dC2inv1, _ddC2inv,
                          _ddC2inv1, _C4, _dC4, _ddC4, _corrector_multiplier,
                          _d_corrector_multiplier, _dd_corrector_multiplier,
                          elastic_moduli

const CLE = JuLIPMaterials.CLE

using LinearAlgebra, Printf
##

# Set up a simple reference atomistic calculator
at = bulk(:Cu)
r0 = rnn(:Cu)
lj = lennardjones(r0=r0, rcut=[1.3*r0, 1.7*r0])
set_calculator!(at, lj)
variablecell!(at)
minimise!(at)
# Get force constants
fcm = ForceConstantMatrix1(lj, at, h = 1e-5)
# Get moduli
ℂ = elastic_moduli(at)

println("----------------------------------------------------------")
println(" Testing the 3D Green's Function Corrector Implementation")
println("----------------------------------------------------------")

@info("Test multiplier functions: dC2")
maxerr = 0.0
for i=1:10
    a, x = Vec3(randvec3()), Vec3(randvec3())
    ∇C2 = ForwardDiff.derivative(t->_C2(x+t*a,ℂ),0.0)
    dC2 = _dC2(x,ℂ)
    dC2a = zeros(3,3)
    @einsum dC2a[i,j] = dC2[i,j,k]*a[k]
    global maxerr = max(maxerr, norm(dC2a-∇C2));
end
println("maxerr = $maxerr: ")
println(@test maxerr < 1e-9)

@info("Test multiplier functions: dC2inv")
maxerr = 0.0
for i=1:10
    a, x = Vec3(randvec3()), Vec3(randvec3())
    ∇C2inv = ForwardDiff.derivative(t->_C2inv(x+t*a,ℂ),0.0)
    dC2inv = _dC2inv(x,ℂ)
    dC2inva = zeros(3,3)
    @einsum dC2inva[i,j] = dC2inv[i,j,k]*a[k]
    global maxerr = max(maxerr, norm(dC2inva-∇C2inv));
end
print("maxerr = $maxerr: ")
println(@test maxerr < 1e-9)

@info("Test multiplier functions: dC2inv1")
maxerr = 0.0
for i=1:10
    a, x = Vec3(randvec3()), Vec3(randvec3())
    ∇C2inv = ForwardDiff.derivative(t->_C2inv(x+t*a,ℂ),0.0)
    dC2inv1 = _dC2inv1(x,ℂ)
    dC2inv1a = zeros(3,3)
    @einsum dC2inv1a[i,j] = dC2inv1[i,j,k]*a[k]
    global maxerr = max(maxerr, norm(dC2inv1a-∇C2inv));
end
print("maxerr = $maxerr: ")
println(@test maxerr < 1e-9)

@info("Test multiplier functions: ddC2inv")
@warn("THIS TEST IS FAILING; TODO -> FIX IT")
maxerr = 0.0
for i=1:10
    a, x = Vec3(randvec3()), Vec3(randvec3())
    ∇²C2inv = ForwardDiff.derivative(s -> ForwardDiff.derivative(t->_C2inv(x+t*a,ℂ),s),0.0)
    ddC2inv = _ddC2inv(x,ℂ)
    ddC2invaa = zeros(3,3)
    @einsum ddC2invaa[i,j] = ddC2inv[i,j,k,l]*a[k]*a[l]
    global maxerr = max(maxerr, norm(ddC2invaa-∇²C2inv))
end
print("maxerr = $maxerr: ")
# println(@test maxerr < 1e-5)

@info("Test multiplier functions: ddC2inv1")
@warn("THIS TEST IS FAILING; TODO -> FIX IT")
maxerr = 0.0
for i=1:10
    a, x = Vec3(randvec3()), Vec3(randvec3())
    ∇²C2inv = ForwardDiff.derivative(s -> ForwardDiff.derivative(t->_C2inv(x+t*a,ℂ),s),0.0)
    ddC2inv1 = _ddC2inv1(x,ℂ)
    ddC2inv1aa = zeros(3,3)
    @einsum ddC2inv1aa[i,j] = ddC2inv1[i,j,k,l]*a[k]*a[l]
    global maxerr = max(maxerr, norm(ddC2inv1aa-∇²C2inv));
end
print("maxerr = $maxerr: ")
# println(@test maxerr < 1e-5)


@info("Test multiplier functions: dC4")
maxerr = 0.0
for i=1:10
    a, x = Vec3(randvec3()), Vec3(randvec3())
    ∇C4 = ForwardDiff.derivative(t->_C4(x+t*a,fcm),0.0)
    dC4 = _dC4(x,fcm)
    dC4a = zeros(3,3)
    @einsum dC4a[i,j] = dC4[i,j,k]*a[k]
    global maxerr = max(maxerr, norm(dC4a-∇C4));
end
print("maxerr = $maxerr: ")
println(@test maxerr < 1e-9)

@info("Test multiplier functions: ddC4")
maxerr = 0.0
for i=1:10
    a, x = Vec3(randvec3()), Vec3(randvec3())
    ∇²C4 = ForwardDiff.derivative(s -> ForwardDiff.derivative(t->_C4(x+t*a,fcm),s),0.0)
    ddC4 = _ddC4(x,fcm)
    ddC4aa = zeros(3,3)
    @einsum ddC4aa[i,j] = ddC4[i,j,k,l]*a[k]*a[l]
    global maxerr = max(maxerr, norm(ddC4aa-∇²C4));
end
print("maxerr = $maxerr: ")
println(@test maxerr < 1e-5)

@info("Test multiplier functions: d_corrector_multiplier")
maxerr = 0.0
for i=1:10
    a, x = Vec3(randvec3()), Vec3(randvec3())
    ∇H4 = ForwardDiff.derivative(t->_corrector_multiplier(x+t*a,ℂ,fcm),0.0)
    dH4 = _d_corrector_multiplier(x,ℂ,fcm)
    dH4a = zeros(3,3)
    @einsum dH4a[i,j] = dH4[i,j,k]*a[k]
    global maxerr = max(maxerr, norm(dH4a-∇H4));
end
print("maxerr = $maxerr: ")
println(@test maxerr < 1e-9)

@info("Test multiplier functions: dd_corrector_multiplier")
@warn("TODO: THIS TEST IS FAILING")
maxerr = 0.0
for i=1:10
    a, x = Vec3(randvec3()), Vec3(randvec3())
    ∇²H4 = ForwardDiff.derivative(s->ForwardDiff.derivative(t->_corrector_multiplier(x+t*a,ℂ,fcm),s),0.0)
    ddH4 = _dd_corrector_multiplier(x,ℂ,fcm)
    ddH4aa = zeros(3,3)
    @einsum ddH4aa[i,j] = ddH4[i,j,k,l]*a[k]*a[l]
    global maxerr = max(maxerr, norm(ddH4aa-∇²H4));
end
println("maxerr = $maxerr")
# @test maxerr < 1e-5

# Set up Green's function and corrector
G0 = GreenFunction(ℂ, Nquad = 32)
Gcorr = GreenFunctionCorrector(ℂ, fcm, Nquad = 32)

println("Test that Gcorr satisfies the PDE div(C2 ∇Gcorr) = C4[∇]G0 ")
@info("TODO: THIS TEST IS FAILING!")
maxerr = 0.0
for i=1:5
    a, x = randvec3(), randvec3()
    # Compute LHS via AD
    v = x -> Gcorr(x)*a
    LHS = cleforce(Vec3(x),v,ℂ)
    # Compute RHS via AD
    RHS = zeros(3)
    for i=1:length(fcm.R)
        # Function to be differentiated
        u0 = t -> G0(x + t * fcm.R[i])*a
        # Compute 4th order derivative in ρ direction
        u1 = t -> ForwardDiff.derivative(u0,t)
        u2 = t -> ForwardDiff.derivative(u1,t)
        u3 = t -> ForwardDiff.derivative(u2,t)
        u4 = t -> ForwardDiff.derivative(u3,t)
        RHS += fcm.H[i]*u4(0.0)/24
    end
    # Compare
    global maxerr = max( norm(LHS-RHS, Inf), maxerr )
end
print("maxerr = $maxerr: ")
# println(@test maxerr < 1e-4)

println("Convergence of Gcorr (please test this visually!): ")
println(" nquad |    err    ")
println("-------|-----------")
xtest = [ randvec3() for n = 1:10 ]
Gref = GreenFunctionCorrector(ℂ, fcm, Nquad = 64)
for nquad in [2, 4, 8, 16, 32]
   global Gref, xtest, fcm, ℂ
   G = GreenFunctionCorrector(ℂ, fcm, Nquad = nquad)
   err = maximum( norm(Gref(x) - G(x), Inf)   for x in xtest )
   @printf("   %2d  | %.3e \n", nquad, err)
end
