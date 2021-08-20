
using LinearAlgebra
using Einsum
using JuLIP
using JuLIP.Potentials

using JuLIPMaterials.CLE

println("=========================")
println(" Testing Elasticity      ")
println("=========================")


println("Testing Voigt conversions")
σ, ϵ = rand(6), rand(6)
σₘ = cauchy_stress(σ)
ϵₘ = cauchy_strain(ϵ)
@test voigt_stress(σₘ) ≈ σ
@test voigt_strain(ϵₘ) ≈ ϵ

# check invariance of scalar product σ⋅ϵ = sum_ij σ_ij⋅ϵ_ij = 
@test  σ' * ϵ ≈ sum([σₘ[i,j] * ϵₘ[i,j] for i=1:3, j=1:3])

C11, C12, C44 = rand(3)
C = voigt_moduli(C11, C12, C44) 
c = elastic_moduli(C)

# check invertability of c_ijkl <--> C_ij conversion
@test voigt_moduli(c) ≈ C 

# check σ_ij = c_ijkl ϵ_kl <--> σ = C⋅ϵ
ϵ = rand(6)
ϵₘ = cauchy_strain(ϵ)
σ = zeros(3,3)
@einsum σ[i,j] = c[i,j,k,l] * ϵₘ[k,l]
@test cauchy_stress(C * ϵ) ≈ σ
@test voigt_stress(σ) ≈ C * ϵ

println("Testing `elastic_moduli`")

eam = JuLIP.Potentials.EAM(test_pots * "w_eam4.fs")
at = bulk(:W)
set_calculator!(at, eam)
variablecell!(at)
minimise!(at)
E0 = energy(at)
c = elastic_moduli(eam, at)
C = voigt_moduli(c)

# apply small random strain and check CLE prediction holds
ϵ = 1e-4*rand(6)
ϵₘ = cauchy_strain(ϵ)
apply_defm!(at, I(3) + ϵₘ)
σₘ = stress(at)
σ = voigt_stress(σₘ)

# check elastic energy matches potential energy
E = energy(at)
@test E - E0 ≈ 0.5 * ϵ' * σ * volume(at) atol=1e-7

# Hooke's law in Voigt form: σ = C ϵ
@test σₘ ≈ cauchy_stress(C * ϵ) atol=1e-5

# check Hooke's law in tensor form: σ_ij = c_ijkl ϵ_kl
σₜ = zeros(3,3)
@einsum σₜ[i,j] = c[i,j,k,l] * ϵₘ[k,l]
@test σₜ ≈ σₘ atol=1e-3

# check C has cubic symmetry and that C11, C12, C44 elastic constants match expected values for this potential
@show cubic_moduli(C, tol=1e-3)
@test all(isapprox.(cubic_moduli(C, tol=1e-3), [3.2644382942616716, 1.2618231295748539, 1.004134784461818], atol=1e-3))

println("Testing rotation of elastic moduli")

A = rotation_matrix(x=[1,1,1], y=[1,-1,0])
at = bulk(:W)
set_calculator!(at, eam)
c = elastic_moduli(at)
c_rot = rotate_moduli(c, A)
C = voigt_moduli(at)
C_rot = rotate_moduli(C, A)

# now we rotate the Atoms directly and compare
apply_defm!(at, A)
c_direct = elastic_moduli(at)
C_direct = voigt_moduli(at)

@test c_rot ≈ c_direct atol=1e-3
@test C_rot ≈ C_direct atol=1e-3
