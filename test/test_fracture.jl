using LinearAlgebra
using Statistics
using PyCall
using Optim
using ASE
using JuLIP
using FiniteDifferences
using JuLIPMaterials.CLE
using JuLIPMaterials.Fracture

ase_units = pyimport("ase.units")

# Parameters

R = 50.0 # radius of domain
crack_surface = [0, 0, 1] # y-diretion, the surface which is opened
crack_front = [1, -1, 0] # z-direction, crack front line
relax_elasticity = false # if true, C_ij matrix computed with internal relaxation
relax_surface = true # if true, surface energy computed with internal relaxation

eam = JuLIP.Potentials.EAM(test_pots * "w_eam4.fs")

unitcell = bulk(:W, cubic=true)
@show energy(eam, unitcell)

variablecell!(unitcell)
set_calculator!(unitcell, eam)
minimise!(unitcell)
alat = mean(diag(cell(unitcell)))
@show alat
@test alat ≈ 3.143390027695279

# make a new bulk so it's perfectly cubic
unitcell = bulk(:W, cubic=true, a=alat)
set_calculator!(unitcell, eam)

# 6x6 elastic constant matrix
C = voigt_moduli(unitcell)
C11, C12, C44 = cubic_moduli(C, tol=1e-3)
@show C11, C12, C44
@test all(isapprox.((C11, C12, C44), (3.2644386308256323, 1.2618234185741113, 1.004134918612743), atol=1e-5))

bulk_at = bulk(:W, cubic=true, a=alat, y=crack_front, z=crack_surface) 
slab = bulk_at * (1, 1, 10)
set_calculator!(slab, eam)
surface = deepcopy(slab)
X = positions(surface) |> mat
X[3, :] .+= 2.0
set_positions!(surface, X)
wrap_pbc!(surface)
c = Matrix(surface.cell)
c[3, :] += [0.0, 0.0, 10.0]
set_cell!(surface, c)
fixedcell!(surface)
set_calculator!(surface, eam)
area = norm(cross(slab.cell[:, 1], slab.cell[:, 2]))
γ = (energy(surface) - energy(slab)) / (2 * area)

@show γ / (ase_units.J / ase_units.m^2)
@test γ / (ase_units.J / ase_units.m^2) ≈ 2.950564005097423

# build the final crystal in correct orientation
r_cut = cutoff(eam) # cutoff distance for the potential

atu = bulk(:W, a=alat, y=crack_front, z=crack_surface)
cryst = cluster(atu, R, dims=[1,2], parity=[nothing,1,nothing])
X = positions(cryst) |> mat                     

rac = RectilinearAnisotropicCrack(PlaneStrain(), C11, C12, C44, 
                                  crack_surface, crack_front)

k_G = k1g(rac, γ)
@show k_G

x0, y0, _ = diag(cell(cryst)) ./ 2

# check gradient wrt finite differnces
u, ∇u = u_CLE(rac, cryst, x0, y0)

X0 = copy(X)

U = [u(1.0, 0.0); zeros(length(cryst))']

clust = deepcopy(cryst)
X = X0 + U
set_positions!(clust, X)

@test maximum((central_fdm(2, 1))(α -> u(1.0, α), 0.0) - ∇u(1.0, 0.0)) < 1e-5

r = sqrt.((X0[1,:] .- x0).^2 + (X0[2,:] .- y0).^2)
set_calculator!(clust, eam)
F = norm.(forces(clust))


# exclude anything within cutoff of a surface atom
surface_atoms = findall([sum(norm.(R) .< 3.0) for (i,j,R) in sites(clust, 3.0)] .!= 8)
mask = [all([jj ∉ surface_atoms for jj in j]) for (i,j,R) in sites(clust, cutoff(eam))]

rm, Fm = r[mask], F[mask]

maxF = Float64[]
maxr = Float64[]
bins = range(minimum(rm), maximum(rm), length=10)
for (lower, upper) in zip(bins[1:end-1], bins[2:end])
   push!(maxr, (lower + upper)/2)
   push!(maxF, quantile(Fm[lower .< rm .< upper], 0.95))
end

# p = plot(; yscale=:log10, xscale=:log10, legend=:bottomleft, xlabel="r",
#             ylabel="|f|")
# scatter!(rm, Fm)
# plot!(r, 8*r.^(-2), label="r^-2", lw=2)
# plot!(maxr, maxF, marker=:o)

# fit a straight line to log-log plot to estimate decay
logR = log10.(maxr)
logF = log10.(maxF)
c = [ones(length(logR)) logR] \ logF
@show c
# plot!(10 .^ logR, 10 .^ (c[1] .+ c[2] * logR))

# decay should be at least r^{-2}
@test c[2] < -2
