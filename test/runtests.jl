using MaterialsScienceTools
using Base.Test

tests = [
   # "test_elasticity.jl",   # TODO: this one is a joke, needs to be redone
   "test_greensfunctions.jl",
]

println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
println("   MaterialsScienceTools: Start Tests   ")
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
@testset "MaterialsScienceTools" begin
   for t in tests
      @testset "$(t[6:end-3])" begin
         include(t)
      end
   end
end






#
# CLE = MaterialsScienceTools.CLE
# GR = MaterialsScienceTools.GreensFunctions
# using MaterialsScienceTools: Vec3, Ten43
#
# # Cvoigt = [ 3.0 1.5 1.5 0.0 0.0 0.0
# #            1.5 3.0 1.5 0.0 0.0 0.0
# #            1.5 1.5 3.0 0.0 0.0 0.0
# #            0.0 0.0 0.0 1.2 0.0 0.0
# #            0.0 0.0 0.0 0.0 1.2 0.0
# #            0.0 0.0 0.0 0.0 0.0 1.2 ]
#
#
# Cvoigt = rand(6,6) |> Symmetric
#
#
# C = CLE.elastic_moduli(Cvoigt)
# G = GR.GreenFunction(C, Nquad = 10)
#
# μ = 5.123
# λ = 6.789
# Gi = GR.IsoGreenFcn3D(λ, μ)
#
# err = 0.0
# err_g = 0.0
# err_iso = 0.0
# err_iso_g = 0.0
#
# for n = 1:100
#    x = rand(Vec3)
#    x = x * (1.0+rand())/norm(x)
#
#    # Gnew = G(x)
#    # Gold, DGold = GR.GreenTensor3D(x, C, 10)
#    # err = max(err, vecnorm(Gnew - Gold, Inf))
#    # DGnew = GR.grad(G, x)
#    # err_g = max(err_g, vecnorm(DGnew - DGold, Inf))
#
#    Gi_old, DGi_old = GR.IsoGreenTensor3D(x, μ, λ)
#    Gi_new = Gi(x)
#    DGi_new = GR.grad(Gi, x)
#    err_iso = max(err_iso, vecnorm(Gi_old - Gi_new, Inf))
#    err_iso_g = max(err_iso_g, vecnorm(DGi_old - DGi_new, Inf))
# end
# @show err, err_g, err_iso, err_iso_g
#
# using BenchmarkTools, StaticArrays
#
# x = rand(Vec3)
# x = x * (1.0+rand())/norm(x)
# C = Ten43{Float64}(C)
# G = GR.GreenFunction(C, Nquad = 20)
#
# @btime GR.eval_green($x, $C, 20);
# @btime G($x)
# @btime GR.grad($G, $x)
# @btime Gi($x)
# @btime GR.grad($Gi, $x)
# @btime GR.IsoGreenTensor3D($x, $λ, $μ);
# # @btime GR.GreenTensor3D($x, $C, 20)
