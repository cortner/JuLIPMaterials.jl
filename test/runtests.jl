using MaterialsScienceTools
using Base.Test

tests = [
   # "test_elasticity.jl",   # TODO: this one is a joke, needs to be redone
   "test_greensfunctions.jl",
]

# println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
# println("   MaterialsScienceTools: Start Tests   ")
# println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
# @testset "MaterialsScienceTools" begin
#    for t in tests
#       @testset "$(t[6:end-3])" begin
#          include(t)
#       end
#    end
# end







CLE = MaterialsScienceTools.CLE
GR = MaterialsScienceTools.GreensFunctions
using MaterialsScienceTools: Vec3, Ten43

# Cvoigt = [ 3.0 1.5 1.5 0.0 0.0 0.0
#            1.5 3.0 1.5 0.0 0.0 0.0
#            1.5 1.5 3.0 0.0 0.0 0.0
#            0.0 0.0 0.0 1.2 0.0 0.0
#            0.0 0.0 0.0 0.0 1.2 0.0
#            0.0 0.0 0.0 0.0 0.0 1.2 ]


Cvoigt = rand(6,6) |> Symmetric


C = CLE.elastic_moduli(Cvoigt)
G = GR.GreenFunction(C, Nquad = 10)

err = 0.0
err_g = 0.0
for n = 1:10
   x = rand(Vec3)
   x /= norm(x)
   Gnew = G(x)
   Gold, DGold = GR.GreenTensor3D(x, C, 10)
   err = max(err, vecnorm(Gnew - Gold, Inf))
   DGnew = GR.grad(G, x)
   err_g = max(err_g, vecnorm(DGnew - DGold, Inf))
end
@show err, err_g

using BenchmarkTools, StaticArrays

x = rand(Vec3)
x /= norm(x)
C = Ten43{Float64}(C)
G = GR.GreenFunction(C, Nquad = 20)

@btime GR.eval_green($x, $C, 20);
@btime G($x)
@btime GR.grad($G, $x)
@btime GR.GreenTensor3D($x, $C, 20)
