using JuLIPMaterials
using Base.Test

tests = [
   # "test_elasticity.jl",   # TODO: this one is a joke, needs to be redone
   # "test_greensfunctions.jl",
   # "test_gfcorrectors3D.jl",
   # "test_dislocations.jl",
   #"test_cauchyborn1.jl",
   "test_forceconstants.jl"
]

println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
println("   JuLIPMaterials: Start Tests   ")
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
@testset "JuLIPMaterials" begin
   for t in tests
      ts = @testset "$(t[6:end-3])" begin
         include(t)
      end
      println(ts)
   end
end
