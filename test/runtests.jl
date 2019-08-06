using JuLIPMaterials
using Test

tests = [
   # "test_elasticity.jl",   # TODO: this one is a joke, needs to be redone
   "test_greensfunctions.jl",
   "test_dislocations.jl",
   "test_cauchyborn1.jl",
]

println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
println("   JuLIPMaterials: Start Tests   ")
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
@testset "JuLIPMaterials" begin
   for t in tests
      @testset "$(t[6:end-3])" begin
         include(t)
      end
   end
end
