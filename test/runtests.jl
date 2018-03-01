using MaterialsScienceTools
using Base.Test

tests = [
   # "test_elasticity.jl",   # TODO: this one is a joke, needs to be redone
   #"test_greensfunctions.jl",
   "test_dislocations.jl",
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
