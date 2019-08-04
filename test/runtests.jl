using JuLIPMaterials
using Test

tests = [
    "test_greensfunctions.jl",
    "test_gfcorrectors3D.jl",
    "test_dislocations.jl",
    "test_forceconstants.jl"
    # "test_cauchyborn1.jl",
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
