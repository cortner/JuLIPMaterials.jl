using JuLIPMaterials
using Test
using JuLIP 

# use the same DataDep as JuLIP
test_pots = JuLIP.Deps.fetch_test_pots()

tests = [
    "test_greensfunctions.jl",
    "test_gfcorrectors3D.jl",
    "test_dislocations.jl",
    "test_elasticity.jl",
    "test_fracture.jl",
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
