using JuLIPMaterials
using Test
using DataDeps

# use the same DataDep as JuLIP
register(DataDep(
    "JuLIP_testpots",
    "A few EAM potentials for testing",
    "https://www.dropbox.com/s/leub1c9ft1mm9fg/JuLIP_data.zip?dl=1",
    post_fetch_method = file -> run(`unzip $file`)
    ))

test_pots = joinpath(datadep"JuLIP_testpots", "JuLIP_data") * "/"

tests = [
    "test_greensfunctions.jl",
    "test_gfcorrectors3D.jl",
    "test_dislocations.jl",
    "test_elasticity.jl",
    "test_fracture.jl",
   # "test_forceconstants.jl"
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
