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


# using BenchmarkTools, StaticArrays
# typealias Vec3{T} SVector{3, T}
# typealias Mat3{T} SMatrix{3,3,T}
# contract1(a, C, b) = Mat3([dot(C[i,:,j,:] * b, a) for i=1:3, j = 1:3])
# # contract2(a, C, b) = Mat3(NTuple{9, Float64}((dot(C[i,:,j,:] * b, a) for i=1:3, j = 1:3)))
# @eval contract3(a, C, b) = Mat3($(Expr(:tuple, (:(dot(C[$i,:,$j,:] * b, a)) for i=1:3, j=1:3)...)))
# a, b = rand(Vec3), rand(Vec3)
# C = @SArray rand(3,3,3,3)
# (@benchmark contract1($a, $C, $b)) |> display; println()
# # (@benchmark contract2(a, C, b)) |> display
# (@benchmark contract3($a, $C, $b)) |> display; println()
