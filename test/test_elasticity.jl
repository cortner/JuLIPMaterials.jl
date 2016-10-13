
using JuLIP
using JuLIP.ASE
using JuLIP.Potentials

using MaterialsScienceTools.Elasticity: elastic_moduli

println("=========================")
println(" Testing Elasticity      ")
println("=========================")

println("Testing `elastic_moduli`")
at = bulk("Cu")
calc = lennardjones(r0 = rnn("Cu"))
C = elastic_moduli(calc, at)

C2d = C[1:2,1:2,1:2,1:2]
println( reshape(C2d, 4, 4) )
