
using JuLIP

import JuLIPMaterials
reload("JuLIPMaterials")
JM = JuLIPMaterials

at = bulk(:Fe)
calc = SitePotential(LennardJones(r0 = rnn(:Fe)) * C2Shift(2.5 * rnn(:Fe)))

lGF = JM.LGFs.LGF(calc, at)




# using JuLIPMaterials: Vec3, Mat3
# R = rand(SVec{Float64}, 10)
# H = rand(SMat{Float64}, 10)
