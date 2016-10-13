
module Dislocations

using JuLIP
using JuLIP.ASE

const Afcc = JMatF([ 0.0 1 1; 1 0 1; 1 1 0])

"""
ensure that species S actually crystallises to FCC
"""
function check_fcc(S::AbstractString)
   F = defm(bulk(S))
   @assert vecnorm(F/F[1,2] - Afcc) < 1e-12
end










"""
`u_edge_isotropic(x, y, b, ν) -> u_x, u_y`

compute the displacement field `ux, uy` for an edge dislocation in an
isotropic linearly elastic medium, with core at (0,0),
burgers vector `b * [1.0;0.0]` and Poisson ratio `ν`

This is to be used primarily for comparison, since the exact solution will
not be the isotropic elasticity solution.
"""
function u_edge_isotropic(x, y, b, ν)
   x[y .< 0] += b/2
   rsq = x.^2 + y.^2
   ux = b/(2*π) * ( arctan(x ./ y) + (x .* y) ./ (2*(1-ν) * rsq) )
   uy = -b/(2*π) * ( (1-2*ν)/(4*(1-ν)) * log(rsq) + (x.^2 - y.^2) ./ (4*(1-nu) * rsq) )
   return ux, uy
end





project12(x, E) = [dot(x, E[1]); dot(x, E[2])]


function basis_a12(at)
    F = defm(at)
    a1, a2, a3 = F[:,1], F[:,2], F[:,3]
    a1 /= vecnorm(a1); a2 /= vecnorm(a2); a3 /= vecnorm(a3)
    e1 = a1
    e2 = a2 - dot(a1, a2) * a1; e2 /= vecnorm(e2)
    e3 = e1 × e2
    return (e1, e2, e3)
end


function pos2d_fcc(at)
    X = positions(at)
    e1, e2, e3 = basis_a12(at)
    x = [dot(e1, x) for x in X]
    y = [dot(e2, x) for x in X]
    z = [dot(e3, x) for x in X]
    zmax = maximum(z) / 2
    z = [ round(Int, zi/zmax) for zi in z ]
    I0 = find(z .== 0)
    I1 = find(z .== 1)
    I2 = find(z .== 2)
    return x, y, I0, I1, I2
end




"""
generates a reference configuration
"""
function edge_fcc(S::AbstractString, R::Float64)
   check_fcc(S)
   # nearest-neighbour distance
   a = rnn(S)
   # elementary cell
   at = bulk(S)
   E = basis(at)

   # calculate how often we need to repeat it
   # ----------------------------------------
   Fu = defm(at)
   a1, a2, a3 = Fu[:,1], Fu[:,2], Fu[:,3]
   # compute inner radius of the cell in the a1-a2 plane
   a1perp = a1 - dot(a1, a2) / vecnorm(a2)^2
   Ri = 0.5 * vecnorm(a1perp)
   # this means we need to repeat the cell Ri/R times in a1, a2 directions
   #   (add 3 for good measure)
   # and 3 times in the a3 direction to get one each of the ABC planes
   nrep = ceil(Int, R/Ri) + 3
   at = at * (nrep, nrep, 3)

   # compute an origin
   # ------------------
   F = defm(at)
   X = positions(at)
   # project into 2D
   x, y, I0, I1, I2 = pos2d_fcc(at)
   # find the site closest to the centre of the cell
   rsq = (x - mean(x)).^2 + (y - mean(y)).^2
   Ictr = find(rsq .== minimum(rsq))
   xctr = X[Ictr]
   # the origin is shifted from xctr into one of the cells

   #
   # ------------------
end

end
