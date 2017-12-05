
# TODO: this is really a module for face-centered diamond-cubic
#       and over time it should become much more general of course

module Si

using JuLIP, JuLIP.ASE, JuLIP.Potentials, QuadGK, ForwardDiff

import MaterialsScienceTools

using MaterialsScienceTools.CLE: elastic_moduli, voigt_moduli,
         fourth_order_basis, sextic_roots, A_coefficients, D_coefficients,
         little_a

CLE = MaterialsScienceTools.CLE
FCC = MaterialsScienceTools.FCC
CauchyBorn = MaterialsScienceTools.CauchyBorn


"""
`si110_plane(s::AbstractString) -> at::ASEAtoms, b, xcore `

Generates a unit cell for an FCC crystal with orthogonal cell vectors chosen
such that the F1 direction is the burgers vector and the F3 direction the normal
to the standard edge dislocation:
   b = F1 ~ a1;  ν = F3 ~ a2-a3
The third cell vector: F2 ~ a * e1. Here, ~ means they are rotated from the
a1, a2, a3 directions. This cell contains two atoms.

Returns
* `at`: Atoms object
* `b` : Burgers vector
* `xcore` : a core-offset (to add to any lattice position)
"""
function si110_plane(s::AbstractString)
   # TODO: can si110_plane be combined with FCC.fcc_110_plane?
   @assert s == "Si"
   # ensure s is actually an FCC species
   # check_fcc(s)
   # get the cubic unit cell dimension
   a = ( bulk(s, cubic=true) |> defm )[1,1]
   #print(a)
   # construct the cell matrix
   F = a*JMat( [ sqrt(2)/2 0    0;
                 0   1     0;
                 0   0    sqrt(2)/2 ] )
   X = a*[ JVec([0.0, 0.0, 0.0]),
         JVec([(1/2)*1/sqrt(2),1/2, 1/(2*sqrt(2))]), JVec([0, -1/4, 1/(2*sqrt(2))]),JVec([(1/2)*1/sqrt(2), 1/4, 0]) ]
   # construct ASEAtoms
   at = ASEAtoms(string(s,"4"))
   set_defm!(at, F)
   set_positions!(at, X)
   # compute a burgers vector in these coordinates
   b =  a*sqrt(2)/2*JVec([1.0,0.0,0.0])
   # compute a core-offset (to add to any lattice position)
   xcore = [-.7, 1.0, 0] # a*sqrt(2)/2 * JVec([1/2, -1/3, 0])  # [1/2, 1/3, 0]
   # return the information
   return at, b, xcore, a
end



"a fully equilibrated SW potential"
function sw_eq()
    T(σ, at) = trace(stress(StillingerWeber(σ=σ), at))
    at = JuLIP.ASE.bulk("Si", pbc=true)
    r0 = 2.09474
    r1 = r0 - 0.1
    s0, s1 = T(r0, at), T(r1, at)
    while (abs(s1) > 1e-8) && abs(r0 - r1) > 1e-8
        rnew = (r0 * s1 - r1 * s0) / (s1 - s0)
        r0, r1 = r1, rnew
        s0, s1 = s1, T(rnew, at)
    end
#     @show r1
    return StillingerWeber(σ=r1)
end


function si110_cluster(species, R)
    @assert isodd(R)   # TODO: why?
    atu, b, _, a = si110_plane(species)
    at = atu * (R, R, 1)
    set_pbc!(at, (false, false, true))
    b = b[1]
    X = positions(at)
    # This choice picks the lower left and upper right atom (not site) positions
    #   TODO I don't like the 0.1 at all
    xcore = (1/2)*(X[length(X)-2]+X[3]) +[-1, 0.1, 0]
    return at, b, xcore
end


"""
a function that identifies multi-lattice structure in 2 layers of bulk-Si
(yes - very restrictive but will do for now!)
"""
function si_multilattice(at)
    J0 = Int[]
    J1 = Int[]
    Jdel = Int[]
    for (i, j, r, R, _) in sites(at, rnn("Si")+0.1)
        foundneig = false
        for (jj, RR) in zip(j, R)
            if (RR[1] == 0.0) && (abs(RR[2] - 1.3575) < 1e-3)
                # neighbour above >> make (i, jj) a site
                push!(J0, i)
                push!(J1, jj)
                foundneig = true
                break
            elseif (RR[1] == 0.0) && (abs(RR[2] + 1.3575) < 1e-3)
                # neighbour below >> (jj, i) is a site that will be pushed when i ↔ jj
                foundneig = true
                break
            end
        end
        if !foundneig
            # i has no neighbour above or below >> probably we just get rid of it
            push!(Jdel, i)
        end
    end
    return J0, J1, Jdel
end



function symml_displacement!(at, u)
    I0, I1, Idel = si_multilattice(at)
    @assert isempty(Idel)  # if Idel is not empty then (for now) we don't know what to do
    X = positions(at)
    W = CauchyBorn.WcbQuad()   # TODO: generalise this to general calculators
    F0 = defm(W.at)
    p0 = W(F0)
    # transformation matrices
    Tp = [0 1/√2  -1/√2; 1 0 0; 0 1/√2 1/√2]
    Tm = diagm([1,1,-1]) * Tp

    for (i0, i1) in zip(I0, I1)   # each pair (i0, i1) corresponds to a ML lattice site
        x0, x1 = X[i0], X[i1]
        x1[3] > x0[3] ? T = Tp : T = Tm
        x̄ = 0.5 * (x0 + x1)   # centre of mass of the bond
        U, ∇U = u(x̄)          # displacement and displacement gradient
        F = T' * (I + ∇U) * T
        q = T * (W(F * F0) - p0)    # construct the shift corresponding to F = Id + ∇U
        X[i0], X[i1] = x0 + U - 0.5 * q, x1 + U + 0.5 * q
    end
    set_positions!(at, X)
    return at
end

function ml_displacement!(at, u)
    I0, I1, Idel = si_multilattice(at)
    @assert isempty(Idel)  # if Idel is not empty then (for now) we don't know what to do
    X = positions(at)
    W = CauchyBorn.WcbQuad()

    # transformation matrices
    Tp = [0 1/√2  -1/√2; 1 0 0; 0 1/√2 1/√2]
    Tm = diagm([1,1,-1]) * Tp

    F0 = defm(W.at)  # get reference information
    p0 = W(F0)

    for (i0, i1) in zip(I0, I1)   # each pair (i0, i1) corresponds to a ML lattice site
        x0, x1 = X[i0], X[i1]
        x1[3] > x0[3] ? T = Tp : T = Tm
        U, ∇U = u(x0)            # displacement and displacement gradient
        F = T' * (I + ∇U) * T
        q = T * (W(F * F0) - p0)   # construct the shift corresponding to F = Id + ∇U
        X[i0], X[i1] = x0 + U, x1 + U + q
    end
    set_positions!(at, X)
    return at
end




"""
`edge110` : produces a high-quality CLE solution for an edge dislocation
in bulk silicon, 110 orientation.
"""
function edge110(species::AbstractString, R::Real;
                  truncate=true, cle=:anisotropic, ν=0.25,
                  calc=sw_eq(), sym = true,
                  TOL=1e-4, zDir=1,
                  eos_correction = true)

   @assert species == "Si"
   # setup undeformed geometry
   at, b, x0 = si110_cluster(species, R)
   a = cell(bulk(species, cubic=true))[1,1]   # lattice parameter

   W = CauchyBorn.WcbQuad()
   C = elastic_moduli(W)
   Cv = round.(voigt_moduli(C), 8)

   if cle != :anisotropic
      error("unknown `cle` option")
   end

   # edge solution # TODO: construct this from a unit cell+calculator???
   U = CLE.EdgeCubic(b, Cv, a, x0 = x0)

   if sym
      symml_displacement!(at, U)
   else
      ml_displacement!(at, U)
   end

   set_calculator!(at, calc)

   return at, x0
end



end
