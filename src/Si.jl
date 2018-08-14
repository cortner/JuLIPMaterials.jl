

# TODO: this is really a module for face-centered diamond-cubic
#       and over time it should become much more general of course

module Si

using JuLIP, QuadGK, ForwardDiff

import JuLIPMaterials

using JuLIPMaterials.CLE: elastic_moduli, voigt_moduli,
         fourth_order_basis, sextic_roots, A_coefficients, D_coefficients,
         little_a

CLE = JuLIPMaterials.CLE
FCC = JuLIPMaterials.FCC
# CauchyBorn = JuLIPMaterials.CauchyBorn

include("CauchyBorn_Si.jl")

"""
`si110_plane(s::AbstractString) -> at::ASEAtoms, b, xcore, a `

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
* `a` : lattice parameter
"""
function si110_plane(s::Symbol;
                     a = defm(bulk(s, cubic=true))[1,1] )
   # TODO: can si110_plane be combined with FCC.fcc_110_plane?
   @assert s == :Si
   # for Si, default a = 5.43
   # TODO: ensure only that s is an FCC species
   # ====================================================
   # construct the cell matrix
   F = a * JMat( [ sqrt(2)/2  0    0;
                         0    1    0;
                         0    0    sqrt(2)/2 ] )
   X = a * [ JVec([         0.0,   0.0,          0.0 ]),
             JVec([ 0.5/sqrt(2),   0.5,  0.5/sqrt(2) ]),
             JVec([         0.0, -0.25,  0.5/sqrt(2) ]),
             JVec([ 0.5/sqrt(2),  0.25,          0.0 ]) ]
   # construct ASEAtoms
   at = Atoms(s, X)
   set_defm!(at, F)
   # compute a burgers vector in these coordinates
   b = a * sqrt(2)/2 * JVec([1.0,0.0,0.0])
   # compute a core-offset (to add to any lattice position)
   xcore = [-.7, 1.0, 0] * a / 5.43
   # return the information
   return at, b, xcore, a
end



"a fully equilibrated SW potential"
function sw_eq()
    T(σ, at) = trace(stress(StillingerWeber(σ=σ), at))
    at = bulk(:Si, pbc=true)
    r0 = 2.09474
    r1 = r0 - 0.1
    s0, s1 = T(r0, at), T(r1, at)
    while (abs(s1) > 1e-8) && abs(r0 - r1) > 1e-8
        rnew = (r0 * s1 - r1 * s0) / (s1 - s0)
        r0, r1 = r1, rnew
        s0, s1 = s1, T(rnew, at)
    end
    return StillingerWeber(σ=r1)
end


function si110_cluster(species, R; kwargs...)
    @assert isodd(R)   # TODO: why?
    atu, b, _, a = si110_plane(species; kwargs...)
    at = atu * (R, R, 1)
    set_pbc!(at, (false, false, true))
    b = b[1]
    X = positions(at)
    # This choice picks the lower left and upper right atom (not site) positions
    #   TODO I don't like the [-1, 0.1, 0] at all
    xcore = (1/2)*(X[length(X)-2]+X[3]) + [-1, 0.1, 0]
    return at, b, xcore
end


"""
a function that identifies multi-lattice structure in 2 layers of bulk-Si
(yes - very restrictive but will do for now!)
"""
function si_multilattice(at; TOL = 0.2)
    J0 = Int[]
    J1 = Int[]
    Jdel = Int[]
    for (i, j, r, R) in sites(at, rnn(:Si)+0.1)
        foundneig = false
        for (jj, RR) in zip(j, R)
            if (abs(RR[1]) <= TOL) && (abs(RR[2] - 1.3575) < TOL)
                # neighbour above >> make (i, jj) a site
                push!(J0, i)
                push!(J1, jj)
                foundneig = true
                break
            elseif (abs(RR[1]) <= TOL) && (abs(RR[2] + 1.3575) < TOL)
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
    W = CauchyBornSi.WcbQuad(calculator(at))   # TODO: generalise this to general calculators
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
    W = CauchyBornSi.WcbQuad(calculator(at))

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

## Keyword Arguments

* a : lattice parameter, default value is the default lattice parameter of the species, other allows values are `:equilibrate`
"""
function edge110(species::Symbol, R::Real;
                  truncate=true, cle=:anisotropic, ν=0.25,
                  calc=sw_eq(), sym = true,
                  TOL=1e-4, zDir=1,
                  eos_correction = true,
                  a = defm(bulk(species, cubic=true))[1,1])

   @assert species == :Si

   # compute the lattice parameter
   if a == :equilibrate
       atu = bulk(species, cubic=true, pbc=true)
       set_calculator!(atu, calc)
       set_constraint!(atu, VariableCell(atu))
       minimise!(atu)
       a = defm(atu)[1,1]
   end

   # setup undeformed geometry
   at, b, x0 = si110_cluster(species, R; a = a)
   set_calculator!(at, calc)


   if cle == :anisotropic
      W = CauchyBornSi.WcbQuad(calc)
      C = elastic_moduli(W)
      Cv = round.(voigt_moduli(C), 8)
      U = CLE.EdgeCubic(b, Cv, a, x0 = x0)
   elseif cle == :isotropic
      @assert sym == nothing
      U = xx -> ([CLE.u_edge_isotropic(xx[1]-x0[1], xx[2]-x0[2], b, ν)...; 0.0], nothing)
   else
      error("unknown `cle` option")
   end

   if sym == true
      @assert cle == :anisotropic
      symml_displacement!(at, U)
   elseif sym == false
      @assert cle == :anisotropic
      ml_displacement!(at, U)
   elseif sym == nothing
      X = positions(at)
      for (i,x) in enumerate(X)
         X[i] += U(x)[1]
      end
      set_positions!(at, X)
   else
      error("edge110: unknown parameters sym = $(sym)")
   end

   return at, x0
end



end
