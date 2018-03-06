
module FCC

using JuLIP

using MaterialsScienceTools.CLE: u_edge_isotropic, u_edge_fcc_110,
      voigt_moduli



const Afcc = JMatF([ 0.0 1 1; 1 0 1; 1 1 0])

"""
ensure that species `sym` actually crystallises to FCC
"""
function check_fcc(sym::Symbol)
   F = defm(bulk(sym))
   @assert vecnorm(F/F[1,2] - Afcc) < 1e-12
end



"""
`fcc_110_plane(sym::Symbol) -> at, b, xcore, a`

Generates an orthorhombic unit cell for an FCC crystal chosen
such that the e1 direction is the burgers vector and the e3 direction the normal
to the standard edge dislocation:
   b = F[:,1] ~ a1;  ν = F[:,3] ~ a2 - a3
The third cell vector: F[:,2] ~ a * e1. Here, ~ means they are rotated from the
a1, a2, a3 directions. This cell contains two atoms.

## Returns

* `at`: Atoms object
* `b` : Burgers vector
* `xcore` : a core-offset (to add to any lattice position)
* `a` : lattice parameter
"""
function fcc_110_plane(sym::Symbol)
   # ensure s is actually an FCC species
   check_fcc(s)
   # get the cubic unit cell dimension
   a = ( bulk(s, cubic=true) |> defm )[1,1]
   # construct the cell matrix
   F = a*JMat( [ sqrt(2)/2 0    0;
                 0   1     0;
                 0   0    sqrt(2)/2 ] )
   X = a*[ JVec([0.0, 0.0, 0.0]),
           JVec([(1/2)*1/sqrt(2), 1/2, 1/(2*sqrt(2))]) ]
   # construct ASEAtoms
   at = ASEAtoms("$(s)2")
   set_defm!(at, F)
   set_positions!(at, X)
   # compute a burgers vector in these coordinates
   b =  a*sqrt(2)/2*JVec([1.0,0.0,0.0])
   # compute a core-offset (to add to any lattice position)
   xcore = a/√2 * JVec([1/4, -sqrt(2)/4, 0])
   # return the information
   return at, b, xcore, a
end


project12(x) = SVec{2}([x[1],x[2]])


"lattice corrector to CLE edge solution"
function xi_solver(Y::Vector, b; TOL = 1e-10, maxnit = 5)
   ξ1(x::Real, y::Real, b) = x - b * angle(x + im * y) / (2*π)
   dξ1(x::Real, y::Real, b) = 1 + b * y / (x^2 + y^2) / (2*π)
    y = Y[2]
    x = y
    for n = 1:maxnit
        f = ξ1(x, y, b) - Y[1]
        if abs(f) <= TOL; break; end
        x = x - f / dξ1(x, y, b)
    end
    if abs(ξ1(x, y, b) - Y[1]) > TOL
        warn("newton solver did not converge; returning input")
        return Y
    end
    return [x, y]
end


function eoscorr(X::Matrix, b)
   Xmod = zeros(X)
   for n = 1:size(X,2)
      Xmod[:, n] = xi_solver(X[:,n], b)
   end
   return Xmod
end


# TODO: rename this edge110 ???

"""
`fcc_edge_geom(s, R; kwargs...) -> at::ASEAtoms`

generates a linear elasticity predictor configuration for
an edge dislocation in FCC, with burgers vector ∝ e₁  and dislocation
core direction ν ∝ e₃

* `s` : chemical symbol
* `R` : domain radius in lattice spacings

**Keyword Arguments:**

* `truncate = true` : truncates to a circular cluster
* `cle = :isotropic` : which CLE model (:isotropic or :anisotropic)
* `ν = 0.25` : Poisson ratio in case `cle == :isotropic`
* `calc=nothing` : calculator is needed if `cle == :anisotropic`
* `TOL=1e-4` : tolerance parameter for evaluating the anisotropic CLE solution; currently ignored
* `zDir=1` : how many layers in z-direction?
* `eos_correction = false` : apply the slip-correction from Ehrlacher, Ortner, Shapeev (Arch. Ration. Mech. Anal. 2016)
"""
function fcc_edge_geom(sym::Symbol, R::Real;
                       truncate=true, cle=:anisotropic, ν=0.25,
                       calc=nothing,
                       TOL=1e-4, zDir=1,
                       eos_correction = true)
   # compute the correct unit cell
   atu, b, xcore, a = fcc_110_plane(sym)
   # multiply the cell to fit a ball of radius a/√2 * R inside
   L1 = ceil(Int, 2*R) + 3
   L2 = ceil(Int, 2*R/√2) + 3
   at = atu * (L1,L2, zDir)
   atp = atu * (L1,L2, zDir)
   # mess with the data
   # turn the Burgers vector into a scalar
   @assert b[2] == b[3] == 0.0
   b = b[1]
   # compute a dislocation core
   xcore = project12(xcore)
   X12 = project12.(positions(at))
   # compute x, y coordinates relative to the core
   x, y = mat(X12)[1,:], mat(X12)[2,:]
   xc, yc = mean(x), mean(y)
   r² = (x-xc).^2 + (y-yc).^2
   tip = minimum(r²) + .0000001
   I0 = find(  tip .> r² .> 0 )[2*zDir]
   xcore = X12[I0] + xcore
   x, y = x - xcore[1], y - xcore[2]

   if eos_correction
      Xmod = eoscorr([x'; y'], -b)
      xmod, ymod = Xmod[1,:][:], Xmod[2,:][:]
   else
      xmod, ymod = x, y
   end

   # compute the dislocation predictor
   if cle == :isotropic
      ux, uy = u_edge_isotropic(xmod, ymod, b, ν)
   elseif cle == :anisotropic
      @assert calc != nothing
      # TODO: this doesn't look clean; maybe we need to pass atu in the future
      # I'm not fully understanding how the function fcc_edge_plane(s) works
      set_pbc!(atu, true)
      atv = bulk(sym, cubic=true) * 4
      Cv = voigt_moduli(calc, atv)
      ux, uy = u_edge_fcc_110(xmod, ymod, b, Cv, a, TOL=TOL)
   else
      error("unknown `cle`")
   end
   # apply the linear elasticity displacement
   X = positions(at) |> mat
   X[1,:], X[2,:] = x + ux + xcore[1], y + uy + xcore[2]
   # if we want a circular cluster, then truncate to an approximate ball (disk)
   if truncate
      F = defm(at) # store F for later use
      X = vecs(X)  # find points within radius
      IR = find( [vecnorm(x[1:2] - xcore) for x in X] .<= R * a/√2 )
      X = X[IR]
      at = Atoms(sym, X)  # generate a new atoms object
      set_defm!(at, F)                 # and insert the old cell shape
   end
   # update positions in Atoms object, set correct BCs and return
   set_positions!(at, X)
   set_pbc!(at, (false, false, true))
   if calc != nothing
      set_calculator!(at, calc)
   end
   return at, xcore
end




end
