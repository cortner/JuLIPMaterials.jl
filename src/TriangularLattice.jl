
module TriangularLattice

using JuLIP

function bulktri(; a0 = 1.0, L = 1)
   X = a0 * [ JVec([0.0, 0.0, 0.0]), JVec([0.5, √3/2, 0.0]) ]
   F = [ 1.0 0.0 0.0; 0.0 √3 0.0; 0.0 0.0 1.0 ]
   at = ASEAtoms("Cu2")
   set_positions!(at, X)
   set_defm!(at, F)

   if L == 1
      set_pbc!(at, (true, true, false))
   else
      Lx = L
      Ly = ceil(Int, L / √3)
      at = at * (Lx, Ly, 1)
   end
   return at
end


function edge_geom(; a0 = 1.0, L = 7, b = 1.0, ν = 0.25,
                     xicorr = true, edgevacancy = true,
                     calc = nothing )
   if calc != nothing
      a0 = get_a0(calc)
   end
   # create a homogeneous lattice
   at = bulktri(a0=a0, L=L)
   X = positions(at)
   # find center-atom
   F = defm(at)
   x0 = JVec( [0.5 * diag(F)[1:2]; 0.0] )
   _ , I0 = findmin( [norm(x - x0) for x in X] )
   # dislocation core
   tcore = a0 * JVec([0.5, √3/6, 0.0])
   xc = X[I0] + tcore
   # shift configuration to move core to 0
   X = [x - xc for x in positions(at)]
   set_positions!(at, X)
   # remove the center-atom
   if edgevacancy
      deleteat!(at, I0)
   end
   # apply dislocation FF predictor
   edge_predictor!(at; b=b, xicorr=xicorr, ν=ν)
   if edgevacancy
      X = [x + tcore for x in positions(at)]
      set_positions!(at, X)
   end
   # TODO: compute ν properly
   return at
end

function edge_cluster(R::Number; kwargs...)
   at = edge_geom(;L = ceil(Int, R) * 2 + 3, kwargs...)
   X = positions(at)
   X = X[find(norm.(X) .<= R)]
   cl = ASEAtoms("Al$(length(X))")
   set_positions!(cl, X)
   set_defm!(cl, defm(at))
   set_pbc!(cl, (false, false, false))
   return cl
end


"""
standard isotropic CLE edge dislocation solution
"""
function ulin_edge_isotropic(X, b, ν)
    x, y = X[1,:], X[2,:]
    r² = x.^2 + y.^2
    ux = b/(2*π) * ( angle(x + im*y) + (x .* y) ./ (2*(1-ν) * r²) )
    uy = -b/(2*π) * ( (1-2*ν)/(4*(1-ν)) * log(r²) + - 2 * y.^2 ./ (4*(1-ν) * r²) )
    return [ux'; uy']
end



"""
lattice corrector to CLE edge solution; cf EOS paper
"""
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
        warn("newton solver did not converge at Y = $Y; returning input")
        return Y
    end
    return [x, y]
end

"""
EOSShapeev edge dislocation solution
"""
function ulin_edge_eos(X, b, ν)
    Xmod = zeros(2, size(X, 2))
    for n = 1:size(X,2)
        Xmod[:, n] = xi_solver(X[1:2,n], b)
    end
    return ulin_edge_isotropic(Xmod, b, ν)
end

function edge_predictor!(at::AbstractAtoms; b = 1.0, xicorr = true, ν = 0.25)
   X = positions(at) |> mat
   if xicorr
      X[1:2,:] += ulin_edge_eos(X, b, ν)
   else
      X[1:2,:] += ulin_edge_isotropic(X, b, ν)
   end
   set_positions!(at, X)
   return at
end


function get_a0(calc::AbstractCalculator)
   mystress(a) = trace(stress(calc, bulktri(a0 = a)))
   a0, a1 = 1.0, 1.02
   s0, s1 = mystress(a0), mystress(a1)
   while abs(a1 - a0) > 1e-8 && abs(s1) > 1e-8
      anew = (a0 * s1 - a1 * s0) / (s1 - s0)
      a0, a1 = a1, anew
      s0, s1 = s1, mystress(a1)
   end
   return a1
end

end
