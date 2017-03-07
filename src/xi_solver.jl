

ξ1(x::Real, y::Real) = x - angle(x + im * y) / (2*π)
dξ1(x::Real, y::Real) = 1 + y / (x^2 + y^2) / (2*π)

function xi_solver(Y::Vector; TOL = 1e-10, maxnit = 5)
    y = Y[2]
    x = y
    for n = 1:maxnit
        f = ξ1(x, y) - Y[1]
        if abs(f) <= TOL; break; end
        x = x - f / dξ1(x, y)
    end
    if abs(ξ1(x, y) - Y[1]) > TOL
        warn("newton solver did not converge; returning input")
        return Y
    end
    return [x, y]
end

# test code for xi_solver
# ξ(X) = [ξ1(X[1], X[2]), X[2]]
#
# for n = 1:10
#     Y = 10.0 * [1.0 + rand(), 1.0 + rand()]
#     X = xi_solver(Y)
#     @show norm(ξ(X) - Y)
# end
