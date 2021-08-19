"""
Near field solution for a crack in a rectilinear anisotropic elastic medium.
See:
G. C. Sih, P. C. Paris and G. R. Irwin, Int. J. Frac. Mech. 1, 189 (1965)

Ported from `matscipy.fracture_mechanics.crack` module by James Kermode
"""
module Fracture

using LinearAlgebra
using Polynomials: Polynomial, roots
using Einsum
using JuLIP
using JuLIPMaterials.CLE

export RectilinearAnisotropicCrack, PlaneStrain, PlaneStress
export displacements, stresses, deformation_gradient, k1g, u_CLE

abstract type Crack end

abstract type StressState end
struct PlaneStress <: StressState end
struct PlaneStrain <: StressState end

abstract type CoordinateSystem end

struct Cartesian <: CoordinateSystem
    x::AbstractVector
    y::AbstractVector
end

struct Cylindrical <: CoordinateSystem
    r::AbstractVector
    θ::AbstractVector
end

Cylindrical(cart::Cartesian) = Cylindrical(sqrt.(cart.x.^2 + cart.y.^2), atan.(cart.y, cart.x))

Cartesian(cyl::Cylindrical) = Cartesian(cyl.r * cos.(cyl.θ), cyl.r * sin.(cyl.θ))

struct RectilinearAnisotropicCrack{T <: Number} <: Crack
    a22::T
    μ1::Complex{T}
    μ2::Complex{T}
    p1::Complex{T}
    p2::Complex{T}
    q1::Complex{T}
    q2::Complex{T}
    inv_μ1_μ2::Complex{T}
    μ1_p2::Complex{T}
    μ2_p1::Complex{T}
    μ1_q2::Complex{T}
    μ2_q1::Complex{T}
end

function RectilinearAnisotropicCrack(a22::T, μ1::Complex{T}, μ2::Complex{T}, p1::Complex{T}, p2::Complex{T}, 
                                     q1::Complex{T}, q2::Complex{T}) where {T}
    inv_μ1_μ2 = 1/(μ1 - μ2)
    μ1_p2 = μ1 * p2
    μ2_p1 = μ2 * p1
    μ1_q2 = μ1 * q2
    μ2_q1 = μ2 * q1
    return RectilinearAnisotropicCrack(a22, μ1, μ2, p1, p2, q1, q2, inv_μ1_μ2, 
                                       μ1_p2, μ2_p1, μ1_q2, μ2_q1)
end

function RectilinearAnisotropicCrack(a11::T, a22::T, a12::T, a16::T, a26::T, a66::T) where {T}
    p = Polynomial(reverse([ a11, -2*a16, 2*a12 + a66, -2*a26, a22 ] )) # NB: opposite order to np.poly1d()
    μ1, μ1s, μ2, μ2s = roots(p)
    (μ1 != conj(μ1s) ||  μ2 != conj(μ2s)) && error("Roots not in pairs.")

    p1 = a11 * μ1^2 + a12 - a16 * μ1
    p2 = a11 * μ2^2 + a12 - a16 * μ2

    q1 = a12 * μ1 + a22 / μ1 - a26
    q2 = a12 * μ2 + a22 / μ2 - a26

    return RectilinearAnisotropicCrack(a22, μ1, μ2, p1, p2, q1, q2)
end

RectilinearAnisotropicCrack(::PlaneStress, S::AbstractMatrix) = RectilinearAnisotropicCrack(S[1, 1], S[2, 2], S[1, 2], S[1, 6], S[2, 6], S[6, 6])

function RectilinearAnisotropicCrack(::PlaneStrain, S::AbstractMatrix)
    b11, b22, b33, b12, b13, b23, b16, b26, b36, b66 = (S[1, 1], S[2, 2], S[3, 3], S[1, 2], S[1, 3], 
                                                        S[2, 3], S[1, 6], S[2, 6], S[3, 6], S[6, 6])
    a11 = b11 - (b13 * b13) / b33
    a22 = b22 - (b23 * b23) / b33
    a12 = b12 - (b13 * b23) / b33
    a16 = b16 - (b13 * b36) / b33
    a26 = b26 - (b23 * b36) / b33
    a66 = b66
    return RectilinearAnisotropicCrack(a11, a22, a12, a16, a26, a66)
end

function RectilinearAnisotropicCrack(ss::StressState, C::AbstractMatrix, crack_surface::AbstractVector, crack_front::AbstractVector)
    A = rotation_matrix(y=crack_surface, z=crack_front)
    Crot = rotate_moduli(C, A)
    S = inv(Crot)
    return RectilinearAnisotropicCrack(ss, S)
end

RectilinearAnisotropicCrack(ss::StressState, 
                            C11, C12, C44, 
                            crack_surface::AbstractVector, crack_front::AbstractVector) = 
                                RectilinearAnisotropicCrack(ss, voigt_moduli(C11, C12, C44), crack_surface, crack_front)

"""
Displacement field in mode I fracture. 

Parameters
----------
r : array
    Distances from the crack tip.
theta : array
    Angles with respect to the plane of the crack.

Returns
-------
u : array
    Displacements parallel to the plane of the crack.
v : array
    Displacements normal to the plane of the crack.
"""
function displacements(crack::RectilinearAnisotropicCrack, cyl::Cylindrical)
    h1 = sqrt.(2.0 * cyl.r / π)
    h2 = sqrt.( cos.(cyl.θ) .+ crack.μ2 * sin.(cyl.θ) )
    h3 = sqrt.( cos.(cyl.θ) .+ crack.μ1 * sin.(cyl.θ) )

    u = h1 .* real.( crack.inv_μ1_μ2 * ( crack.μ1_p2 * h2 - crack.μ2_p1 * h3 ) )
    v = h1 .* real.( crack.inv_μ1_μ2 * ( crack.μ1_q2 * h2 - crack.μ2_q1 * h3 ) )

    return u, v
end

"""
Deformation gradient tensor in mode I fracture.

Parameters
----------
r : array_like
    Distances from the crack tip.
theta : array_like
    Angles with respect to the plane of the crack.

Returns a Jacobian ```[du_dx du_dy
                       dv_dx dv_dz]```

where

* `du_dx` - Derivatives of displacements parallel to the plane within the plane.
* `du_dy`- Derivatives of displacements parallel to the plane perpendicular to the plane.
* `dv_dx`- Derivatives of displacements normal to the plane of the crack within the plane.
* `dv_dy` - Derivatives of displacements normal to the plane of the crack perpendicular to the plane.
"""
function deformation_gradient(crack::RectilinearAnisotropicCrack, cyl::Cylindrical)
    f = 1 ./ sqrt.(2 * π * cyl.r)

    h1 = (crack.μ1 * crack.μ2) * crack.inv_μ1_μ2
    h2 = sqrt.( cos.(cyl.θ) + crack.μ2 * sin.(cyl.θ) )
    h3 = sqrt.( cos.(cyl.θ) + crack.μ1 * sin.(cyl.θ) )

    du_dx = f .* real.( crack.inv_μ1_μ2 * ( crack.μ1_p2 ./ h2 - crack.μ2_p1 ./ h3 ) )
    du_dy = f .* real.( h1 * ( crack.p2 ./ h2 - crack.p1 ./ h3 ) )

    dv_dx = f .* real.( crack.inv_μ1_μ2 * ( crack.μ1_q2 ./ h2 - crack.μ2_q1 ./ h3 ) )
    dv_dy = f .* real.( h1 * ( crack.q2 ./ h2 - crack.q1 ./ h3 ) )

    return [[du_dx] [du_dy]
            [dv_dx] [dv_dy]]
end

"""
Stress field in mode I fracture.

Parameters
----------
r : array
    Distances from the crack tip.
theta : array
    Angles with respect to the plane of the crack.

Returns
-------
sig_x : array
    Diagonal component of stress tensor parallel to the plane of the
    crack.
sig_y : array
    Diagonal component of stress tensor normal to the plane of the
    crack.
sig_xy : array
    Off-diagonal component of the stress tensor.
"""
function stresses(crack::RectilinearAnisotropicCrack, cyl::Cylindrical)
    f = 1 ./ sqrt.(2.0 * π * cyl.r)

    h1 = (crack.μ1 * crack.μ2) * crack.inv_μ1_μ2
    h2 = sqrt.( cos.(cyl.θ) + crack.μ2 * sin.(cyl.θ) )
    h3 = sqrt.( cos.(cyl.θ) + crack.μ1 * sin.(cyl.θ) )

    sig_x  = f * real.(h1 * (crack.μ2 ./ h2 - crack.μ1 ./ h3))
    sig_y  = f * real.(crack.inv_μ1_μ2 * (crack.μ1 ./ h2 - crack.μ2 ./ h3))
    sig_xy = f * real.(h1 * (1 ./ h3 - 1 ./ h2))

    return sig_x, sig_y, sig_xy
end

"""
K1G, Griffith critical stress intensity in mode I fracture
"""
function k1g(crack::RectilinearAnisotropicCrack, surface_energy)
    return sqrt(abs(4 * surface_energy / 
                     imag(crack.a22 * ((crack.μ1 + crack.μ2) / (crack.μ1  * crack.μ2 )))))
end

# Cartesian coordinate convenience wrappers

displacements(crack::RectilinearAnisotropicCrack, cart::Cartesian) = displacements(crack, Cylindrical(cart)) 
displacements(crack::RectilinearAnisotropicCrack, x::AbstractVector, y::AbstractVector) = displacements(crack, Cartesian(x, y)) 

deformation_gradient(crack::RectilinearAnisotropicCrack, cart::Cartesian) = deformation_gradient(crack, Cylindrical(cart))
deformation_gradient(crack::RectilinearAnisotropicCrack, x::AbstractVector, y::AbstractVector) = deformation_gradient(crack, Cartesian(x, y))

stresses(crack::RectilinearAnisotropicCrack, cart::Cartesian) = stresses(crack, Cylindrical(cart)) 
stresses(crack::RectilinearAnisotropicCrack, x::AbstractVector, y::AbstractVector) = stresses(crack, Cartesian(x, y)) 

function u_CLE(crack::RectilinearAnisotropicCrack, x::AbstractVector, y::AbstractVector)
    function u(k, α) 
        ux, uy = displacements(crack, x .- α, y)
        return k * [ux uy]'
    end

    function du(k, α)
        D = deformation_gradient(crack, x .- α, y)
        return  -k * [D[1,1] D[2,1]]'
    end

    return u, du
end

function u_CLE(crack::RectilinearAnisotropicCrack, crystal::AbstractAtoms{T}, x0::T, y0::T) where {T}
    X = positions(crystal) |> mat
    return u_CLE(crack, X[1, :] .- x0, X[2, :] .- y0)
end 

end