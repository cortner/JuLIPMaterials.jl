
module GreensFunctions

using StaticArrays

typealias Vec3{T} SVector{3, T}

# extend `angle` to avoid going via C
Base.angle(x, y) = Base.angle(x + im * y)   # atan2(y, x)

"convert normalised vector to spherical coordinates"
spherical(x) = angle(x[1], x[2]), angle(norm(x[1:2]), x[3])

"convert spherical to euclidean coordinates on the unit sphere"
euclidean(φ, ψ) = Vec3(cos(ψ) * cos(φ), cos(ψ) * sin(φ), sin(ψ))

"given a vector x ∈ R³, return `(z0, z1)` where `(x/norm(x),z0,z1)` form an ONB."
function onb{T}(x::Vec3{T})
   x /= norm(x)
   φ, ψ = spherical(x)
   return Vec3{T}(-sin(φ), cos(φ), 0.0),
          Vec3{T}(sin(ψ)*cos(φ), sin(ψ)*sin(φ), -cos(ψ))
end

"explicit inverse to enable AD, if A is an `SMatrix` then the conversion is free"
inv3x3(A) = inv( SMatrix{3,3}(A) )


"""
* `GreenTensor3D(x,ℂ,quadpts=10) -> G, DG`
Computes the full 3D Green tensor, G, and its gradient, DG, at the point x for elasticity
tensor ℂ ∈ ℝ³ˣ³ˣ³ˣ³, based on using quadpts quadrature points on a circle.
"""
function GreenTensor3D{T}(x::AbstractVector{T}, ℂ, quadpts=10)
    # Some basic error handling which should be redone with types.
    if ~( size(x)==(3,) ) && ~( size(x)==(3,1) )
        error("Input is not a 3D column vector")
    elseif ~( size(ℂ)==(3,3,3,3) )
        error("Elasticity tensor incorrect size")
    end

    # Initialise tensors.
    G = zeros(T,3,3)
    DG = zeros(T,3,3,3)
    x̂ = x/norm(x)
    F = zeros(T,3,3)

    # basis is a 3x2 matrix of vectors orthogonal to x.
    basis = hcat(onb(Vec3(x))...)

    # Integrate
    for m=0:quadpts-1
        zz = zeros(T,3,3)
        zT = zeros(T,3,3)
        z = basis*[cos(pi*m/quadpts), sin(pi*m/quadpts)]
        for i=1:3, j=1:3
            zz[i,j] = dot(z, ℂ[i,:,j,:] * z)
            zT[i,j] = dot(z, ℂ[i,:,j,:] * x̂)
        end
        zzinv = inv3x3(zz)

        # Perform integration
        G += zzinv
        F = (zz\(zT+zT'))/zz
        for i=1:3, j=1:3, k=1:3
            DG[i,j,k] += zzinv[i,j]*x̂[k]-F[i,j]*z[k]
        end
    end

    # Normalise appropriately
    G = G/(4*pi*norm(x)*quadpts)
    DG = -DG/(4*pi*norm(x)^2*quadpts)

    return G, DG
end

"""
* `IsoGreenTensor3D(x,μ::Float64,λ::Float64) -> G, DG`
Returns the 3D Isotropic Green tensor and gradient at the point x, via exact formula
with Lamé parameters μ and λ.
"""
function IsoGreenTensor3D(x,μ::Float64,λ::Float64)
    # Error handling which should be redone with types.
    if ~( size(x)==(3,) ) && ~( size(x)==(3,1) )
        error("Input is not a 3D column vector");
    end

    # Construct Green tensor
    G = ((λ+3*μ)/(λ+2*μ)*eye(3)/norm(x)+(λ+μ)/(λ+2*μ)*x*x'/norm(x)^3)/(8*pi*μ);

    # Construct gradient of Green tensor
    DG = zeros(Float64,3,3,3);
    Id = eye(3);
    for i=1:3, j=1:3, k=1:3
            DG[i,j,k] = ( (λ+μ)*(Id[i,k]*x[j]+Id[j,k]*x[i]) - (λ+3*μ)*eye(3)[i,j]*x[k] ) -
                            3*(λ+μ)*x[i]*x[j]*x[k]/(norm(x)^2);
    end
    DG = DG/(8*pi*μ*(λ+2*μ)*norm(x)^3);

    return G, DG
end

end
