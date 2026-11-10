# Log densities

Implement the callback that NUTS calls at each leapfrog position, then check its
sign and scaling before running a chain.

## Callback

The target is an object with a method for
[`NUTS.log_density_gradient!`](@ref). Its position and gradient are vectors of
unconstrained real coordinates. Return the scalar log density and overwrite
every element of the supplied gradient. An additive constant in the log density
does not affect the transition.

```@example target
using NUTS, LinearAlgebra

struct GaussianTarget
    center::Vector{Float64}
    scale::Vector{Float64}
end

function NUTS.log_density_gradient!(target::GaussianTarget, q, gradient)
    gradient .= -(q .- target.center) ./ target.scale.^2
    return -sum(abs2, (q .- target.center) ./ target.scale) / 2
end

target = GaussianTarget([1.0, -1.0], [0.5, 2.0])
q = [1.5, 1.0]
gradient = similar(q)
logdensity = NUTS.log_density_gradient!(target, q, gradient)
(; logdensity, gradient)
```

The gradient points back toward the center. Each squared scale divides the
corresponding displacement, so the narrower coordinate has greater curvature.
The target's data are read-only; only the gradient destination is overwritten.

## Gradient check

Compare the callback with central differences at a position away from the
center. Checking only the center would miss a sign error because both gradients
vanish there.

```@example target
logdensity_at(x) = NUTS.log_density_gradient!(target, x, similar(x))
h = cbrt(eps(Float64))
finite_difference = map(eachindex(q)) do j
    left, right = copy(q), copy(q)
    left[j] -= h
    right[j] += h
    (logdensity_at(right) - logdensity_at(left)) / (2h)
end
matches = isapprox(gradient, finite_difference; rtol=1e-7, atol=1e-9)
@assert matches
(; analytic=gradient, finite_difference, matches)
```

This checks the callback, not convergence of a Markov chain. For a model with
several terms, also check positions where each term contributes and verify the
scalar log density against a separate calculation.

## Coordinates

NUTS integrates in the coordinates supplied in `position`. It does not transform
bounded parameters. When mapping an unconstrained coordinate to a constrained
parameter, include the log absolute Jacobian determinant in the log density and
differentiate the resulting expression. The gradient must use the same
coordinates as the position vector.

The callback supplies the derivative directly; NUTS does not select an automatic
differentiation backend. A callback may call a derivative implementation from
another package, provided it fills the supplied buffer and returns the scalar
log density. The current implementation constructs its internal phase-space
buffers with `Float64` elements; the examples therefore use `Vector{Float64}`.

Keep the target fixed while reusing a sampler state. It contains a cached log
density and gradient. A changed target needs a new initial state so that those
values are recomputed.

Use this callback in the [Gaussian example](../examples/gaussian.md).
