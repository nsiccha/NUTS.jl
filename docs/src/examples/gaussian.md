# Gaussian coordinates

Sample a Gaussian target whose coordinates have different scales, and compare
its sample quantiles with the distribution specified by the target.

## Target

There are no observations or fitted parameters in this example. The target is
a product of Gaussian densities, specified by its center and coordinate scales.
Its known mean and variance provide a reference for inspecting the sampler's
output. This is a teaching calculation executed during the documentation build;
it does not run an inference fit, a performance comparison or a cached analysis.

Define the log density and its gradient in the same coordinates as the position.
The normalization constant is omitted because its derivative is zero and it
cancels from the transition probabilities. The fields printed after the code
are the actual target parameters used below.

```@example gaussian
using NUTS, Random, LinearAlgebra, Statistics

struct GaussianTarget
    center::Vector{Float64}
    scale::Vector{Float64}
end

function NUTS.log_density_gradient!(target::GaussianTarget, q, gradient)
    gradient .= -(q .- target.center) ./ target.scale.^2
    return -sum(abs2, (q .- target.center) ./ target.scale) / 2
end

target = GaussianTarget([1.0, -1.0], [0.5, 2.0])
(; center=target.center, scale=target.scale)
```

Each component contributes a quadratic term to the log density. Dividing a
displacement by its squared scale gives the corresponding gradient magnitude.
The [log-density guide](../guides/targets.md) checks this callback by finite
differences away from the center, where a sign or scale error can be detected.
The same protocol applies to a posterior log density: the sampler only needs
the scalar value and the gradient at its current position.

## Configuration

Use the known diagonal covariance as the sampler's `squared_scale`, together
with its diagonal factor as `scale`. This changes the integration geometry; it
does not change the target density. The multiplication and solve conventions
are described in the [state guide](../guides/state.md).

```@example gaussian
seed = 42
initial = (rng=Xoshiro(seed), posterior=target,
           position=copy(target.center), stepsize=0.6,
           scale=Diagonal(target.scale),
           squared_scale=Diagonal(target.scale.^2))
state = initial
(; initial_stepsize=state.stepsize,
   covariance=diag(state.squared_scale),
   position=copy(state.position))
```

The initial position is the center, rather than a draw from the stationary
distribution. A preliminary segment is discarded below. That segment does not
adapt the step size or covariance: both remain exactly the objects supplied in
the initial state. For an unknown posterior, an adaptation method must supply
those choices separately. A satisfactory result for this Gaussian does not
justify using its step size for a different target.

## Draws

Retain each position in a separate column. The sampler reuses its internal
arrays, so keeping references to the returned position would lose earlier
draws. The assignment from `nuts!!` is also necessary: it retains the initialized
work storage and the diagnostics from each transition.

```@example gaussian
discard = 200
n_draws = 2_000
state, draws, acceptance, leapfrogs, divergences = let state=state
    for _ in 1:discard
        state = nuts!!(state)
    end
    draws = Matrix{Float64}(undef, length(target.center), n_draws)
    acceptance = Vector{Float64}(undef, n_draws)
    leapfrogs = Vector{Int}(undef, n_draws)
    divergences = falses(n_draws)
    for i in 1:n_draws
        state = nuts!!(state)
        draws[:, i] .= state.position
        acceptance[i] = state.accept_prob
        leapfrogs[i] = state.n_leapfrog
        divergences[i] = state.divergent
    end
    (state, draws, acceptance, leapfrogs, divergences)
end
@assert all(isfinite, draws)
(; coordinates=size(draws, 1), retained=size(draws, 2), discarded=discard)
```

The matrix layout makes the storage convention explicit: rows correspond to
coordinates and columns to successive transitions. A retained draw can equal
its predecessor, and neighboring draws are generally dependent. The displayed
count therefore describes stored transitions rather than an effective sample
size. Increasing the count cannot repair an incorrect gradient or an unsuitable
integration configuration.

## Diagnostics

Inspect the transition diagnostics together with the sample moments. The build
checks that positions are finite and that the sample means and standard
deviations fall within a tolerance of the known target, expressed in units of
each coordinate's scale. The tolerance is a regression check for this seeded
teaching example; it is not a general convergence criterion.

```@example gaussian
estimated_mean = vec(mean(draws; dims=2))
estimated_scale = vec(std(draws; dims=2))
mean_error = (estimated_mean .- target.center) ./ target.scale
scale_error = estimated_scale ./ target.scale .- 1
@assert maximum(abs, mean_error) < 0.2
@assert maximum(abs, scale_error) < 0.2
@assert !any(divergences)
(; estimated_mean=round.(estimated_mean; digits=3),
   estimated_scale=round.(estimated_scale; digits=3),
   divergent_transitions=count(divergences),
   mean_acceptance=round(mean(acceptance); digits=3),
   mean_leapfrogs=round(mean(leapfrogs); digits=2))
```

The acceptance statistic averages integration acceptance probabilities; it is
not the fraction of distinct positions. A divergence indicates that the
integrator crossed the configured energy-error threshold. Absence of divergences
does not establish exploration of every region of a posterior. This example
uses one chain and does not report a between-chain convergence diagnostic or
claim independent samples.

## Marginals

Standardize each coordinate using the specified target, not its estimated
sample moments. Agreement with standard-normal quantiles then checks both the
location and spread of the retained draws. Standardizing by the estimated mean
and variance would force those features to match and weaken the comparison.

```@example gaussian
standardized = (draws .- target.center) ./ target.scale
using AlgebraOfVega, Distributions
probabilities = range(0.01, 0.99; length=99)
reference = quantile.(Normal(), probabilities)
rows = [(reference=reference[k],
         sample=quantile(standardized[j, :], probabilities[k]),
         coordinate="Coordinate $j", series="NUTS draws")
        for j in axes(standardized, 1) for k in eachindex(reference)]
identity_rows = [(reference=x, sample=x, coordinate="Coordinate $j",
                  series="Normal target")
                 for j in axes(standardized, 1) for x in reference]
axes_mapping = mapping(:reference => "Normal quantile (SD units)",
                       :sample => "Sample quantile (SD units)";
                       col=:coordinate, color=:series)
figure = (data(rows) * axes_mapping * visual(Scatter) +
          data(identity_rows) * axes_mapping * visual(Lines)) *
         config(width=230, height=230,
                scales=scales(Color=(categories=["NUTS draws", "Normal target"],
                                     palette=["#2a78d6", "#eb6834"])))
include(joinpath(@__DIR__, "..", "..", "figure.jl")) # hide
show_figure(figure, n_draws)
```

Each blue point pairs a sample quantile with the corresponding normal quantile;
the orange line marks equality. A shift moves points away from the line's
intercept, while a scale mismatch changes their slope. Curvature can indicate
a difference in distribution shape. Because the coordinates use the specified
center and scale, these comparisons do not normalize away the discrepancies
being examined. The plotted probabilities omit the most extreme observations,
so the figure is not a check of arbitrarily remote tails.

Quantiles summarize the marginal distributions, not the order in which
the chain visited positions. Their estimates fluctuate with the finite sample,
and agreement here does not test tail behavior of a different distribution.
For a model without known marginals, inspect appropriate posterior quantities,
multiple independently initialized chains and diagnostics suited to that model.

The target in this example has independent coordinates. Checking its separate
marginals therefore addresses the features that were specified in the model,
but the same pair of plots would not verify dependence in a correlated target.
For such a model, compare joint structure as well as coordinate summaries.
A transformation that mixes coordinates also requires a matching covariance
factor; copying only the diagonal scales discards that information. Keep the
log-density callback, the position coordinates and the supplied factor in the
same parameterization throughout a chain.

## Reproduce

The receipt records the inputs, runtime and source content that produced the
output. The small synthetic run and both themed figures are regenerated by the
build. Neither posterior draws nor rendered figures are committed to Git.

```@raw html
<details><summary>Commands and build receipt</summary>
```

Run these shell commands from the repository root; they prepare the docs
environment rather than running inside a documentation code block.

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

```@example gaussian
using SHA
(; julia=string(VERSION), nuts=string(pkgversion(NUTS)),
   source_sha256=bytes2hex(sha256(read(pathof(NUTS)))),
   seed, discard, n_draws, stepsize=initial.stepsize,
   center=target.center, scale=target.scale)
```

The [historical timing experiment](https://github.com/nsiccha/NUTS.jl/blob/main/bench/README.md)
is preserved separately under `bench/` in this repository. Its performance
results are not used in this example.

```@raw html
</details>
```
