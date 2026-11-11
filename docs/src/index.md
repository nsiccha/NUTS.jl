# NUTS.jl

NUTS.jl takes No-U-Turn Sampler transitions for a log density and gradient you
provide, using a step size and metric you choose.

Install from a terminal; this command changes your Julia environment and is not
run by the documentation build.

```sh
julia -e 'using Pkg; Pkg.add(url="https://github.com/nsiccha/NUTS.jl")'
```

Take one transition for a standard-normal target:

```@example start
using NUTS, Random, LinearAlgebra
struct StandardNormal end
function NUTS.log_density_gradient!(::StandardNormal, q, gradient)
    gradient .= -q
    return -sum(abs2, q) / 2
end
state = (rng=Xoshiro(42), posterior=StandardNormal(),
         position=zeros(2), stepsize=0.5, scale=I, squared_scale=I)
state = nuts!!(state)
@assert all(isfinite, state.position) # hide
(; position=round.(state.position; digits=3), state.divergent)
```

The returned position is a draw after one transition. Reassign the returned
state on each call; its arrays are reused.

## Reading

| Task | Page |
| :--- | :--- |
| Define and check a target | [Log density](guides/targets.md) |
| Configure transitions and retain draws | [Sampler state](guides/state.md) |
| Run and inspect a complete chain | [Gaussian coordinates](examples/gaussian.md) |
| Look up the interface | [API](reference/api.md) |

The package supplies the transition kernel. It does not provide automatic
differentiation, constrained-parameter transformations or warmup adaptation.
