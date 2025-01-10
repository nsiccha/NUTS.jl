# NUTS.jl

A non-allocating NUTS implementation. Faster than and equivalent to [Stan](https://mc-stan.org/)'s default implementation, [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl)'s implementation, and [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl)'s `HMCKernel(Trajectory{MultinomialTS}(Leapfrog(stepsize), StrictGeneralisedNoUTurn()))`. 

For a 100 dimensional standard normal target with unit stepsize and 1k samples, [I measure it](https://nsiccha.github.io/NUTS.jl/#benchmark) to be ~5x slower than direct sampling (`randn!(...)`), ~6x faster than DynamicHMC, ~15x faster than AdvancedHMC and ~25x faster than [Stan.jl](https://github.com/StanJulia/Stan.jl). **For most other posteriors the computational cost will be dominated by the cost of evaluating the log density gradient, so any real world speed-ups should be smaller.**

## Usage

Exports a single function, `nuts!!(state)`. Use e.g. as

```julia
nuts_sample!(samples, rng, posterior; stepsize, position=randn(rng, size(samples, 1)), n_samples=size(samples, 2)) = begin
    state = (;rng, posterior, stepsize, position)
    for i in 1:n_samples
        state = nuts!!(state)
        samples[:, i] .= state.position
    end
    state
end
```

where `posterior` has to implement `log_density = NUTS.log_density_gradient!(posterior, position, log_density_gradient)`,
i.e. it returns the log density and writes its gradient into `log_density_gradient`.
