# API

## Transitions

```@docs
nuts!!
```

The [state guide](../guides/state.md) explains buffer reuse, metric conventions
and the interpretation of transition diagnostics.

## Target callbacks

```@docs
NUTS.log_density_gradient!
```

The [log-density guide](../guides/targets.md) implements this extension point
and checks its gradient. The [Gaussian example](../examples/gaussian.md) uses it
in a complete chain.
