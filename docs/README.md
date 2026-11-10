# Documentation

The page list is the reader's navigation, in this order:

| Group | Page | Purpose |
| --- | --- | --- |
| Start | `src/index.md` | Install NUTS and execute one transition with an explicit gradient. |
| Guides | `src/guides/targets.md` | Implement and check a log-density gradient callback. |
| Guides | `src/guides/state.md` | Reuse state, supply a metric and read transition diagnostics. |
| Examples | `src/examples/gaussian.md` | Sample a Gaussian target with unequal scales and compare draws with its known distribution. |
| Reference | `src/reference/api.md` | Look up the transition function and callback contract. |

The old Quarto comparison lives in `../bench/`; it is not part of the site.
The root README links to Start instead of maintaining another example.

Build from the repository root:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The build executes the teaching examples and generates their figures under
`docs/build/`. It runs no benchmark or fitted-data model. It checks runnable
fences, prohibited prose, page lengths, example figures and the navigation
before Documenter runs with `warnonly=false` and `checkdocs=:exports`.

VitePress uses its upstream theme. The palette and figure rendering are local
configuration pending the shared DocsKit package; no other site's theme or
widget implementation is copied here. AlgebraOfVega generates Vega-Lite figures
from build-executed calculations; its runtime follows the page's light/dark theme.
