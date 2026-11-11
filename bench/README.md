# Gaussian timing comparison

`gaussian.qmd` preserves the experiment formerly published as the NUTS landing
page. Its results are not included in the documentation and no figures or draws
are committed. The original experiment is also available at commit
`c22a7f0c51d9104c50ceb42574c35c63761de027`.

This is a research generator, not a validation test. Julia methods time their
transition loops; Stan times an external sampling call. The non-standard target
and Stan initialization use global randomness. These choices must be revised
and recorded before drawing new performance conclusions. No timings from this
script are asserted by the current documentation.

For an explicitly requested benchmark, provision CmdStan, set `CMDSTAN` to that
installation, and prepare the research environment from the repository root:

```sh
julia --project=bench -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
quarto render bench/gaussian.qmd
```

Configure Quarto's Julia kernel to use `bench/Project.toml`. Record the source
revision, package versions, hardware, seeds, timing boundaries and command with
any result. Results belong outside repository history. The documentation build
does not invoke this generator.
