using AlgebraOfVega
import HTMXObjects

function show_figure(spec, n_draws)
    caption = "Quantiles from $n_draws retained draws per coordinate. Blue points " *
              "compare sample and normal quantiles; the orange line marks equality. " *
              "Both axes use units of the target standard deviation."
    node = with_plot_caption(spec; plot_id="gaussian-quantiles",
                             title="Gaussian quantiles", short=caption)
    runtime = join(repr(MIME"text/html"(), script)
                   for script in vega_head(; source=:vendor, base="../assets/vega",
                                          runtime=:linked))
    return Base.HTML(runtime * repr(MIME"text/html"(), node))
end
