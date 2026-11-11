using Documenter, DocumenterVitepress, NUTS
using AlgebraOfVega
include("vitepress_html.jl")

# Copy generated assets before the renderer builds each VitePress version base.
abstract type CopyVegaAssets <: Documenter.Builder.DocumentPipeline end
Documenter.Selectors.order(::Type{CopyVegaAssets}) = 5.5
function Documenter.Selectors.runner(::Type{CopyVegaAssets}, doc::Documenter.Document)
    build = isabspath(doc.user.build) ? doc.user.build : joinpath(doc.user.root, doc.user.build)
    destination = joinpath(build, "public", "assets", "vega")
    mkpath(dirname(destination))
    cp(vega_vendor_dir(), destination; force=true)
end

const DOCS_PAGES = [
    "Start" => ["NUTS.jl" => "index.md"],
    "Guides" => ["Log density" => "guides/targets.md",
                 "Sampler state" => "guides/state.md"],
    "Examples" => ["Gaussian coordinates" => "examples/gaussian.md"],
    "Reference" => ["API" => "reference/api.md"],
]
include("lint.jl")
lint_docs(joinpath(@__DIR__, "src"), DOCS_PAGES)

makedocs(;
    modules=[NUTS],
    sitename="NUTS.jl",
    format=DocumenterVitepress.MarkdownVitepress(
        repo="github.com/nsiccha/NUTS.jl",
        devbranch="main", devurl="dev",
    ),
    pages=DOCS_PAGES,
    checkdocs=:exports,
    warnonly=false,
    doctest=true,
)

DocumenterVitepress.deploydocs(;
    repo="github.com/nsiccha/NUTS.jl", devbranch="main", push_preview=true,
)
