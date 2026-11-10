using Documenter, DocumenterVitepress, NUTS

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
        repo="https://github.com/nsiccha/NUTS.jl",
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
