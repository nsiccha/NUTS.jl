"Check the reader-facing page inventory and executable-example conventions."
function lint_docs(root, pages)
    first.(pages) == ["Start", "Guides", "Examples", "Reference"] ||
        error("Navigation must have Start, Guides, Examples, Reference in that order")
    listed = [last(page) for group in pages for page in last(group)]
    actual = [relpath(joinpath(dir, file), root)
              for (dir, _, files) in walkdir(root) for file in files if endswith(file, ".md")]
    sort(listed) == sort(actual) || error("Every page must appear in navigation exactly once")
    fences = r"(?ms)^```([^\n]*)\n(.*?)^```[ \t]*$"
    banned = r"(?i)\b(seamless|robust|powerful|leverage|delve|crucial|comprehensive|unlock|simply)\b|it's worth noting"
    for page in listed
        source = read(joinpath(root, page), String)
        for block in eachmatch(fences, source)
            kind = strip(block.captures[1])
            (startswith(kind, "@") || kind in ("sh", "bash", "shell")) ||
                error("$page: runnable code must use an executable Documenter block: $kind")
        end
        prose = replace(source, fences => "", r"\[([^]]+)\]\([^)]*\)" => s"\1")
        occursin(banned, prose) && error("$page: prohibited prose")
        words = length(split(prose))
        startswith(page, "guides/") && words >= 1500 && error("$page: guide exceeds 1500 words")
        if startswith(page, "examples/")
            800 <= words <= 1500 || error("$page: expected 800–1500 prose words, got $words")
            occursin("```@example", source) || error("$page: missing executed example")
            occursin("show_figure(", source) || error("$page: missing generated figure")
        end
        println("DOCS-LINT $page prose_words=$words")
    end
end
