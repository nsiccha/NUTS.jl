module VitepressHTML

import DocumenterVitepress

export vitepress_html

# DocumenterVitepress through 0.3.9 omits backslash escaping in its v-html
# template literal. Stop compensating when the installed writer fixes it.
const DV_ESCAPES_BACKSLASH = let io = IOBuffer()
    DocumenterVitepress.render_mime(io, MIME"text/html"(), nothing,
                                  HTML("\\"), nothing, nothing)
    occursin("\\\\", String(take!(io)))
end

struct VitepressOutput
    html::String
end

"""Return an executed HTML result with its backslashes preserved in VitePress."""
vitepress_html(nodes...) =
    VitepressOutput(join(sprint(show, MIME"text/html"(), n) for n in nodes))

Base.show(io::IO, ::MIME"text/html", x::VitepressOutput) =
    print(io, DV_ESCAPES_BACKSLASH ? x.html : replace(x.html, '\\' => "\\\\"))

end
