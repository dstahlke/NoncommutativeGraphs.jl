using Documenter, NoncommutativeGraphs

DocMeta.setdocmeta!(NoncommutativeGraphs, :DocTestSetup, :( using NoncommutativeGraphs; using LinearAlgebra ); recursive=true)

makedocs(
    sitename="NoncommutativeGraphs.jl",
    modules=[NoncommutativeGraphs],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "Home" => "index.md",
        "Reference" => "reference.md",
    ],
)

deploydocs(
    repo = "github.com/dstahlke/NoncommutativeGraphs.jl.git",
    devbranch = "main",
    branch = "gh-pages",
)
