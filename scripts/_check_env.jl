import Pkg
Pkg.activate(joinpath(@__DIR__, "."))
Pkg.instantiate()
using OpenDSSDirect, PowerModelsDistribution, Ipopt
println("JL OK")