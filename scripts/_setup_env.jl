import Pkg

# ---- (A) 通过包服务器添加 General（不会走 git clone）----
try
    Pkg.Registry.add("General")
    Pkg.Registry.update()
catch e
    @warn "Registry add/update failed; will still try to add packages" err=e
end

# ---- (B) 安装依赖（这三个都是注册包，包服务器可直接发 tarball）----
Pkg.add(["OpenDSSDirect", "Ipopt", "PowerModelsDistribution"])

# 可选：再更新一下依赖解析
# Pkg.update()

# 关自动预编译也能跑，但这里显式预编译一次更稳
try
    Pkg.precompile()
catch e
    @warn "precompile failed" err=e
end

# ---- (C) 自检 ----
using OpenDSSDirect, PowerModelsDistribution, Ipopt
println("JL OK")
