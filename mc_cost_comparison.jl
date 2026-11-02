# mc_cost_comparison.jl
#
# Mede, para as 24 combinações (E[N], frequência, CV) já calibradas em
# run_grid.jl, quanto tempo o Monte Carlo precisa para chegar perto do erro
# médio que Panjer/FFT entregam nos cenários bem discretizados (0,18% no
# VaR 99,9%, ver RESULTADOS.md). Usa a referência já salva em
# grid_referencias.csv (10^7 simulações, 5 sementes) como "verdade", e roda
# Monte Carlo com n crescente (1e5, 3e5, 1e6, 3e6) para ver em que ponto o
# erro cruza o patamar de Panjer/FFT e quanto tempo isso custou de fato,
# não por extrapolação.

include(joinpath(@__DIR__, "src", "AggregateLoss.jl"))
using .AggregateLoss
using Distributions, Random, Statistics, Printf

const OUT = joinpath(@__DIR__, "data", "output")

function lognormal_from_mean_cv(mean_::Float64, cv::Float64)
    sigma2 = log(1 + cv^2)
    mu = log(mean_) - sigma2 / 2
    return LogNormal(mu, sqrt(sigma2))
end

const NB_R = 0.72
const CV_MAP = Dict("baixo" => 0.1619, "medio" => 0.3237, "alto" => 0.6475)
const ALPHA = 0.999
const N_LEVELS = [100_000, 300_000, 1_000_000, 3_000_000]

function read_referencias(path)
    rows = NamedTuple[]
    open(path, "r") do io
        readline(io)
        for line in eachline(io)
            cols = split(line, ',')
            push!(rows, (en = parse(Float64, cols[1]), freq = cols[2], cv_nome = cols[3],
                          ref_var = parse(Float64, cols[6]), ref_es = parse(Float64, cols[7])))
        end
    end
    return rows
end

function main()
    refs = read_referencias(joinpath(OUT, "grid_referencias.csv"))
    out_rows = NamedTuple[]
    cell_seed = 777

    for r in refs
        sev = lognormal_from_mean_cv(1.0, CV_MAP[r.cv_nome])
        freq = r.freq == "poisson" ? PoissonFreq(r.en) : NegBinFreq(r.en, NB_R)
        @printf("\n=== E[N]=%.0f freq=%s CV=%s (ref VaR=%.4f) ===\n", r.en, r.freq, r.cv_nome, r.ref_var)
        for n in N_LEVELS
            rng = MersenneTwister(cell_seed)
            t = @elapsed samp = monte_carlo_aggregate(freq, sev, n; rng = rng)
            ve = var_es(samp; alpha = ALPHA)
            erro = 100 * (ve.var - r.ref_var) / r.ref_var
            @printf("  n=%-9d t=%8.4fs  VaR=%.4f  erro=%+.3f%%\n", n, t, ve.var, erro)
            push!(out_rows, (en = r.en, freq = r.freq, cv_nome = r.cv_nome, n = n,
                              tempo_s = t, var = ve.var, erro_pct = erro))
            cell_seed += 1
        end
    end

    open(joinpath(OUT, "mc_custo_precisao.csv"), "w") do io
        println(io, "en,freq,cv_nome,n,tempo_s,var,erro_pct")
        for r in out_rows
            println(io, join([r.en, r.freq, r.cv_nome, r.n, r.tempo_s, r.var, r.erro_pct], ","))
        end
    end
    @printf("\n%d linhas exportadas em mc_custo_precisao.csv\n", length(out_rows))
end

main()
