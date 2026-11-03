# measure_reference_noise.jl
#
# Quantifica o ruído de amostragem da própria referência de Monte Carlo
# (10^7 simulações, 5 sementes de 2x10^6) usada em run_grid.jl, agora com
# r=0,542 (mediana 2012-2025). Para cada uma das 24 combinações (E[N],
# frequência, CV), roda as mesmas 5 sementes de 2x10^6 simulações, calcula
# o VaR 99,9% de cada semente isoladamente, e reporta o coeficiente de
# variação (desvio-padrão / média) entre as 5 estimativas. Resume a faixa
# de CV% separadamente para Poisson e para binomial negativa.

include(joinpath(@__DIR__, "src", "AggregateLoss.jl"))
using .AggregateLoss
using Distributions, Random, Statistics, Printf

const OUT = joinpath(@__DIR__, "data", "output")

function lognormal_from_mean_cv(mean_::Float64, cv::Float64)
    sigma2 = log(1 + cv^2)
    mu = log(mean_) - sigma2 / 2
    return LogNormal(mu, sqrt(sigma2))
end

const EN_GRID = [1.0, 10.0, 100.0, 1000.0]
const CV_GRID = [("baixo", 0.1619), ("medio", 0.3237), ("alto", 0.6475)]
const NB_R = 0.542
const ALPHA = 0.999

freq_specs(en) = [("poisson", PoissonFreq(en)), ("bn", NegBinFreq(en, NB_R))]

function main()
    rows = NamedTuple[]
    for en in EN_GRID
        for (freq_name, freq) in freq_specs(en)
            for (cv_name, cv) in CV_GRID
                sev = lognormal_from_mean_cv(1.0, cv)
                vars_por_semente = Float64[]
                t = @elapsed for s in 1:5
                    rng = MersenneTwister(1000 + s)
                    samp = monte_carlo_aggregate(freq, sev, 2_000_000; rng = rng)
                    ve = var_es(samp; alpha = ALPHA)
                    push!(vars_por_semente, ve.var)
                end
                mu = mean(vars_por_semente)
                sdv = std(vars_por_semente)
                cvpct = 100 * sdv / mu
                @printf("E[N]=%-6.0f %-8s CV=%-6s CV%% entre sementes = %.3f%% (%.1fs)\n",
                        en, freq_name, cv_name, cvpct, t)
                push!(rows, (en = en, freq = freq_name, cv_nome = cv_name, cv_pct_sementes = cvpct))
            end
        end
    end

    for freq_name in ("poisson", "bn")
        sub = [r.cv_pct_sementes for r in rows if r.freq == freq_name]
        @printf("\n%s: CV%% entre sementes varia de %.3f%% a %.3f%% (mediana %.3f%%)\n",
                freq_name, minimum(sub), maximum(sub), median(sub))
    end

    open(joinpath(OUT, "ruido_referencia.csv"), "w") do io
        println(io, "en,freq,cv_nome,cv_pct_sementes")
        for r in rows
            println(io, join([r.en, r.freq, r.cv_nome, r.cv_pct_sementes], ","))
        end
    end
    @printf("\n%d linhas exportadas em data/output/ruido_referencia.csv\n", length(rows))
end

main()
