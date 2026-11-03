# mc_tolerance_cost.jl
#
# Para os 16 cenários (de 24) com h < 0,3×DP da severidade em m=10.000
# (run_grid.jl), varre o número de simulações n do Monte Carlo numa grade
# log-espaçada de 10^4 a 10^7 e, em cada n, roda várias sementes
# independentes para estimar o erro ESPERADO do VaR 99,9% (RMSE entre
# sementes, contra a referência de alta resolução). O número de sementes
# cai para n grandes, por custo computacional, declarado na própria saída.
#
# A referência é a FFT determinística em m=1.000.000 (sem ruído de
# amostragem), já usada e validada em mc_cost_comparison.jl. Os pontos
# (n, rmse) desta varredura alimentam, num passo de análise em Python, a
# interpolação log-log que encontra o menor n para cada tolerância de erro
# (1%, 0,5%, 0,1%) e a comparação de tempo contra Panjer e FFT em m=10.000.
#
# Uso: julia --project=. mc_tolerance_cost.jl <E[N]>
#   julia --project=. mc_tolerance_cost.jl 1
#   julia --project=. mc_tolerance_cost.jl 10
#   julia --project=. mc_tolerance_cost.jl 100
#   julia --project=. mc_tolerance_cost.jl 1000
# Cada chamada acrescenta linhas a data/output/mc_tolerance_sweep.csv e
# data/output/mc_tolerance_pontual.csv (o arquivo é recriado só na chamada
# com E[N]=1).

include(joinpath(@__DIR__, "src", "AggregateLoss.jl"))
using .AggregateLoss
using Distributions, Random, Statistics, Printf

const OUT = joinpath(@__DIR__, "data", "output")

function lognormal_from_mean_cv(mean_::Float64, cv::Float64)
    sigma2 = log(1 + cv^2)
    mu = log(mean_) - sigma2 / 2
    return LogNormal(mu, sqrt(sigma2))
end

const NB_R = 0.542
const CV_MAP = Dict("baixo" => 0.1619, "medio" => 0.3237, "alto" => 0.6475)
const ALPHA = 0.999
const M_TRUTH = 1_000_000
const M_TEST = 10_000

# 14 pontos log-espaçados de 10^4 a 10^7, razão ~1,70 (mais fino que √3≈1,73)
const N_GRID = round.(Int, exp10.(range(4, 7, length = 14)))

const CENARIOS_POR_EN = Dict(
    1.0 => [("bn", "alto"), ("bn", "baixo"), ("bn", "medio"),
            ("poisson", "alto"), ("poisson", "baixo"), ("poisson", "medio")],
    10.0 => [("bn", "alto"), ("bn", "baixo"), ("bn", "medio"),
             ("poisson", "alto"), ("poisson", "baixo"), ("poisson", "medio")],
    100.0 => [("poisson", "alto"), ("poisson", "baixo"), ("poisson", "medio")],
    1000.0 => [("poisson", "alto")],
)

freq_for(en, freq_name) = freq_name == "poisson" ? PoissonFreq(en) : NegBinFreq(en, NB_R)

"Número de sementes por nível de n, mais baixo quanto maior E[N] e quanto maior n (custo computacional)."
function nseeds_for(en::Float64, n::Int)
    if en >= 1000.0
        n < 100_000 ? 10 : (n < 1_000_000 ? 5 : 2)
    elseif en >= 100.0
        n < 100_000 ? 15 : (n < 1_000_000 ? 8 : 4)
    else
        n < 100_000 ? 20 : (n < 1_000_000 ? 10 : 5)
    end
end

function main()
    length(ARGS) == 1 || error("uso: julia mc_tolerance_cost.jl <E[N]>")
    en = parse(Float64, ARGS[1])
    cenarios = CENARIOS_POR_EN[en]

    sweep_rows = NamedTuple[]
    pontual_rows = NamedTuple[]
    cell_seed = 51000 + round(Int, en)

    for (freq_name, cv_name) in cenarios
        cv = CV_MAP[cv_name]
        freq = freq_for(en, freq_name)
        sev = lognormal_from_mean_cv(1.0, cv)
        domain = suggest_domain(freq, sev)

        h_truth = domain / M_TRUTH
        g_truth = fft_aggregate(freq, sev, h_truth, M_TRUTH; pad_factor = 4)
        ve_truth = var_es(g_truth, h_truth; alpha = ALPHA)

        h_test = domain / M_TEST
        f_test = max.(discretize_severity(sev, h_test, M_TEST), 0.0)
        t_panjer = @elapsed (g_panjer, _) = panjer_recursion(freq, f_test)
        ve_panjer = var_es(g_panjer, h_test; alpha = ALPHA)
        erro_panjer = (ve_panjer.var - ve_truth.var) / ve_truth.var

        t_fft = @elapsed g_fft = fft_aggregate(freq, sev, h_test, M_TEST; pad_factor = 4)
        ve_fft = var_es(g_fft, h_test; alpha = ALPHA)

        @printf("\n=== E[N]=%.0f freq=%-7s CV=%-6s (VaR truth=%.4f) ===\n", en, freq_name, cv_name, ve_truth.var)
        @printf("  Panjer m=10000: erro=%+.4f%% t=%.4fs | FFT: t=%.4fs\n", 100erro_panjer, t_panjer, t_fft)

        push!(pontual_rows, (en = en, freq = freq_name, cv_nome = cv_name,
                              var_truth = ve_truth.var, erro_panjer_pct = 100erro_panjer,
                              tempo_panjer_s = t_panjer, tempo_fft_s = t_fft))

        for n in N_GRID
            ns = nseeds_for(en, n)
            vars = Float64[]
            tempos = Float64[]
            for s in 1:ns
                cell_seed += 1
                rng = MersenneTwister(cell_seed)
                t_mc = @elapsed mc_sample = monte_carlo_aggregate(freq, sev, n; rng = rng)
                ve_mc = var_es(mc_sample; alpha = ALPHA)
                push!(vars, ve_mc.var)
                push!(tempos, t_mc)
            end
            erros_pct = [(v - ve_truth.var) / ve_truth.var * 100 for v in vars]
            rmse_pct = sqrt(mean(erros_pct .^ 2))
            mae_pct = mean(abs.(erros_pct))
            tempo_mediana = median(tempos)
            @printf("  n=%-9d sementes=%-2d RMSE=%.4f%% MAE=%.4f%% tempo_mediano=%.4fs\n",
                    n, ns, rmse_pct, mae_pct, tempo_mediana)
            push!(sweep_rows, (en = en, freq = freq_name, cv_nome = cv_name, n = n, nseeds = ns,
                                rmse_pct = rmse_pct, mae_pct = mae_pct, tempo_mediana_s = tempo_mediana))
        end
    end

    sweep_path = joinpath(OUT, "mc_tolerance_sweep.csv")
    pontual_path = joinpath(OUT, "mc_tolerance_pontual.csv")
    fresh = en == 1.0 || !isfile(sweep_path)

    open(sweep_path, fresh ? "w" : "a") do io
        fresh && println(io, "en,freq,cv_nome,n,nseeds,rmse_pct,mae_pct,tempo_mediana_s")
        for r in sweep_rows
            println(io, join([r.en, r.freq, r.cv_nome, r.n, r.nseeds, r.rmse_pct, r.mae_pct, r.tempo_mediana_s], ","))
        end
    end
    open(pontual_path, fresh ? "w" : "a") do io
        fresh && println(io, "en,freq,cv_nome,var_truth,erro_panjer_pct,tempo_panjer_s,tempo_fft_s")
        for r in pontual_rows
            println(io, join([r.en, r.freq, r.cv_nome, r.var_truth, r.erro_panjer_pct, r.tempo_panjer_s, r.tempo_fft_s], ","))
        end
    end
    @printf("\nE[N]=%.0f: %d cenários, %d linhas de sweep exportadas\n", en, length(cenarios), length(sweep_rows))
end

main()
