# remeasure_panjer_fft_timing.jl
#
# Mede o tempo de Panjer e da FFT em m=10.000, nos mesmos 16 cenários de
# mc_tolerance_cost.jl, com aquecimento da JIT (uma chamada descartada antes
# de cronometrar) e mediana de 7 execuções por cenário, no mesmo ambiente
# (mesma sessão Julia, mesma máquina, sem outro processo pesado concorrente
# além do próprio script). Cada tempo cronometrado inclui a discretização da
# severidade (discretize_severity), a recursão ou a transformada
# propriamente dita, e a extração do quantil (var_es) — o mesmo trabalho de
# ponta a ponta que o Monte Carlo faz (simular, depois extrair o quantil).
#
# Mede também, separadamente, o tempo da própria extração do quantil
# (var_es sobre uma amostra ordenada) em função do tamanho da amostra, para
# somar esse custo ao tempo do Monte Carlo já medido em mc_tolerance_cost.jl
# (que cronometrou só monte_carlo_aggregate, sem var_es), sem precisar
# re-rodar a simulação em si.
#
# Sobrescreve data/output/mc_tolerance_pontual.csv com os tempos de Panjer e
# FFT remedidos, e grava data/output/mc_quantile_overhead.csv com o tempo
# mediano de var_es por tamanho de amostra (mesma grade de n de
# mc_tolerance_cost.jl).

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
const N_REPS = 7
const N_GRID = round.(Int, exp10.(range(4, 7, length = 14)))

const CENARIOS = [
    (1.0, "bn", "alto"), (1.0, "bn", "baixo"), (1.0, "bn", "medio"),
    (1.0, "poisson", "alto"), (1.0, "poisson", "baixo"), (1.0, "poisson", "medio"),
    (10.0, "bn", "alto"), (10.0, "bn", "baixo"), (10.0, "bn", "medio"),
    (10.0, "poisson", "alto"), (10.0, "poisson", "baixo"), (10.0, "poisson", "medio"),
    (100.0, "poisson", "alto"), (100.0, "poisson", "baixo"), (100.0, "poisson", "medio"),
    (1000.0, "poisson", "alto"),
]

freq_for(en, freq_name) = freq_name == "poisson" ? PoissonFreq(en) : NegBinFreq(en, NB_R)

"Mediana de N_REPS execuções cronometradas, após uma chamada de aquecimento descartada."
function timed_median(f::Function)
    f()  # aquecimento: força a compilação JIT antes de cronometrar
    tempos = [(@elapsed f()) for _ in 1:N_REPS]
    return median(tempos)
end

function main()
    rows = NamedTuple[]

    for (en, freq_name, cv_name) in CENARIOS
        cv = CV_MAP[cv_name]
        freq = freq_for(en, freq_name)
        sev = lognormal_from_mean_cv(1.0, cv)
        domain = suggest_domain(freq, sev)
        h_test = domain / M_TEST

        t_panjer = timed_median() do
            f = max.(discretize_severity(sev, h_test, M_TEST), 0.0)
            g, _ = panjer_recursion(freq, f)
            var_es(g, h_test; alpha = ALPHA)
        end

        t_fft = timed_median() do
            g = fft_aggregate(freq, sev, h_test, M_TEST; pad_factor = 4)
            var_es(g, h_test; alpha = ALPHA)
        end

        # erro de Panjer contra a referência de alta resolução (FFT m=1.000.000),
        # recalculado aqui para manter pontual.csv autocontido (valor idêntico
        # ao medido em mc_tolerance_cost.jl, não depende do tempo de execução)
        h_truth = domain / M_TRUTH
        g_truth = fft_aggregate(freq, sev, h_truth, M_TRUTH; pad_factor = 4)
        ve_truth = var_es(g_truth, h_truth; alpha = ALPHA)
        f_test = max.(discretize_severity(sev, h_test, M_TEST), 0.0)
        g_panjer, _ = panjer_recursion(freq, f_test)
        ve_panjer = var_es(g_panjer, h_test; alpha = ALPHA)
        erro_panjer = (ve_panjer.var - ve_truth.var) / ve_truth.var

        @printf("E[N]=%-6.0f %-7s %-6s Panjer=%.4fs (med. %d) FFT=%.4fs (med. %d)\n",
                en, freq_name, cv_name, t_panjer, N_REPS, t_fft, N_REPS)

        push!(rows, (en = en, freq = freq_name, cv_nome = cv_name,
                      var_truth = ve_truth.var, erro_panjer_pct = 100erro_panjer,
                      tempo_panjer_s = t_panjer, tempo_fft_s = t_fft))
    end

    open(joinpath(OUT, "mc_tolerance_pontual.csv"), "w") do io
        println(io, "en,freq,cv_nome,var_truth,erro_panjer_pct,tempo_panjer_s,tempo_fft_s")
        for r in rows
            println(io, join([r.en, r.freq, r.cv_nome, r.var_truth, r.erro_panjer_pct, r.tempo_panjer_s, r.tempo_fft_s], ","))
        end
    end
    @printf("\n%d cenários remedidos em mc_tolerance_pontual.csv\n", length(rows))

    # --- tempo da extração do quantil isolada (var_es_partial, seleção parcial) ---
    println("\n=== Tempo de var_es_partial (quantil) por tamanho de amostra ===")
    overhead_rows = NamedTuple[]
    rng = MersenneTwister(99)
    for n in N_GRID
        amostra_base = rand(rng, n)
        t_quantil = timed_median() do
            var_es_partial(copy(amostra_base); alpha = ALPHA)
        end
        @printf("  n=%-9d tempo_quantil=%.5fs\n", n, t_quantil)
        push!(overhead_rows, (n = n, tempo_quantil_s = t_quantil))
    end
    open(joinpath(OUT, "mc_quantile_overhead.csv"), "w") do io
        println(io, "n,tempo_quantil_s")
        for r in overhead_rows
            println(io, join([r.n, r.tempo_quantil_s], ","))
        end
    end
    @printf("\n%d linhas exportadas em mc_quantile_overhead.csv\n", length(overhead_rows))
end

main()
