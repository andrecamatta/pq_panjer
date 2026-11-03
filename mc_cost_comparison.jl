# mc_cost_comparison.jl
#
# Mede, cenário a cenário, quanto tempo o Monte Carlo precisa para igualar o
# erro que Panjer e a FFT entregam em m=10.000 — restrito aos 16 cenários
# (de 24) em que a largura de discretização h já está abaixo de 0,3×DP da
# severidade nessa grade (run_grid.jl, data/output/grid_resultados.csv).
# Fora desses 16, Panjer e a FFT já erram dezenas de pontos percentuais por
# falta de resolução, e comparar "tempo para a mesma precisão" não faz
# sentido (ver seção "A dispersão real do crédito setorial explode o
# domínio" do artigo).
#
# A referência usada para medir erro NÃO é mais o Monte Carlo de 10^7
# simulações (que carrega ruído de amostragem da mesma ordem do erro de
# Panjer nesses cenários bem discretizados, inflando artificialmente o
# "piso" do erro do Monte Carlo). Em vez disso, cada cenário usa uma
# referência de FFT em m=1.000.000 pontos — determinística, sem ruído de
# amostragem, convergida nesse domínio (h cem vezes menor que o de m=10.000,
# já bem dentro da faixa onde o erro de discretização é desprezível).
#
# Para cada um dos 16 cenários, roda o Monte Carlo com n crescente
# (10^4 a 10^7) e procura o primeiro n cujo erro (contra a referência FFT)
# fica abaixo do erro de Panjer no mesmo cenário. Se isso não acontece até
# 10^7, o cenário é marcado como "piso": o ruído de amostragem do próprio
# Monte Carlo, não mais da referência, impede medir onde o cruzamento
# aconteceria, e o tempo a 10^7 é reportado como limite inferior.

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
const N_LEVELS = [10_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000]

# os 16 cenários (de 24) com h/DP < 0,3 em m=10.000 na grade principal
# (run_grid.jl, r=0,542), identificados em data/output/grid_resultados.csv
const CENARIOS_BEM_DISCRETIZADOS = [
    (1.0, "bn", "alto"), (1.0, "bn", "baixo"), (1.0, "bn", "medio"),
    (1.0, "poisson", "alto"), (1.0, "poisson", "baixo"), (1.0, "poisson", "medio"),
    (10.0, "bn", "alto"), (10.0, "bn", "baixo"), (10.0, "bn", "medio"),
    (10.0, "poisson", "alto"), (10.0, "poisson", "baixo"), (10.0, "poisson", "medio"),
    (100.0, "poisson", "alto"), (100.0, "poisson", "baixo"), (100.0, "poisson", "medio"),
    (1000.0, "poisson", "alto"),
]

freq_for(en, freq_name) = freq_name == "poisson" ? PoissonFreq(en) : NegBinFreq(en, NB_R)

function main()
    rows = NamedTuple[]
    resumo = NamedTuple[]
    cell_seed = 31000

    for (en, freq_name, cv_name) in CENARIOS_BEM_DISCRETIZADOS
        cv = CV_MAP[cv_name]
        freq = freq_for(en, freq_name)
        sev = lognormal_from_mean_cv(1.0, cv)
        domain = suggest_domain(freq, sev)

        # --- referência de alta resolução: FFT determinística em m=1.000.000 ---
        h_truth = domain / M_TRUTH
        g_truth = fft_aggregate(freq, sev, h_truth, M_TRUTH; pad_factor = 4)
        ve_truth = var_es(g_truth, h_truth; alpha = ALPHA)

        # --- Panjer e FFT em m=10.000 (a grade testada no artigo) ---
        h_test = domain / M_TEST
        f_test = max.(discretize_severity(sev, h_test, M_TEST), 0.0)
        t_panjer = @elapsed (g_panjer, _) = panjer_recursion(freq, f_test)
        ve_panjer = var_es(g_panjer, h_test; alpha = ALPHA)
        erro_panjer = (ve_panjer.var - ve_truth.var) / ve_truth.var

        t_fft = @elapsed g_fft = fft_aggregate(freq, sev, h_test, M_TEST; pad_factor = 4)
        ve_fft = var_es(g_fft, h_test; alpha = ALPHA)
        erro_fft = (ve_fft.var - ve_truth.var) / ve_truth.var

        @printf("\n=== E[N]=%.0f freq=%-7s CV=%-6s (VaR truth=%.4f) ===\n", en, freq_name, cv_name, ve_truth.var)
        @printf("  Panjer m=10000: erro=%+.4f%% t=%.4fs\n", 100erro_panjer, t_panjer)
        @printf("  FFT    m=10000: erro=%+.4f%% t=%.4fs\n", 100erro_fft, t_fft)

        cruzamento_n = missing
        cruzamento_t = missing
        piso = true

        for n in N_LEVELS
            cell_seed += 1
            rng = MersenneTwister(cell_seed)
            t_mc = @elapsed mc_sample = monte_carlo_aggregate(freq, sev, n; rng = rng)
            ve_mc = var_es(mc_sample; alpha = ALPHA)
            erro_mc = (ve_mc.var - ve_truth.var) / ve_truth.var
            @printf("  MC n=%-9d erro=%+.4f%% t=%.4fs\n", n, 100erro_mc, t_mc)
            push!(rows, (en = en, freq = freq_name, cv_nome = cv_name, n = n,
                          erro_mc_pct = 100erro_mc, tempo_mc_s = t_mc,
                          erro_panjer_pct = 100erro_panjer, tempo_panjer_s = t_panjer,
                          erro_fft_pct = 100erro_fft, tempo_fft_s = t_fft,
                          var_truth = ve_truth.var))
            if ismissing(cruzamento_n) && abs(erro_mc) <= abs(erro_panjer)
                cruzamento_n = n
                cruzamento_t = t_mc
                piso = false
            end
        end

        ultimo_n = N_LEVELS[end]
        ultimo_t = rows[end].tempo_mc_s
        n_ref = piso ? ultimo_n : cruzamento_n
        t_ref = piso ? ultimo_t : cruzamento_t
        razao_panjer = t_ref / t_panjer
        razao_fft = t_ref / t_fft
        @printf("  -> %s: n=%d t=%.3fs (%.1fx Panjer, %.1fx FFT)%s\n",
                piso ? "PISO (limite inferior)" : "cruzamento", n_ref, t_ref,
                razao_panjer, razao_fft, piso ? " [não convergiu até 10^7]" : "")

        push!(resumo, (en = en, freq = freq_name, cv_nome = cv_name,
                        erro_panjer_pct = 100erro_panjer, erro_fft_pct = 100erro_fft,
                        tempo_panjer_s = t_panjer, tempo_fft_s = t_fft,
                        n_cruzamento = n_ref, tempo_cruzamento_s = t_ref,
                        razao_tempo_panjer = razao_panjer, razao_tempo_fft = razao_fft,
                        piso = piso))
    end

    open(joinpath(OUT, "mc_custo_precisao.csv"), "w") do io
        println(io, "en,freq,cv_nome,n,erro_mc_pct,tempo_mc_s,erro_panjer_pct,tempo_panjer_s,erro_fft_pct,tempo_fft_s,var_truth")
        for r in rows
            println(io, join([r.en, r.freq, r.cv_nome, r.n, r.erro_mc_pct, r.tempo_mc_s,
                               r.erro_panjer_pct, r.tempo_panjer_s, r.erro_fft_pct, r.tempo_fft_s, r.var_truth], ","))
        end
    end
    open(joinpath(OUT, "mc_custo_precisao_resumo.csv"), "w") do io
        println(io, "en,freq,cv_nome,erro_panjer_pct,erro_fft_pct,tempo_panjer_s,tempo_fft_s,n_cruzamento,tempo_cruzamento_s,razao_tempo_panjer,razao_tempo_fft,piso")
        for r in resumo
            println(io, join([r.en, r.freq, r.cv_nome, r.erro_panjer_pct, r.erro_fft_pct,
                               r.tempo_panjer_s, r.tempo_fft_s, r.n_cruzamento, r.tempo_cruzamento_s,
                               r.razao_tempo_panjer, r.razao_tempo_fft, r.piso], ","))
        end
    end

    @printf("\n%d cenários, %d linhas de n exportadas\n", length(resumo), length(rows))
    n_piso = count(r -> r.piso, resumo)
    @printf("%d de %d cenários ficaram no piso (não convergiram até 10^7 simulações)\n", n_piso, length(resumo))
end

main()
