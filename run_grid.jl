# run_grid.jl
#
# Grade fatorial completa do brief: E[N] ∈ {1,10,100,1000} × {Poisson, BN} ×
# CV da severidade {baixo,médio,alto} × grid {10²,10³,10⁴} pontos, com
# referência de Monte Carlo de alta precisão (10⁷ simulações, 5 sementes) por
# combinação (E[N], frequência, CV), comparando Panjer, FFT sem tilt, FFT com
# tilt automático, e Monte Carlo rápido (10⁴ e 10⁵) contra essa referência.
#
# Severidade: lognormal, média normalizada em 1 unidade de perda, CV
# ancorado no WP193 do BCB (ver calibrate_severity.jl). Frequência: Poisson e
# binomial negativa com dispersão r calibrada na heterogeneidade setorial do
# SCR.data (ver calibrate_frequency.jl).
#
# Domínio de discretização: E[S] + 15·SD(S) (fórmulas de momento do
# composto), dimensionado por cenário (E[N], frequência, CV), ANTES de
# escolher a largura h = domínio/m — ou seja, o "grid de 10^k pontos" varia a
# RESOLUÇÃO (h), não o alcance do domínio, o que é o próprio objeto de
# medição pedido (efeito da largura de discretização).

include(joinpath(@__DIR__, "src", "AggregateLoss.jl"))
using .AggregateLoss
using Distributions, Random, Statistics, Printf

const OUT = joinpath(@__DIR__, "data", "output")
mkpath(OUT)

function lognormal_from_mean_cv(mean_::Float64, cv::Float64)
    sigma2 = log(1 + cv^2)
    mu = log(mean_) - sigma2 / 2
    return LogNormal(mu, sqrt(sigma2))
end

const EN_GRID = [1.0, 10.0, 100.0, 1000.0]
const CV_GRID = [("baixo", 0.1619), ("medio", 0.3237), ("alto", 0.6475)]
const M_GRID = [100, 1000, 10000]
const NB_R = 0.72  # calibrado em calibrate_frequency.jl (média de 4 trimestres de 2025)
const ALPHA = 0.999
const P_LOWER_KS = 0.99

freq_specs(en) = [("poisson", PoissonFreq(en)), ("bn", NegBinFreq(en, NB_R))]

"Referência de Monte Carlo: nseeds sementes de nsim simulações cada, agrupadas."
function mc_reference(freq, sev; nsim = 2_000_000, nseeds = 5, base_seed = 1000)
    chunks = Vector{Vector{Float64}}(undef, nseeds)
    t = @elapsed for s in 1:nseeds
        rng = MersenneTwister(base_seed + s)
        chunks[s] = monte_carlo_aggregate(freq, sev, nsim; rng = rng)
    end
    pooled = vcat(chunks...)
    return pooled, t
end

function run_cell(en, freq_name, freq, cv_name, cv, m, sev, ref_sample, ref_var, ref_es, cell_seed)
    domain = suggest_domain(freq, sev)
    h = domain / m

    rows = NamedTuple[]

    # --- Panjer ---
    f = max.(discretize_severity(sev, h, m), 0.0)
    # média da severidade discretizada: quantifica a distorção do método do
    # arredondamento quando h é grande em relação ao DESVIO-PADRÃO da
    # severidade (não só à média) — ver nota no RESULTADOS.md.
    sev_mean_discretizado = sum((k - 1) * h * f[k] for k in 1:m)
    t_panjer = @elapsed begin
        g_panjer, underflow_panjer = panjer_recursion(freq, f)
    end
    ve = var_es(g_panjer, h; alpha = ALPHA)
    ks = ks_tail(g_panjer, h, ref_sample; p_lower = P_LOWER_KS, presorted = true)
    push!(rows, (metodo = "panjer", var = ve.var, es = ve.es, ks = ks, tempo_s = t_panjer,
                  underflow = underflow_panjer, massa_truncada = 1.0 - sum(f)))

    # --- FFT sem tilt ---
    t_fft = @elapsed begin
        g_fft = fft_aggregate(freq, sev, h, m; pad_factor = 4)
    end
    ve = var_es(g_fft, h; alpha = ALPHA)
    ks = ks_tail(g_fft, h, ref_sample; p_lower = P_LOWER_KS, presorted = true)
    push!(rows, (metodo = "fft_sem_tilt", var = ve.var, es = ve.es, ks = ks, tempo_s = t_fft,
                  underflow = false, massa_truncada = NaN))

    # --- FFT com tilt (auto, regra ~20/domínio completo) ---
    t_ffttilt = @elapsed begin
        g_ffttilt = fft_aggregate(freq, sev, h, m; auto_tilt = true, pad_factor = 4)
    end
    ve = var_es(g_ffttilt, h; alpha = ALPHA)
    ks = ks_tail(g_ffttilt, h, ref_sample; p_lower = P_LOWER_KS, presorted = true)
    push!(rows, (metodo = "fft_com_tilt", var = ve.var, es = ve.es, ks = ks, tempo_s = t_ffttilt,
                  underflow = false, massa_truncada = NaN))

    # --- Monte Carlo rápido: 1e4 e 1e5 ---
    for (label, nsim) in (("mc_1e4", 10_000), ("mc_1e5", 100_000))
        rng = MersenneTwister(cell_seed)
        t_mc = @elapsed begin
            mc_sample = monte_carlo_aggregate(freq, sev, nsim; rng = rng)
        end
        ve = var_es(mc_sample; alpha = ALPHA)
        ks = ks_tail(mc_sample, ref_sample; p_lower = P_LOWER_KS, ref_presorted = true)
        push!(rows, (metodo = label, var = ve.var, es = ve.es, ks = ks, tempo_s = t_mc,
                      underflow = false, massa_truncada = NaN))
    end

    return [(en = en, freq = freq_name, cv_nome = cv_name, cv = cv, m = m, h = h, domain = domain,
              sev_mean_discretizado = sev_mean_discretizado,
              ref_var = ref_var, ref_es = ref_es,
              erro_rel_var = (r.var - ref_var) / ref_var, erro_rel_es = (r.es - ref_es) / ref_es, r...)
             for r in rows]
end

function main()
    all_rows = NamedTuple[]
    ref_rows = NamedTuple[]
    cell_seed = 42

    for en in EN_GRID
        for (freq_name, freq) in freq_specs(en)
            for (cv_name, cv) in CV_GRID
                sev = lognormal_from_mean_cv(1.0, cv)
                @printf("\n=== Referência MC: E[N]=%.0f  freq=%s  CV=%s ===\n", en, freq_name, cv_name)
                ref_sample, t_ref = mc_reference(freq, sev)
                sort!(ref_sample)  # ordena 1 vez; ks_tail reaproveita via presorted/ref_presorted
                ve_ref = var_es(ref_sample; alpha = ALPHA)
                @printf("  %d amostras, %.1fs, VaR99.9%%=%.4f ES99.9%%=%.4f\n",
                        length(ref_sample), t_ref, ve_ref.var, ve_ref.es)
                push!(ref_rows, (en = en, freq = freq_name, cv_nome = cv_name,
                                  n_ref = length(ref_sample), tempo_ref_s = t_ref,
                                  ref_var = ve_ref.var, ref_es = ve_ref.es))

                for m in M_GRID
                    cell_seed += 1
                    cells = run_cell(en, freq_name, freq, cv_name, cv, m, sev,
                                      ref_sample, ve_ref.var, ve_ref.es, cell_seed)
                    append!(all_rows, cells)
                    for r in cells
                        @printf("    m=%-6d %-14s VaR=%.3f (erro %+.2f%%) ES=%.3f (erro %+.2f%%) KS=%.4f t=%.4fs%s\n",
                                m, r.metodo, r.var, 100r.erro_rel_var, r.es, 100r.erro_rel_es, r.ks, r.tempo_s,
                                r.underflow ? "  [UNDERFLOW CORRIGIDO]" : "")
                    end
                end
            end
        end
    end

    # --- exporta CSVs ---
    open(joinpath(OUT, "grid_referencias.csv"), "w") do io
        println(io, "en,freq,cv_nome,n_ref,tempo_ref_s,ref_var,ref_es")
        for r in ref_rows
            println(io, join([r.en, r.freq, r.cv_nome, r.n_ref, r.tempo_ref_s, r.ref_var, r.ref_es], ","))
        end
    end

    open(joinpath(OUT, "grid_resultados.csv"), "w") do io
        println(io, "en,freq,cv_nome,cv,m,h,domain,sev_mean_discretizado,ref_var,ref_es,metodo,var,es,erro_rel_var,erro_rel_es,ks,tempo_s,underflow,massa_truncada")
        for r in all_rows
            println(io, join([r.en, r.freq, r.cv_nome, r.cv, r.m, r.h, r.domain, r.sev_mean_discretizado,
                               r.ref_var, r.ref_es, r.metodo, r.var, r.es, r.erro_rel_var, r.erro_rel_es,
                               r.ks, r.tempo_s, r.underflow, r.massa_truncada], ","))
        end
    end

    @printf("\n\n%d linhas de referência, %d linhas de resultado exportadas em %s\n",
            length(ref_rows), length(all_rows), OUT)
end

main()
