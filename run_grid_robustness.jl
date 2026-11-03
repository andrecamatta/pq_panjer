# run_grid_robustness.jl
#
# Checa se as conclusões da grade principal (run_grid.jl), calibrada com a
# mediana anual do r da binomial negativa (r≈0,542, 2012-2025, ver
# calibrate_frequency_multiyear.jl), se mantêm nos dois extremos observados
# nessa mesma série: r mínimo (maior dispersão setorial, 2021) e r máximo
# (menor dispersão setorial, 2025).
#
# Escopo: binomial negativa, E[N] ∈ {100, 1.000}, os três níveis de CV da
# severidade, grade de m=10.000 pontos, mesma metodologia de referência
# (Monte Carlo de 10⁷ simulações, 5 sementes) e mesmas métricas do teste
# principal.
#
# Uso: julia --project=. run_grid_robustness.jl <label> <r>
#   julia --project=. run_grid_robustness.jl r_min 0.413
#   julia --project=. run_grid_robustness.jl r_max 0.720
# Cada chamada acrescenta linhas a data/output/grid_robustez_r.csv; o
# arquivo é truncado e recriado só quando o rótulo é "r_min", para permitir
# rodar os dois extremos em invocações separadas sem duplicar linhas.

include(joinpath(@__DIR__, "src", "AggregateLoss.jl"))
using .AggregateLoss
using Distributions, Random, Statistics, Printf

const OUT = joinpath(@__DIR__, "data", "output")

function lognormal_from_mean_cv(mean_::Float64, cv::Float64)
    sigma2 = log(1 + cv^2)
    mu = log(mean_) - sigma2 / 2
    return LogNormal(mu, sqrt(sigma2))
end

const EN_GRID = [100.0, 1000.0]
const CV_GRID = [("baixo", 0.1619), ("medio", 0.3237), ("alto", 0.6475)]
const M = 10_000
const ALPHA = 0.999
const P_LOWER_KS = 0.99

function mc_reference(freq, sev; nsim = 2_000_000, nseeds = 5, base_seed = 5000)
    chunks = Vector{Vector{Float64}}(undef, nseeds)
    t = @elapsed for s in 1:nseeds
        rng = MersenneTwister(base_seed + s)
        chunks[s] = monte_carlo_aggregate(freq, sev, nsim; rng = rng)
    end
    pooled = vcat(chunks...)
    return pooled, t
end

function main()
    length(ARGS) == 2 || error("uso: julia run_grid_robustness.jl <label> <r>")
    label = ARGS[1]
    r = parse(Float64, ARGS[2])

    all_rows = NamedTuple[]
    ref_rows = NamedTuple[]
    cell_seed = label == "r_min" ? 9000 : 9100

    for en in EN_GRID
        freq = NegBinFreq(en, r)
        for (cv_name, cv) in CV_GRID
            sev = lognormal_from_mean_cv(1.0, cv)
            @printf("\n=== %s r=%.3f E[N]=%.0f CV=%s ===\n", label, r, en, cv_name)
            ref_sample, t_ref = mc_reference(freq, sev)
            sort!(ref_sample)
            ve_ref = var_es(ref_sample; alpha = ALPHA)
            @printf("  referência: %d amostras, %.1fs, VaR99.9%%=%.4f ES99.9%%=%.4f\n",
                    length(ref_sample), t_ref, ve_ref.var, ve_ref.es)
            push!(ref_rows, (r_label = label, r = r, en = en, cv_nome = cv_name, n_ref = length(ref_sample),
                              tempo_ref_s = t_ref, ref_var = ve_ref.var, ref_es = ve_ref.es))

            domain = suggest_domain(freq, sev)
            h = domain / M
            f = max.(discretize_severity(sev, h, M), 0.0)

            t_panjer = @elapsed (g_panjer, underflow) = panjer_recursion(freq, f)
            ve_panjer = var_es(g_panjer, h; alpha = ALPHA)
            ks_panjer = ks_tail(g_panjer, h, ref_sample; p_lower = P_LOWER_KS, presorted = true)
            erro_panjer = (ve_panjer.var - ve_ref.var) / ve_ref.var
            @printf("  Panjer/FFT (m=%d, h=%.3f, domínio=%.1f): VaR=%.3f (erro %+.2f%%) KS=%.4f\n",
                    M, h, domain, ve_panjer.var, 100erro_panjer, ks_panjer)
            push!(all_rows, (r_label = label, r = r, en = en, cv_nome = cv_name, cv = cv, m = M, h = h, domain = domain,
                              ref_var = ve_ref.var, ref_es = ve_ref.es, metodo = "panjer",
                              var = ve_panjer.var, es = ve_panjer.es,
                              erro_rel_var = erro_panjer, erro_rel_es = (ve_panjer.es - ve_ref.es) / ve_ref.es,
                              ks = ks_panjer, tempo_s = t_panjer, underflow = underflow))

            for (lbl, nsim) in (("mc_1e4", 10_000), ("mc_1e5", 100_000))
                cell_seed += 1
                rng = MersenneTwister(cell_seed)
                t_mc = @elapsed mc_sample = monte_carlo_aggregate(freq, sev, nsim; rng = rng)
                ve_mc = var_es(mc_sample; alpha = ALPHA)
                ks_mc = ks_tail(mc_sample, ref_sample; p_lower = P_LOWER_KS, ref_presorted = true)
                erro_mc = (ve_mc.var - ve_ref.var) / ve_ref.var
                @printf("  %s: VaR=%.3f (erro %+.2f%%) KS=%.4f t=%.4fs\n",
                        lbl, ve_mc.var, 100erro_mc, ks_mc, t_mc)
                push!(all_rows, (r_label = label, r = r, en = en, cv_nome = cv_name, cv = cv, m = M, h = h, domain = domain,
                                  ref_var = ve_ref.var, ref_es = ve_ref.es, metodo = lbl,
                                  var = ve_mc.var, es = ve_mc.es,
                                  erro_rel_var = erro_mc, erro_rel_es = (ve_mc.es - ve_ref.es) / ve_ref.es,
                                  ks = ks_mc, tempo_s = t_mc, underflow = false))
            end
        end
    end

    resultados_path = joinpath(OUT, "grid_robustez_r.csv")
    referencias_path = joinpath(OUT, "grid_robustez_r_referencias.csv")
    fresh = label == "r_min" || !isfile(resultados_path)

    open(resultados_path, fresh ? "w" : "a") do io
        fresh && println(io, "r_label,r,en,cv_nome,cv,m,h,domain,ref_var,ref_es,metodo,var,es,erro_rel_var,erro_rel_es,ks,tempo_s,underflow")
        for r_ in all_rows
            println(io, join([r_.r_label, r_.r, r_.en, r_.cv_nome, r_.cv, r_.m, r_.h, r_.domain, r_.ref_var, r_.ref_es,
                               r_.metodo, r_.var, r_.es, r_.erro_rel_var, r_.erro_rel_es, r_.ks, r_.tempo_s, r_.underflow], ","))
        end
    end
    open(referencias_path, fresh ? "w" : "a") do io
        fresh && println(io, "r_label,r,en,cv_nome,n_ref,tempo_ref_s,ref_var,ref_es")
        for r_ in ref_rows
            println(io, join([r_.r_label, r_.r, r_.en, r_.cv_nome, r_.n_ref, r_.tempo_ref_s, r_.ref_var, r_.ref_es], ","))
        end
    end
    @printf("\n%d linhas exportadas (rótulo %s) em grid_robustez_r.csv\n", length(all_rows), label)
end

main()
