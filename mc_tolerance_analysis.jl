# mc_tolerance_analysis.jl
#
# Converte a varredura de erro do Monte Carlo por n (mc_tolerance_sweep.csv,
# gerado por mc_tolerance_cost.jl) e os tempos remedidos de Panjer e da FFT
# (mc_tolerance_pontual.csv, gerado por remeasure_panjer_fft_timing.jl) no
# resultado final por tolerância de erro do VaR 99,9% (1%, 0,5%, 0,1%):
# o número de simulações e o tempo que o Monte Carlo precisa para atingir
# cada tolerância, e a razão desse tempo contra Panjer e contra a FFT.
#
# O tempo do Monte Carlo somado aqui inclui a extração do quantil (var_es
# sobre a amostra ordenada), medida à parte em mc_quantile_overhead.csv,
# porque o tempo cronometrado em mc_tolerance_cost.jl cobriu só
# monte_carlo_aggregate — sem isso, a comparação favorecia o Monte Carlo
# artificialmente, já que Panjer e a FFT (remedidos nesta revisão) já
# incluem a extração do quantil no próprio tempo. Reaproveita a varredura
# de erro já rodada, sem resimular.
#
# Erro(n) e tempo(n) são bem aproximados por uma reta em escala log-log em
# cada cenário (ver R² impresso abaixo), o que permite interpolar o n e o
# tempo necessários para cada tolerância. Quando o cruzamento previsto cai
# fora da grade testada (10⁴ a 10⁷), o resultado é travado na borda mais
# próxima e marcado como extrapolação (abaixo de 10⁴) ou piso (acima de
# 10⁷, sem convergência observada) — nenhum dos dois é um cruzamento medido
# diretamente.

using Statistics, Printf

const OUT = joinpath(@__DIR__, "data", "output")
const TOLS = [1.0, 0.5, 0.1]
const N_MIN, N_MAX = 1.0e4, 1.0e7

function read_csv(path)
    rows = NamedTuple[]
    header = String[]
    open(path, "r") do io
        header = split(readline(io), ',')
        for line in eachline(io)
            cols = split(line, ',')
            push!(rows, NamedTuple(Symbol(h) => parse(Float64, c) for (h, c) in zip(header, cols)))
        end
    end
    return rows
end

function read_csv_mixed(path, strcols)
    rows = NamedTuple[]
    header = String[]
    open(path, "r") do io
        header = split(readline(io), ',')
        for line in eachline(io)
            cols = split(line, ',')
            vals = [Symbol(h) in strcols ? c : parse(Float64, c) for (h, c) in zip(header, cols)]
            push!(rows, NamedTuple(Symbol(h) => v for (h, v) in zip(header, vals)))
        end
    end
    return rows
end

"Ajuste de reta y = slope*x + intercept por mínimos quadrados (sem dependências externas)."
function linfit(x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    mx, my = mean(x), mean(y)
    slope = sum((x .- mx) .* (y .- my)) / sum((x .- mx) .^ 2)
    intercept = my - slope * mx
    pred = slope .* x .+ intercept
    r2 = 1 - sum((y .- pred) .^ 2) / sum((y .- my) .^ 2)
    return (slope = slope, intercept = intercept, r2 = r2)
end

function main()
    sweep = read_csv_mixed(joinpath(OUT, "mc_tolerance_sweep.csv"), [:freq, :cv_nome])
    pont = read_csv_mixed(joinpath(OUT, "mc_tolerance_pontual.csv"), [:freq, :cv_nome])
    overhead = read_csv(joinpath(OUT, "mc_quantile_overhead.csv"))
    overhead_by_n = Dict(round(Int, r.n) => r.tempo_quantil_s for r in overhead)

    cenarios = unique([(r.en, r.freq, r.cv_nome) for r in sweep])
    result_rows = NamedTuple[]

    println("=== Ajuste log-log de erro(n) e tempo(n), por cenário ===")
    for (en, freq, cv) in cenarios
        srows = sort(filter(r -> r.en == en && r.freq == freq && r.cv_nome == cv, sweep), by = r -> r.n)
        prow = only(filter(r -> r.en == en && r.freq == freq && r.cv_nome == cv, pont))

        ns = [r.n for r in srows]
        rmse = [r.rmse_pct for r in srows]
        # tempo "justo": simulação (já medida) + extração do quantil (medida à parte)
        tempo_total = [r.tempo_mediana_s + overhead_by_n[round(Int, r.n)] for r in srows]

        fit_r = linfit(log10.(ns), log10.(rmse))
        fit_t = linfit(log10.(ns), log10.(tempo_total))
        @printf("EN=%-6.0f %-7s %-6s  erro: slope=%+.3f R2=%.3f | tempo: slope=%+.3f R2=%.3f\n",
                en, freq, cv, fit_r.slope, fit_r.r2, fit_t.slope, fit_t.r2)

        for tol in TOLS
            log_n_cross = (log10(tol) - fit_r.intercept) / fit_r.slope
            n_cross_raw = 10^log_n_cross
            extrapolado_abaixo = n_cross_raw < N_MIN
            piso = n_cross_raw > N_MAX
            log_n_eval = clamp(log_n_cross, log10(N_MIN), log10(N_MAX))
            n_cross = 10^log_n_eval
            tempo_cross = 10^(fit_t.slope * log_n_eval + fit_t.intercept)
            razao_panjer = tempo_cross / prow.tempo_panjer_s
            razao_fft = tempo_cross / prow.tempo_fft_s
            push!(result_rows, (en = en, freq = freq, cv_nome = cv, tolerance = tol,
                                 n_cross_raw = n_cross_raw, n_cross = n_cross, tempo_cross_s = tempo_cross,
                                 razao_panjer = razao_panjer, razao_fft = razao_fft,
                                 piso = piso, extrapolado_abaixo = extrapolado_abaixo))
        end
    end

    open(joinpath(OUT, "mc_tolerance_resultado.csv"), "w") do io
        println(io, "en,freq,cv_nome,tolerance,n_cross_raw,n_cross,tempo_cross_s,razao_panjer,razao_fft,piso,extrapolado_abaixo")
        for r in result_rows
            println(io, join([r.en, r.freq, r.cv_nome, r.tolerance, r.n_cross_raw, r.n_cross, r.tempo_cross_s,
                               r.razao_panjer, r.razao_fft, r.piso, r.extrapolado_abaixo], ","))
        end
    end

    println("\n=== Resumo por tolerância (mediana e faixa, 16 cenários) ===")
    for tol in TOLS
        sub = filter(r -> r.tolerance == tol, result_rows)
        rp = [r.razao_panjer for r in sub]
        rf = [r.razao_fft for r in sub]
        n_mc_mais_rapido_panjer = count(<(1.0), rp)
        n_mc_mais_rapido_fft = count(<(1.0), rf)
        n_piso = count(r -> r.piso, sub)
        n_extrap = count(r -> r.extrapolado_abaixo, sub)
        @printf("tol=%.1f%%: razao_panjer med=%.3f [%.3f,%.3f] MCmaisrapido=%d/16 | razao_fft med=%.3f [%.3f,%.3f] MCmaisrapido=%d/16 | piso=%d extrap_abaixo=%d\n",
                tol, median(rp), minimum(rp), maximum(rp), n_mc_mais_rapido_panjer,
                median(rf), minimum(rf), maximum(rf), n_mc_mais_rapido_fft, n_piso, n_extrap)
    end

    total_pairs = length(result_rows)
    fft_wins = count(r -> r.razao_fft > 1.0, result_rows)
    @printf("\nFFT mais rápida que o MC em %d de %d combinações (cenário × tolerância)\n", fft_wins, total_pairs)
end

main()
