# calibrate_frequency_multiyear.jl
#
# Repete, para cada ano de 2012 a 2025, exatamente o mesmo recorte e o mesmo
# método usados em prepare_data.jl e calibrate_frequency.jl (corte
# transversal por CNAE, pessoa jurídica, modalidade Empréstimos, exclusão de
# células suprimidas por sigilo e da categoria "Não informados", mínimo de
# 30 operações por setor, estimador de momentos do parâmetro r da binomial
# negativa), para obter o parâmetro de dispersão r usado na grade fatorial
# principal (run_grid.jl): a mediana entre os r anuais de 2012 a 2025,
# r≈0,542. Os dois extremos da série (r mínimo e r máximo observados) são
# usados em run_grid_robustness.jl para checar se as conclusões da grade
# principal se mantêm fora da mediana.
#
# Usa os trimestres disponíveis de cada ano (março, junho, setembro,
# dezembro; 2012 só tem setembro e dezembro, porque a série do SCR.data
# começa em julho de 2012). Não baixa os dados: espera que os arquivos
# scrdata_<AAAAMM>.csv já estejam em data/raw/ (não versionados).

using Statistics, Printf

const RAW = joinpath(@__DIR__, "data", "raw")
const OUT = joinpath(@__DIR__, "data", "output")

# As duas funções abaixo são as mesmas de prepare_data.jl (parse_brnum e
# aggregate_by_cnae), repetidas aqui para que este script rode sozinho sobre
# qualquer mês disponível em data/raw/, sem acoplar ao main() daquele script.

"Converte string BR ('1.234,56' ou '1234,56') para Float64."
function parse_brnum(s::AbstractString)
    s = replace(s, "\"" => "")
    s = replace(s, "." => "")
    s = replace(s, "," => ".")
    return parse(Float64, s)
end

"Lê um CSV do SCR.data (campo 4 = cliente, campo 7 = modalidade) e agrega por CNAE (campo 5)."
function aggregate_by_cnae(path::AbstractString; cliente::String = "PJ", modalidade::String = "Empréstimos")
    agg = Dict{String,NamedTuple{(:n_ops, :vencido90, :carteira_ativa),Tuple{Float64,Float64,Float64}}}()
    open(path, "r") do io
        readline(io)  # descarta cabeçalho
        for line in eachline(io)
            cols = split(line, ';')
            length(cols) < 23 && continue
            cli = replace(cols[4], "\"" => "")
            mod = replace(cols[7], "\"" => "")
            (cli == cliente && mod == modalidade) || continue
            cnae = replace(cols[5], "\"" => "")
            n_ops = parse(Float64, replace(cols[11], "\"" => ""))
            n_ops < 0 && continue  # célula suprimida por sigilo (<= 15 operações)
            vencido90 = parse_brnum(cols[20])
            carteira_ativa = parse_brnum(cols[22])
            carteira_ativa <= 0 && continue
            prev = get(agg, cnae, (n_ops = 0.0, vencido90 = 0.0, carteira_ativa = 0.0))
            agg[cnae] = (n_ops = prev.n_ops + n_ops,
                         vencido90 = prev.vencido90 + vencido90,
                         carteira_ativa = prev.carteira_ativa + carteira_ativa)
        end
    end
    return agg
end

"Estimador de momentos do parâmetro r da binomial negativa (Var = μ + μ²/r)."
function nb_r_momentos(x::Vector{Float64})
    mu = mean(x)
    s2 = var(x)
    s2 <= mu && return (r = Inf, mu = mu, s2 = s2)
    r = mu^2 / (s2 - mu)
    return (r = r, mu = mu, s2 = s2)
end

function r_do_mes(csvpath::String)
    agg = aggregate_by_cnae(csvpath)
    nd = Float64[]
    for (cnae, v) in agg
        cnae == "Não informados" && continue
        v.n_ops < 30 && continue
        taxa = v.vencido90 / v.carteira_ativa
        push!(nd, v.n_ops * taxa)
    end
    return nb_r_momentos(nd)
end

function main()
    anos = 2012:2025
    out_rows = NamedTuple[]
    println("=== r da binomial negativa, por ano, mesmo recorte e método de calibrate_frequency.jl ===\n")

    for ano in anos
        rs_ano = Float64[]
        meses_usados = String[]
        for mm in ("03", "06", "09", "12")
            mes = "$(ano)$(mm)"
            csvpath = joinpath(RAW, "scrdata_$(mes).csv")
            isfile(csvpath) || continue
            est = r_do_mes(csvpath)
            if isfinite(est.r)
                push!(rs_ano, est.r)
                push!(meses_usados, mes)
            end
        end
        if isempty(rs_ano)
            @printf("%d: nenhum trimestre disponível ou sem sobredispersão detectável\n", ano)
            continue
        end
        r_ano = mean(rs_ano)
        @printf("%d: %d trimestre(s) (%s), r médio=%.3f (min=%.3f, max=%.3f)\n",
                ano, length(rs_ano), join(meses_usados, ","), r_ano, minimum(rs_ano), maximum(rs_ano))
        push!(out_rows, (ano = ano, n_trimestres = length(rs_ano), r_medio = r_ano,
                          r_min_trimestre = minimum(rs_ano), r_max_trimestre = maximum(rs_ano)))
    end

    rs_anuais = [r.r_medio for r in out_rows]
    @printf("\n=== Faixa de r entre os %d anos com dado disponível ===\n", length(rs_anuais))
    @printf("mínimo=%.3f (%d), máximo=%.3f (%d), média=%.3f, mediana=%.3f\n",
            minimum(rs_anuais), out_rows[argmin(rs_anuais)].ano,
            maximum(rs_anuais), out_rows[argmax(rs_anuais)].ano,
            mean(rs_anuais), median(rs_anuais))

    open(joinpath(OUT, "nb_dispersion_r_multiyear.csv"), "w") do io
        println(io, "ano,n_trimestres,r_medio,r_min_trimestre,r_max_trimestre")
        for r in out_rows
            println(io, join([r.ano, r.n_trimestres, r.r_medio, r.r_min_trimestre, r.r_max_trimestre], ","))
        end
    end
    @printf("\n%d linhas exportadas em data/output/nb_dispersion_r_multiyear.csv\n", length(out_rows))
end

main()
