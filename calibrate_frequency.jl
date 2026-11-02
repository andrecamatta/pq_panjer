# calibrate_frequency.jl
#
# A partir do corte transversal por CNAE (data/output/scr_frequencia_setorial.csv,
# produzido por prepare_data.jl), estima o grau de sobredispersão da frequência
# de "eventos de default" entre setores, para decidir o parâmetro de dispersão
# da binomial negativa (mistura Poisson-Gama, estilo CreditRisk+) usado na
# grade fatorial do teste Panjer/FFT/Monte Carlo.
#
# Não usamos os valores absolutos de n_default_aprox (que são agregados de
# um setor inteiro, da ordem de milhares) como os E[N] da grade — a grade usa
# E[N] ∈ {1,10,100,1000} como cenários sintéticos de carteiras de tamanhos
# diferentes. O que extraímos daqui é o parâmetro de FORMA r da binomial
# negativa (na parametrização Var = μ + μ²/r), que é adimensional em relação
# à escala de μ e pode ser transplantado para qualquer E[N] da grade. Esse r
# é a melhor estimativa disponível, a partir de dado público real, de quanto
# a incerteza sistemática setorial (o "fator Gama" do CreditRisk+) infla a
# variância da frequência de defaults além do que a Poisson prevê.

using Statistics, Printf

const OUT = joinpath(@__DIR__, "data", "output")

function read_csv(path)
    rows = NamedTuple[]
    open(path, "r") do io
        readline(io)  # header
        for line in eachline(io)
            # campos: mes,cnae,n_operacoes,carteira_ativa,vencido_90,taxa_inadimplencia,n_default_aprox
            # cnae está entre aspas e pode conter vírgulas -> parse manual
            m = match(r"^(\d+),\"(.*)\",([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)$", line)
            m === nothing && continue
            push!(rows, (mes = m.captures[1], cnae = m.captures[2],
                          n_operacoes = parse(Float64, m.captures[3]),
                          carteira_ativa = parse(Float64, m.captures[4]),
                          vencido_90 = parse(Float64, m.captures[5]),
                          taxa_inadimplencia = parse(Float64, m.captures[6]),
                          n_default_aprox = parse(Float64, m.captures[7])))
        end
    end
    return rows
end

"Estimador de momentos do parâmetro r da binomial negativa (Var = μ + μ²/r)."
function nb_r_momentos(x::Vector{Float64})
    mu = mean(x)
    s2 = var(x)
    s2 <= mu && return (r = Inf, mu = mu, s2 = s2)  # sem sobredispersão detectável -> Poisson já serve
    r = mu^2 / (s2 - mu)
    return (r = r, mu = mu, s2 = s2)
end

function main()
    rows = read_csv(joinpath(OUT, "scr_frequencia_setorial.csv"))
    # exclui "Não informados": não é setor econômico real, é resíduo de classificação,
    # e no mês 202512 vira um outlier extremo (célula com 67 operações, taxa = 1,0).
    rows = filter(r -> r.cnae != "Não informados", rows)

    meses = sort(unique(r.mes for r in rows))
    println("=== Sobredispersão da frequência de default implícita, por CNAE, por mês ===")
    println("(exclui \"Não informados\"; n = número de setores CNAE com amostra mínima)\n")

    rs = Float64[]
    for mes in meses
        sub = filter(r -> r.mes == mes, rows)
        nd = [r.n_default_aprox for r in sub]
        est = nb_r_momentos(nd)
        razao = est.s2 / est.mu
        @printf("%s: n=%d setores, média=%.1f, var/média=%.1f, r(BN)=%.3f\n",
                mes, length(sub), est.mu, razao, est.r)
        isfinite(est.r) && push!(rs, est.r)
    end

    r_pooled = mean(rs)
    @printf("\nr médio entre os %d meses observados: %.3f (mediana %.3f)\n",
            length(rs), r_pooled, median(rs))
    @printf("Interpretação: r pequeno = forte sobredispersão entre setores (o fator\n")
    @printf("sistemático setorial do CreditRisk+ domina); r → ∞ equivaleria à Poisson pura.\n")

    # arredonda para um valor de referência redondo, documentado como escolha, não como
    # resultado de precisão falsa (dado agregado e aproximação de N_default já discutidas)
    r_ref = round(r_pooled, sigdigits = 2)
    @printf("\nParâmetro de referência adotado na grade fatorial: r = %.2g\n", r_ref)
    @printf("(binomial negativa com mesma média da Poisson em cada cenário E[N], e Var = E[N] + E[N]²/r)\n")

    sep = ","
    open(joinpath(OUT, "nb_dispersion_r.txt"), "w") do io
        println(io, "r_ref=$(r_ref)")
        println(io, "r_por_mes=$(join(round.(rs, digits=3), sep))")
    end
end

main()
