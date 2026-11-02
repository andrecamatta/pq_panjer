# prepare_data.jl
#
# Calibra a frequência de eventos de inadimplência a partir do SCR.data do
# Banco Central (sistema de informações de crédito), seguindo a estrutura do
# CreditRisk+: a frequência de defaults varia ENTRE setores (CNAE) dentro de
# um mesmo mês, por causa de um fator sistemático setorial, não dentro de um
# único setor ao longo do tempo. Usamos um corte transversal (um mês) com
# muitas células (UF x CNAE), agregadas por CNAE, para estimar essa variação
# entre setores.
#
# Segmento escolhido: cliente = PJ (pessoa jurídica), modalidade =
# "Empréstimos", todas as UF e portes agregados por CNAE. Esse recorte tem
# volume suficiente de operações para a maioria dos setores não cair no
# sigilo de células com poucas operações, e representa um produto de crédito
# com significado econômico direto (empréstimo a empresa).
#
# Dado: número de operações (quando não suprimido por sigilo, ou seja,
# numero_de_operacoes >= 0) e saldo vencido acima de 90 dias / carteira ativa
# por célula. Não há, nos dados públicos, uma contagem direta de "número de
# operações que entraram em default no mês": aproximamos o número de
# operações em default no período por
#   N_default(cnae) ≈ numero_de_operacoes(cnae) × (vencido_90(cnae) / carteira_ativa(cnae))
# assumindo que as operações inadimplentes têm, em média, o mesmo tamanho das
# operações da carteira daquele CNAE. É uma aproximação declarada, não uma
# contagem de sinistros no sentido atuarial estrito, mas calibrada em dado
# público real, não inventada.

using Statistics, Printf

const RAW = joinpath(@__DIR__, "data", "raw")
const OUT = joinpath(@__DIR__, "data", "output")
mkpath(RAW); mkpath(OUT)

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
    n_lido = 0
    n_usado = 0
    open(path, "r") do io
        header = readline(io)  # descarta cabeçalho
        for line in eachline(io)
            n_lido += 1
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
            n_usado += 1
            prev = get(agg, cnae, (n_ops = 0.0, vencido90 = 0.0, carteira_ativa = 0.0))
            agg[cnae] = (n_ops = prev.n_ops + n_ops,
                         vencido90 = prev.vencido90 + vencido90,
                         carteira_ativa = prev.carteira_ativa + carteira_ativa)
        end
    end
    @printf("  %s: %d linhas lidas, %d usadas (cliente=%s, modalidade=%s, numero_de_operacoes não suprimido)\n",
            basename(path), n_lido, n_usado, cliente, modalidade)
    return agg
end

function main()
    meses = ["202503", "202506", "202509", "202512"]  # trimestres de 2025, para checar estabilidade entre meses
    all_rows = NamedTuple[]
    for mes in meses
        csvpath = joinpath(RAW, "scrdata_$(mes).csv")
        isfile(csvpath) || error("Arquivo não encontrado: $csvpath. Rode a etapa de download antes.")
        agg = aggregate_by_cnae(csvpath)
        for (cnae, v) in agg
            v.n_ops < 30 && continue  # exige amostra mínima de operações no CNAE para a aproximação ter sentido
            taxa = v.vencido90 / v.carteira_ativa
            n_default = v.n_ops * taxa
            push!(all_rows, (mes = mes, cnae = cnae, n_ops = v.n_ops, carteira_ativa = v.carteira_ativa,
                              vencido90 = v.vencido90, taxa_inadimplencia = taxa, n_default_aprox = n_default))
        end
    end

    outpath = joinpath(OUT, "scr_frequencia_setorial.csv")
    open(outpath, "w") do io
        println(io, "mes,cnae,n_operacoes,carteira_ativa,vencido_90,taxa_inadimplencia,n_default_aprox")
        for r in all_rows
            println(io, join([r.mes, "\"$(r.cnae)\"", r.n_ops, r.carteira_ativa, r.vencido90,
                               r.taxa_inadimplencia, r.n_default_aprox], ","))
        end
    end
    @printf("\n%d linhas (mês x CNAE) exportadas em %s\n", length(all_rows), outpath)

    # Resumo rápido do último mês (referência principal da calibração)
    ult = filter(r -> r.mes == meses[end], all_rows)
    nd = [r.n_default_aprox for r in ult]
    @printf("\n=== Resumo %s (n=%d setores CNAE) ===\n", meses[end], length(nd))
    @printf("N_default aproximado por setor: média=%.2f, variância=%.2f, razão var/média=%.2f\n",
            mean(nd), var(nd), var(nd) / mean(nd))
    @printf("min=%.2f, mediana=%.2f, max=%.2f\n", minimum(nd), median(nd), maximum(nd))
    taxa = [r.taxa_inadimplencia for r in ult]
    @printf("Taxa de inadimplência (vencido>90d / carteira ativa) por setor: média=%.4f, desvio=%.4f\n",
            mean(taxa), std(taxa))
end

main()
