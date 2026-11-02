# calibrate_severity.jl
#
# Não há microdado público de severidade por operação no Brasil (sigilo
# bancário, confirmado no research brief). A severidade é modelada por uma
# distribuição paramétrica (lognormal), com o COEFICIENTE DE VARIAÇÃO (CV)
# ancorado na única evidência pública concreta: o Working Paper 193 do BCB
# (SILVA; MARINS; NEVES, 2009), que estima LGD bimodal para operações
# prefixadas não-varejo, com média entre 47% e 92% segundo a metodologia.
#
# Tratamos esses dois números (47% e 92%) como os dois modos de uma mistura
# simples de dois pontos, com peso igual, só para extrair uma ORDEM DE
# GRANDEZA real de dispersão relativa (CV) a partir de dado publicado — não
# como uma reconstrução fiel da distribuição completa de LGD do paper, que
# não está disponível em microdado. Esse CV "médio" ancorado no WP193 é o
# ponto central da grade; "baixo" e "alto" são desvios simétricos (metade e
# dobro), para testar sensibilidade, não achados novos sobre LGD brasileiro.
#
# A média da severidade é normalizada para 1 unidade de perda (ν₀): como o
# teste compara métodos numéricos (Panjer/FFT/Monte Carlo) entre si, não
# estima um valor de capital real em reais, a escala monetária absoluta é
# irrelevante — o que importa para VaR/ES relativo e para o erro de
# discretização é a FORMA da distribuição (CV), não a unidade.

using Printf

const OUT = joinpath(@__DIR__, "data", "output")

function main()
    lgd_baixo, lgd_alto = 0.47, 0.92
    p = 0.5
    media_lgd = p * lgd_baixo + (1 - p) * lgd_alto
    var_lgd = p * (lgd_baixo - media_lgd)^2 + (1 - p) * (lgd_alto - media_lgd)^2
    cv_medio = sqrt(var_lgd) / media_lgd

    cv_baixo = cv_medio / 2
    cv_alto = cv_medio * 2

    @printf("LGD WP193 (BCB): modos em %.0f%% e %.0f%% (mistura 2 pontos, peso igual)\n", 100lgd_baixo, 100lgd_alto)
    @printf("média implícita = %.3f, desvio padrão implícito = %.3f, CV implícito = %.3f\n\n",
            media_lgd, sqrt(var_lgd), cv_medio)

    @printf("Grade de CV da severidade adotada no teste (lognormal, média normalizada em 1 unidade de perda):\n")
    @printf("  baixo  = %.3f (metade do CV ancorado no WP193)\n", cv_baixo)
    @printf("  médio  = %.3f (CV ancorado diretamente no WP193)\n", cv_medio)
    @printf("  alto   = %.3f (dobro do CV ancorado no WP193)\n", cv_alto)

    open(joinpath(OUT, "severity_cv_grid.txt"), "w") do io
        println(io, "cv_baixo=$(round(cv_baixo, digits=4))")
        println(io, "cv_medio=$(round(cv_medio, digits=4))")
        println(io, "cv_alto=$(round(cv_alto, digits=4))")
    end
end

main()
