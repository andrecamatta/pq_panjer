# pq_panjer

Código de apoio ao artigo "Atuária para quants, parte 2: quando a recursão de Panjer vence o Monte Carlo, e quando não", da publicação [Pílulas de Quant](https://pilulasdequant.com.br).

O repositório calcula a distribuição de perdas agregadas de uma carteira de crédito (frequência de defaults × severidade) por três vias, a recursão de Panjer, a FFT e o Monte Carlo, calibradas em dado público do Banco Central, e compara as três numa grade fatorial de cenários.

## O que o código faz

`src/AggregateLoss.jl` é o módulo numérico central. Implementa a classe (a,b,0) de distribuições de frequência (Poisson e binomial negativa), a discretização da severidade pelo método do arredondamento, f₀ = F(h/2) e fₖ = F(kh+h/2) − F(kh−h/2) para k ≥ 1, e a recursão de Panjer propriamente dita:

```
g₀ = P_N(f₀)
gₖ = [1/(1 − a·f₀)] · Σⱼ₌₁ᵏ (a + b·j/k)·fⱼ·gₖ₋ⱼ,  k ≥ 1
```

onde P_N é a função geradora de probabilidade da frequência e (a,b) dependem da distribuição escolhida: (0, λ) para Poisson, (1−p, (r−1)(1−p)) para a binomial negativa com média μ e parâmetro de forma r (p = r/(r+μ)). A condição inicial correta é g₀ = P_N(f₀), coincidindo com P(N=0) apenas no caso particular f₀ = 0. O módulo também reescala a recursão para lidar com o *underflow* de e^(−λ) quando λ é grande, implementa a mesma conta via FFT (com *zero-padding* e *exponential tilting* contra *aliasing*), roda Monte Carlo vetorizado, e calcula VaR, Expected Shortfall e a distância de Kolmogorov-Smirnov restrita à cauda.

`prepare_data.jl` lê os arquivos mensais do SCR.data do Banco Central, filtra pessoa jurídica e modalidade Empréstimos, exclui células suprimidas por sigilo bancário (até 15 operações) e agrega por setor da Classificação Nacional de Atividades Econômicas (CNAE), aproximando o número de eventos de default por setor como numero_de_operações × (vencido_acima_de_90_dias / carteira_ativa). Grava `data/output/scr_frequencia_setorial.csv`.

`calibrate_frequency.jl` parte desse corte setorial para estimar, pelo método dos momentos, o parâmetro de forma r da binomial negativa (parametrização Var = μ + μ²/r) que melhor descreve a sobredispersão observada entre setores, repetindo a estimativa em quatro trimestres de 2025 para checar estabilidade.

`calibrate_severity.jl` deriva uma grade de coeficientes de variação (CV) para a severidade a partir do LGD bimodal do Working Paper 193 do Banco Central, tratado como uma mistura de dois pontos com peso igual.

`run_grid.jl` roda a grade fatorial completa: E[N] ∈ {1, 10, 100, 1.000}, Poisson e binomial negativa, três níveis de CV de severidade e três tamanhos de grade de discretização, comparando Panjer, FFT (com e sem *tilting*) e Monte Carlo contra uma referência de Monte Carlo de 10⁷ simulações por cenário. Grava `data/output/grid_resultados.csv` e `data/output/grid_referencias.csv`.

`mc_cost_comparison.jl` mede, para os mesmos cenários de referência, quanto tempo o Monte Carlo precisa para chegar perto da precisão que Panjer e a FFT entregam, rodando o Monte Carlo com número crescente de simulações e registrando erro e tempo de execução em `data/output/mc_custo_precisao.csv`.

`test_sanity.jl` roda um cenário único como checagem rápida de consistência entre Panjer, FFT e Monte Carlo antes de qualquer execução mais longa.

`make_figures.py` e `capture_table.py` são utilitários em Python (matplotlib e Selenium) que geram as figuras e as tabelas em PNG usadas no artigo a partir dos CSVs em `data/output/`. Não fazem parte do pipeline de cálculo.

## Como reproduzir

```bash
julia --project=. -e '
using Pkg
Pkg.instantiate()'
julia --project=. prepare_data.jl
julia --project=. calibrate_frequency.jl
julia --project=. calibrate_severity.jl
julia --project=. run_grid.jl
julia --project=. mc_cost_comparison.jl
```

Os arquivos do SCR.data não são baixados automaticamente. Em [dadosabertos.bcb.gov.br](https://dadosabertos.bcb.gov.br) ou diretamente em `https://www.bcb.gov.br/pda/desig/scrdata_<ano>.zip`, baixe o ZIP anual de 2025, descompacte e copie os quatro arquivos trimestrais (`scrdata_202503.csv`, `scrdata_202506.csv`, `scrdata_202509.csv`, `scrdata_202512.csv`) para `data/raw/`. `prepare_data.jl` para com erro se algum deles faltar.

`prepare_data.jl` e `calibrate_frequency.jl` rodam em segundos. `run_grid.jl` e `mc_cost_comparison.jl` são os scripts mais longos, da ordem de minutos cada, porque incluem referências de Monte Carlo de milhões de simulações por cenário.

Cada script grava os CSVs intermediários e finais em `data/output/` (não versionado). `src/AggregateLoss.jl` é incluído (`include`) pelos demais scripts, como módulo de apoio.

## Fonte e limitações

A frequência vem do SCR.data, sistema de informações de crédito do Banco Central, segmento pessoa jurídica, modalidade Empréstimos, agregado por setor CNAE em quatro trimestres de 2025. O SCR.data não registra uma contagem direta de eventos de default: o número usado aqui é uma aproximação, construída a partir do número de operações e da taxa de inadimplência de cada setor, assumindo que as operações inadimplentes têm, em média, o tamanho médio das operações daquele setor. O parâmetro de forma r da binomial negativa é estimado numa escala de milhares de operações por setor e aplicado, na grade de teste, também a cenários sintéticos de E[N] bem menores, uma extrapolação explícita, não uma medição direta nessas escalas.

A severidade não tem microdado público no Brasil, por desenho de sigilo bancário. A grade de CV usada aqui é paramétrica (severidade lognormal), ancorada no LGD bimodal do Working Paper 193 do Banco Central (operações prefixadas não varejo, 2003–2007), não em perdas observadas por operação. Qualquer leitura numérica deste código vale para a amostra, o período e o método descritos acima, não como propriedade geral de Panjer, FFT ou Monte Carlo fora desse escopo.

## Referências centrais

PANJER, H. H. Recursive evaluation of a family of compound distributions. *ASTIN Bulletin*, v. 12, n. 1, 1981.

KLUGMAN, S. A., PANJER, H. H. e WILLMOT, G. E. *Loss Models: From Data to Decisions*. 5. ed. Wiley, 2019.

GERHOLD, S., SCHMOCK, U. e WARNUNG, R. A generalization of Panjer's recursion and numerically stable risk aggregation. *Finance and Stochastics*, v. 14, n. 1, 2010.

EMBRECHTS, P. e FREI, M. Panjer recursion versus FFT for compound distributions. *Mathematical Methods of Operations Research*, v. 69, 2009.

CREDIT SUISSE FINANCIAL PRODUCTS. *CreditRisk+: A Credit Risk Management Framework*. Technical Document. Londres, 1997.
