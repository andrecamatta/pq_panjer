# pq_panjer

Código de apoio ao artigo "Atuária para quants, parte 2: a recursão de Panjer e a FFT para a perda agregada de crédito", da publicação [Pílulas de Quant](https://pilulasdequant.com.br).

O repositório calcula a distribuição de perdas agregadas de uma carteira de crédito (frequência de defaults × severidade) por três vias, a recursão de Panjer, a FFT e o Monte Carlo, calibradas em dado público do Banco Central, e compara as três numa grade fatorial de cenários.

## O que o código faz

`src/AggregateLoss.jl` é o módulo numérico central. Implementa a classe (a,b,0) de distribuições de frequência (Poisson e binomial negativa), a discretização da severidade pelo método do arredondamento, f₀ = F(h/2) e fₖ = F(kh+h/2) − F(kh−h/2) para k ≥ 1, e a recursão de Panjer propriamente dita:

```
g₀ = P_N(f₀)
gₖ = [1/(1 − a·f₀)] · Σⱼ₌₁ᵏ (a + b·j/k)·fⱼ·gₖ₋ⱼ,  k ≥ 1
```

onde P_N é a função geradora de probabilidade da frequência e (a,b) dependem da distribuição escolhida: (0, λ) para Poisson, (1−p, (r−1)(1−p)) para a binomial negativa com média μ e parâmetro de forma r (p = r/(r+μ)). A condição inicial correta é g₀ = P_N(f₀), coincidindo com P(N=0) apenas no caso particular f₀ = 0. O módulo também reescala a recursão para lidar com o *underflow* de e^(−λ) quando λ é grande, implementa a mesma conta via FFT (com *zero-padding* e *exponential tilting* contra *aliasing*), roda Monte Carlo vetorizado, e calcula VaR, Expected Shortfall e a distância de Kolmogorov-Smirnov restrita à cauda. `var_es_partial` entrega o mesmo VaR e ES de uma amostra de Monte Carlo por seleção parcial (`partialsort!` com um intervalo, não um índice único), bem mais barato que ordenar a amostra inteira quando o quantil de interesse está perto da borda, o caso do VaR 99,9%.

`prepare_data.jl` lê os arquivos mensais do SCR.data do Banco Central, filtra pessoa jurídica e modalidade Empréstimos, exclui células suprimidas por sigilo bancário (até 15 operações) e agrega por setor da Classificação Nacional de Atividades Econômicas (CNAE), aproximando o número de eventos de default por setor como numero_de_operações × (vencido_acima_de_90_dias / carteira_ativa). Grava `data/output/scr_frequencia_setorial.csv`.

`calibrate_frequency.jl` parte desse corte setorial para estimar, pelo método dos momentos, o parâmetro de forma r da binomial negativa (parametrização Var = μ + μ²/r) que melhor descreve a sobredispersão observada entre setores, repetindo a estimativa em quatro trimestres de 2025 para checar estabilidade dentro de um único ano.

`calibrate_frequency_multiyear.jl` estende essa mesma estimativa, com o mesmo recorte e método, a cada ano completo de 2012 a 2025 (a série do SCR.data começa em julho de 2012). É a fonte do r efetivamente usado na grade principal, a mediana entre os r anuais, r≈0,542. Os dois extremos da série (r mínimo e r máximo observados) alimentam `run_grid_robustness.jl`. Lê os arquivos trimestrais já presentes em `data/raw/` (não baixa nada) e grava `data/output/nb_dispersion_r_multiyear.csv`, com o r médio e a faixa entre trimestres de cada ano.

`calibrate_severity.jl` deriva uma grade de coeficientes de variação (CV) para a severidade a partir do LGD bimodal do Working Paper 193 do Banco Central, tratado como uma mistura de dois pontos com peso igual.

`run_grid.jl` roda a grade fatorial completa: E[N] ∈ {1, 10, 100, 1.000}, Poisson e binomial negativa (r=0,542, mediana de `calibrate_frequency_multiyear.jl`), três níveis de CV de severidade e três tamanhos de grade de discretização, comparando Panjer, FFT (com e sem *tilting*) e Monte Carlo contra uma referência de Monte Carlo de 10⁷ simulações por cenário. Grava `data/output/grid_resultados.csv` e `data/output/grid_referencias.csv`.

`run_grid_robustness.jl` repete o subconjunto binomial negativa dessa grade (E[N] ∈ {100, 1.000}, os três níveis de CV, m=10.000) nos dois extremos observados na série 2012-2025, para checar se as conclusões da grade principal (calibrada na mediana) se mantêm fora dela. Recebe o rótulo e o valor de r por linha de comando (`julia run_grid_robustness.jl r_min 0.413` ou `r_max 0.720`). Cada chamada acrescenta linhas a `data/output/grid_robustez_r.csv` e `data/output/grid_robustez_r_referencias.csv`, e o arquivo é recriado só na chamada com rótulo `r_min`.

`measure_reference_noise.jl` quantifica o ruído de amostragem da própria referência de Monte Carlo (10⁷ simulações, 5 sementes) usada em `run_grid.jl`: para cada uma das 24 combinações de E[N], frequência e CV, roda as mesmas 5 sementes isoladamente e reporta o coeficiente de variação do VaR 99,9% entre elas. Grava `data/output/ruido_referencia.csv`.

`mc_cost_comparison.jl` mede, cenário a cenário, quanto tempo o Monte Carlo precisa para igualar o erro que Panjer e a FFT entregam em m=10.000, restrito aos 16 (de 24) cenários em que h já fica abaixo de 0,3×DP da severidade nessa grade. A referência de erro não é mais o Monte Carlo de 10⁷ simulações (que carrega ruído de amostragem da mesma ordem do erro de Panjer nesses cenários bem discretizados), e sim uma FFT determinística em m=1.000.000, sem ruído de amostragem. Para cada cenário, roda o Monte Carlo com n crescente (10⁴ a 10⁷) e procura o primeiro n cujo erro cruza abaixo do de Panjer. Se isso não acontece até 10⁷, marca o cenário como piso, um limite inferior de tempo, não um cruzamento real. Grava `data/output/mc_custo_precisao.csv` (todas as linhas de n) e `data/output/mc_custo_precisao_resumo.csv` (um cruzamento ou piso por cenário).

`mc_tolerance_cost.jl` mede, para os mesmos 16 cenários e a mesma referência de alta resolução de `mc_cost_comparison.jl`, o erro esperado do Monte Carlo em função do número de simulações, para depois converter isso em tempo até tolerâncias de erro do VaR 99,9% que uma aplicação de capital poderia de fato exigir (1%, 0,5%, 0,1%), em vez do erro real (abaixo de 0,01%) de Panjer e da FFT. Para cada cenário, varre uma grade log-espaçada de n entre 10 mil e 10 milhões de simulações, com várias sementes por n (de 20, nos tamanhos menores, a 2, nos maiores, por custo computacional) para estimar o erro esperado via RMSE entre sementes. Recebe o E[N] por linha de comando (`julia mc_tolerance_cost.jl 1`, `10`, `100` ou `1000`, cada chamada mais cara que a anterior). A primeira (E[N]=1) recria os CSVs, as demais acrescentam linhas. Grava `data/output/mc_tolerance_sweep.csv` (erro e tempo de simulação por n, sem a extração do quantil).

`remeasure_panjer_fft_timing.jl` remede o tempo de Panjer e da FFT em m=10.000, nos mesmos 16 cenários, com aquecimento da compilação (uma chamada descartada antes de cronometrar) e mediana de 7 execuções por cenário, num único ambiente. Cada tempo cronometrado inclui a discretização da severidade, a recursão ou a transformada, e a extração do quantil (`var_es`), o mesmo trabalho de ponta a ponta que o Monte Carlo faz. Sobrescreve `data/output/mc_tolerance_pontual.csv` e grava `data/output/mc_quantile_overhead.csv`, o tempo da extração do quantil isolada por tamanho de amostra (via `var_es_partial`, seleção parcial), usado para somar esse custo ao tempo do Monte Carlo já medido em `mc_tolerance_cost.jl`, sem precisar re-rodar a simulação.

`mc_tolerance_analysis.jl` converte a varredura de erro (`mc_tolerance_sweep.csv`), o tempo da extração do quantil (`mc_quantile_overhead.csv`) e os tempos remedidos de Panjer e da FFT (`mc_tolerance_pontual.csv`) no resultado final por tolerância. Ajusta uma reta em escala log-log a erro(n) e a tempo(n) por cenário (sem dependências externas, mínimos quadrados manual), interpola o n e o tempo necessários para cada tolerância, e calcula a razão desse tempo contra Panjer e contra a FFT. Quando o cruzamento previsto cai fora da grade testada, o resultado é travado na borda mais próxima (10 mil ou 10 milhões) e marcado como extrapolação ou piso, não um cruzamento medido diretamente. Grava `data/output/mc_tolerance_resultado.csv`.

`test_sanity.jl` roda um cenário único como checagem rápida de consistência entre Panjer, FFT e Monte Carlo antes de qualquer execução mais longa.

`test/runtests.jl` é a suíte formal (Test.jl): roda `test_sanity.jl`, confere que as massas de severidade e das distribuições agregadas somam 1, que Panjer, FFT com e sem *tilting* e Monte Carlo dão o mesmo VaR e ES 99,9% num cenário simples, e que `var_es_partial` bate com `var_es` (ordenação completa) em amostras de vários tamanhos e níveis de confiança. Roda com `julia --project=. -e 'using Pkg; Pkg.test()'`.

## Como reproduzir

```bash
julia --project=. -e '
using Pkg
Pkg.instantiate()'
julia --project=. prepare_data.jl
julia --project=. calibrate_frequency.jl
julia --project=. calibrate_severity.jl
julia --project=. run_grid.jl
julia --project=. measure_reference_noise.jl
julia --project=. mc_cost_comparison.jl
julia --project=. mc_tolerance_cost.jl 1
julia --project=. mc_tolerance_cost.jl 10
julia --project=. mc_tolerance_cost.jl 100
julia --project=. mc_tolerance_cost.jl 1000
julia --project=. remeasure_panjer_fft_timing.jl
julia --project=. mc_tolerance_analysis.jl
```

Os arquivos do SCR.data não são baixados automaticamente. Em [dadosabertos.bcb.gov.br](https://dadosabertos.bcb.gov.br) ou diretamente em `https://www.bcb.gov.br/pda/desig/scrdata_<ano>.zip`, baixe o ZIP anual de 2025, descompacte e copie os quatro arquivos trimestrais (`scrdata_202503.csv`, `scrdata_202506.csv`, `scrdata_202509.csv`, `scrdata_202512.csv`) para `data/raw/`. `prepare_data.jl` para com erro se algum deles faltar.

Para reproduzir também a checagem de robustez do r ao longo do tempo, baixe da mesma forma os ZIPs anuais de 2012 a 2024 e copie os trimestres disponíveis de cada ano para `data/raw/` (2012 só tem setembro e dezembro, porque a série começa em julho daquele ano), depois rode:

```bash
julia --project=. calibrate_frequency_multiyear.jl
julia --project=. run_grid_robustness.jl r_min 0.413
julia --project=. run_grid_robustness.jl r_max 0.720
```

`prepare_data.jl` e `calibrate_frequency.jl` rodam em segundos. `run_grid.jl`, `mc_cost_comparison.jl` e `measure_reference_noise.jl` são scripts mais longos, da ordem de minutos cada, porque incluem referências de Monte Carlo de milhões de simulações por cenário. `calibrate_frequency_multiyear.jl` roda em segundos; `run_grid_robustness.jl` é da mesma ordem de minutos que `run_grid.jl`, pelo mesmo motivo. `mc_tolerance_cost.jl` é o mais longo de todos: a chamada com E[N]=1 roda em cerca de 1 a 2 minutos, mas o custo cresce com E[N], e a chamada com E[N]=1.000 sozinha leva uns 5 minutos. `remeasure_panjer_fft_timing.jl` e `mc_tolerance_analysis.jl` rodam em menos de um minuto cada.

Cada script grava os CSVs intermediários e finais em `data/output/`. Os CSVs de resultado usados no artigo estão versionados nessa pasta; os arquivos brutos do SCR.data em `data/raw/` não. `src/AggregateLoss.jl` é incluído (`include`) pelos demais scripts, como módulo de apoio.

## Fonte e limitações

A frequência vem do SCR.data, sistema de informações de crédito do Banco Central, segmento pessoa jurídica, modalidade Empréstimos, agregado por setor CNAE, trimestre a trimestre, de 2012 a 2025. O SCR.data não registra uma contagem direta de eventos de default: o número usado aqui é uma aproximação, construída a partir do número de operações e da taxa de inadimplência de cada setor, assumindo que as operações inadimplentes têm, em média, o tamanho médio das operações daquele setor. O parâmetro de forma r da binomial negativa usado na grade principal é a mediana entre os 14 anos com dado disponível (r≈0,542). Os extremos da série, r≈0,413 em 2021 e r≈0,720 em 2025, são testados à parte em `run_grid_robustness.jl`. Esse r é estimado numa escala de milhares de operações por setor e aplicado, na grade de teste, também a cenários sintéticos de E[N] bem menores, uma extrapolação explícita, não uma medição direta nessas escalas.

A severidade não tem microdado público no Brasil, por desenho de sigilo bancário. A grade de CV usada aqui é paramétrica (severidade lognormal), ancorada no LGD bimodal do Working Paper 193 do Banco Central (operações prefixadas não varejo, 2003–2007), não em perdas observadas por operação. Qualquer leitura numérica deste código vale para a amostra, o período e o método descritos acima, não como propriedade geral de Panjer, FFT ou Monte Carlo fora desse escopo.

## Referências centrais

PANJER, H. H. Recursive evaluation of a family of compound distributions. *ASTIN Bulletin*, v. 12, n. 1, 1981.

KLUGMAN, S. A., PANJER, H. H. e WILLMOT, G. E. *Loss Models: From Data to Decisions*. 5. ed. Wiley, 2019.

GERHOLD, S., SCHMOCK, U. e WARNUNG, R. A generalization of Panjer's recursion and numerically stable risk aggregation. *Finance and Stochastics*, v. 14, n. 1, 2010.

EMBRECHTS, P. e FREI, M. Panjer recursion versus FFT for compound distributions. *Mathematical Methods of Operations Research*, v. 69, 2009.

CREDIT SUISSE FINANCIAL PRODUCTS. *CreditRisk+: A Credit Risk Management Framework*. Technical Document. Londres, 1997.
