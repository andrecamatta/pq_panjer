module AggregateLoss

using Distributions
using FFTW
using Random
using Statistics

export FrequencySpec, PoissonFreq, NegBinFreq, pgf_a_b,
       discretize_severity, panjer_recursion, fft_aggregate, monte_carlo_aggregate,
       var_es, ks_tail, compound_moments, suggest_domain

# ---------------------------------------------------------------------------
# Especificação da frequência: classe (a,b,0) de Panjer (1981)
# ---------------------------------------------------------------------------

abstract type FrequencySpec end

struct PoissonFreq <: FrequencySpec
    lambda::Float64
end

struct NegBinFreq <: FrequencySpec
    r::Float64      # parâmetro de forma (tamanho), Var = μ + μ²/r
    p::Float64      # parâmetro de probabilidade da NB em Distributions.jl
    mu::Float64
end

"Constrói uma NegBinFreq com média μ e parâmetro de forma r dados (Var = μ + μ²/r)."
function NegBinFreq(mu::Float64, r::Float64)
    p = r / (r + mu)  # Distributions.jl: NegativeBinomial(r, p), média = r(1-p)/p
    return NegBinFreq(r, p, mu)
end

"Parâmetros (a,b) da recursão de Panjer, g_k = (a + b/k) f_k g_{k-1} em k=0 não se aplica; ver panjer_recursion."
function pgf_a_b(freq::PoissonFreq)
    return (a = 0.0, b = freq.lambda)
end

function pgf_a_b(freq::NegBinFreq)
    a = 1.0 - freq.p
    b = (freq.r - 1.0) * (1.0 - freq.p)
    return (a = a, b = b)
end

"""
    g0_pgf(freq, z) -> valor de alta precisão da PGF de `freq` em `z`

Condição inicial correta da recursão de Panjer: g_0 = P(S=0) = P_N(f_0), a
função geradora de probabilidade (PGF) de N avaliada em f_0 = P(severidade
discretizada = 0), NÃO simplesmente P(N=0). As duas coincidem apenas quando
f_0 = 0 (severidade sem massa em zero). Como o método do arredondamento usado
em `discretize_severity` produz f_0 > 0 sempre que a largura `h` não é
pequena face à escala da severidade — o caso típico quando a frequência
esperada é grande e o domínio precisa ser largo —, usar P(N=0) nesse regime
SUBESTIMA g_0 e distorce toda a recursão (sintoma observado: a soma de g
fica bem abaixo de 1 mesmo com domínio adequado, não por truncamento real,
mas por essa condição inicial errada).
"""
function g0_pgf(freq::PoissonFreq, z::Float64)
    arg = freq.lambda * (z - 1.0)  # ≤ 0
    arg > -700.0 && return exp(arg)
    return exp(big(arg))  # evita underflow total; BigFloat preserva o valor
end

function g0_pgf(freq::NegBinFreq, z::Float64)
    return (freq.p / (1.0 - (1.0 - freq.p) * z))^freq.r
end

Distributions.mean(freq::PoissonFreq) = freq.lambda
Distributions.mean(freq::NegBinFreq) = freq.mu

sampler(freq::PoissonFreq) = Distributions.Poisson(freq.lambda)
sampler(freq::NegBinFreq) = Distributions.NegativeBinomial(freq.r, freq.p)

freq_variance(freq::PoissonFreq) = freq.lambda
freq_variance(freq::NegBinFreq) = freq.mu + freq.mu^2 / freq.r

"""
    compound_moments(freq, sev_dist) -> (mean=..., sd=...)

Média e desvio padrão da perda agregada S = Σ X_i, N ~ freq, X_i ~ sev_dist
iid, pelas fórmulas padrão: E[S] = E[N]E[X], Var(S) = E[N]Var(X) + Var(N)E[X]².
Usado para dimensionar o domínio de discretização ANTES de rodar Panjer/FFT
(ver `suggest_domain`), não como resultado numérico do teste.
"""
function compound_moments(freq::FrequencySpec, sev_dist)
    exn, varn = Distributions.mean(freq), freq_variance(freq)
    exx, varx = Distributions.mean(sev_dist), Distributions.var(sev_dist)
    mean_s = exn * exx
    var_s = exn * varx + varn * exx^2
    return (mean = mean_s, sd = sqrt(var_s))
end

"""
    suggest_domain(freq, sev_dist; k=15) -> Float64

Limite superior sugerido para o domínio de discretização (E[S] + k·SD(S)),
com margem generosa (`k=15` por padrão) para que o quantil 99,9% e a cauda
usada no KS fiquem bem dentro do grid, evitando truncar massa real — um erro
de dimensionamento distinto do aliasing da FFT ou do underflow do Panjer,
mas que também corrompe VaR/ES se não for verificado.
"""
function suggest_domain(freq::FrequencySpec, sev_dist; k::Float64 = 15.0)
    mom = compound_moments(freq, sev_dist)
    return mom.mean + k * mom.sd
end

# ---------------------------------------------------------------------------
# Discretização da severidade (método do arredondamento, Klugman-Panjer-Willmot)
# ---------------------------------------------------------------------------

"""
    discretize_severity(sev_dist, h, m) -> Vector{Float64}

Discretiza a severidade em m pontos de largura h pelo método do
arredondamento de massa (mass dispersal / rounding): a massa de
probabilidade de cada intervalo [k·h - h/2, k·h + h/2) é atribuída ao ponto
k·h, com o ponto k=0 recebendo também a massa de (-∞, h/2). A massa que cai
acima de (m-1)·h + h/2 é truncada e reportada (erro de truncamento).
"""
function discretize_severity(sev_dist, h::Float64, m::Int)
    f = zeros(Float64, m)
    f[1] = cdf(sev_dist, h / 2)
    for k in 1:(m - 2)
        f[k + 1] = cdf(sev_dist, (k + 0.5) * h) - cdf(sev_dist, (k - 0.5) * h)
    end
    f[m] = 1.0 - cdf(sev_dist, (m - 1.5) * h)  # absorve a massa truncada da cauda no último ponto
    return f
end

# ---------------------------------------------------------------------------
# Recursão de Panjer, com rescale para evitar underflow de g_0 = P(N=0)
# ---------------------------------------------------------------------------

"""
    panjer_recursion(freq, f; rescale=true) -> Vector{Float64}

Recursão de Panjer (1981) para a classe (a,b,0):
    g_0 = P(N=0)                    (ou P(N=0) ajustado por composição se f_0 > 0)
    g_n = [1/(1 - a f_0)] Σ_{j=1}^{n} (a + b j/n) f_j g_{n-j},  n ≥ 1

`f` é a severidade discretizada (f[1] = f_0, ..., f[m] = f_{m-1}). Retorna a
distribuição de probabilidade agregada discretizada no mesmo grid.

Quando a frequência esperada é grande (ex.: Poisson com λ ≳ 745), g_0 = e^{-λ}
já sofre underflow de REPRESENTAÇÃO em Float64 (vira exatamente 0.0, não só
"muito pequeno"): multiplicar o resultado por esse valor zeraria toda a
recursão. A saída, quando `rescale=true` (padrão), não depende de conhecer o
valor absoluto de g_0: a recursão é linear e homogênea em g_0, então rodar a
partir de g_0_work=1 produz um vetor proporcional ao resultado verdadeiro
por um fator constante 1/g_0_true. Mas essa proporção pode, ela mesma,
exceder o maior Float64 representável (~1,8e308) quando λ é muito grande (ex.:
λ=1000 já dá g_0_true ~ 1e-434, fora até do alcance do BigFloat default):
rodar a recursão toda em g_0_work=1 faria valores intermediários passarem por
Inf antes de qualquer renormalização final conseguir corrigir. A mitigação é
reescalonar PERIODICAMENTE durante a própria recursão (o mesmo truque usado
no algoritmo forward de HMM): sempre que o valor corrente excede um limiar
seguro, toda a parte já computada do vetor é dividida por um fator grande, e
o processo continua. Como a divisão é aplicada uniformemente a tudo que já
foi calculado, a FORMA relativa da distribuição não se altera — só a escala,
que de qualquer forma será descartada na renormalização final por soma. Essa
é a mitigação mais simples citada na literatura para a classe (a,b,0) pura
(Poisson, binomial negativa); GERHOLD, SCHMOCK e WARNUNG (2010) tratam o
caso mais geral de classes estendidas/compostas, fora do escopo testado aqui.
"""
function panjer_recursion(freq::FrequencySpec, f::Vector{Float64}; rescale::Bool = true)
    m = length(f)
    a, b = pgf_a_b(freq)
    g0_true = g0_pgf(freq, f[1])

    underflowed = g0_true < 1e-300
    g0_work = (underflowed && rescale) ? 1.0 : Float64(g0_true)

    g = zeros(Float64, m)
    g[1] = g0_work
    denom = 1.0 - a * f[1]
    n_rescales = 0
    for n in 1:(m - 1)
        s = 0.0
        @inbounds for j in 1:n
            s += (a + b * j / n) * f[j + 1] * g[n - j + 1]
        end
        g[n + 1] = s / denom

        # Reescalona em AMBAS as direções, a cada passo em que o valor corrente sai
        # de uma faixa segura — não só quando cresce demais. Um recorte que só
        # trata overflow (valor grande) deixa passar o problema simétrico: a
        # recursão parte de g_0=1 artificial (bem acima do g_0 verdadeiro, que
        # underflowou), cresce até um pico e depois PRECISA decair de volta para
        # representar a cauda — e esse decaimento pode underflowar para 0.0 exato
        # ANTES do fim do grid. Uma vez em 0.0 exato, a informação da região onde a
        # massa realmente deveria se concentrar (perto da média) se perde; o que
        # sobra é só um resíduo residual vindo de g[1] (a semente original,
        # sempre não-nula) combinado com a cauda da própria severidade — o que
        # aparece, depois de normalizar, como massa artificialmente empilhada na
        # borda do domínio. Reescalonar o valor corrente de volta para perto de 1
        # a cada passo evita os dois extremos simetricamente.
        if underflowed && rescale && g[n + 1] > 0.0 && (g[n + 1] > 1e100 || g[n + 1] < 1e-100)
            factor = 1.0 / g[n + 1]
            @inbounds for k in 1:(n + 1)
                g[k] *= factor
            end
            n_rescales += 1
        end
    end

    if underflowed && rescale
        # renormaliza pela própria soma: evita multiplicar por g0_true (que
        # pode ser 0.0 em Float64 por underflow de representação, não só a
        # recursão). sum(g) aqui já está na mesma escala de 1/g0_true (após
        # os reescalonamentos periódicos, que não mudam a forma relativa).
        s = sum(g)
        if !isfinite(s) || s <= 0.0
            error("panjer_recursion: underflow não corrigido mesmo com reescalonamento periódico " *
                  "(n_rescales=$n_rescales); frequência/severidade/grid fora do alcance numérico testado.")
        end
        g ./= s
    end

    return g, underflowed
end

# ---------------------------------------------------------------------------
# Agregação via FFT, com zero-padding e exponential tilting (Esscher)
# ---------------------------------------------------------------------------

"Função geradora de probabilidade (PGF) da frequência, avaliada em z complexo."
function pgf(freq::PoissonFreq, z)
    return exp(freq.lambda * (z - 1))
end

function pgf(freq::NegBinFreq, z)
    return (freq.p / (1 - (1 - freq.p) * z))^freq.r
end

"""
    fft_aggregate(freq, sev_dist, h, m; tilt=0.0, auto_tilt=false, pad_factor=2) -> Vector{Float64}

Calcula a distribuição agregada discretizada via FFT: discretiza a
severidade num grid de `m * pad_factor` pontos (zero-padding para mitigar
aliasing), aplica opcionalmente *exponential tilting* (transformada de
Esscher) à severidade antes da FFT (desfeito no resultado final), transforma,
aplica a PGF da frequência ponto a ponto no domínio da frequência, e inverte.
Retorna só os primeiros `m` pontos (a parte de interesse; o restante é o
espaço de padding).

O parâmetro de tilt é sempre definido em relação ao DOMÍNIO COMPLETO (`n·h`,
já incluindo o padding), não ao domínio útil (`m·h`): a regra prática citada
por SHEVCHENKO (2010, via Grübel e Hermesmeier) — produto entre o limite do
domínio e o parâmetro de tilting próximo de 20 — se refere ao domínio sobre o
qual a FFT de fato roda. Se `tilt` não for informado e `auto_tilt=true`, usa
`tilt = 20 / (n·h)` automaticamente. Usar um tilt maior que isso em relação a
`n·h` amplifica ruído de ponto flutuante no último bin da severidade
discretizada (que absorve a massa de cauda truncada e pode ser um número
muito pequeno, por cancelamento numérico em `1 - cdf(...)`, inclusive
ligeiramente negativo) por um fator exponencial, corrompendo o resultado.
"""
function fft_aggregate(freq::FrequencySpec, sev_dist, h::Float64, m::Int;
                        tilt::Float64 = 0.0, auto_tilt::Bool = false, pad_factor::Int = 2)
    n = m * pad_factor
    f = max.(discretize_severity(sev_dist, h, n), 0.0)  # zera ruído de cancelamento no bin de cauda

    theta = tilt
    if theta == 0.0 && auto_tilt
        theta = 20.0 / (n * h)
    end

    if theta != 0.0
        # fft(f .* pesos)[j] = Σ_k f_k e^{θ(k-1)h} e^{-2πi(k-1)j/n} = M_X(θ - 2πij/n),
        # a FGM de X avaliada no ponto complexo deslocado por θ. NÃO renormalizar
        # f.*pesos para somar 1 antes da FFT: dividir por Σf_k e^{θ(k-1)h} = M_X(θ)
        # aqui mudaria a escala ANTES de passar pela PGF (não-linear) da frequência,
        # dando um resultado diferente (e errado) do que tilting realmente significa.
        weights = [exp(theta * (k - 1) * h) for k in 1:n]
        fhat = fft(f .* weights)
    else
        fhat = fft(f)
    end

    ghat = pgf.(Ref(freq), fhat)
    g = real.(ifft(ghat))

    if theta != 0.0
        # ifft(ghat)[k] = g(k)·e^{θ(k-1)h} diretamente (mesma lógica acima, agora
        # para a composta S em vez da severidade X) — desfazer é só multiplicar
        # pelo peso inverso, sem nenhuma renormalização extra por uma constante
        # M_S(θ) separada (isso seria o erro de novo, só que do lado de saída).
        invweights = [exp(-theta * (k - 1) * h) for k in 1:n]
        g = g .* invweights
    end

    g = max.(g, 0.0)
    g ./= sum(g)
    return g[1:m]
end

# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

"""
    monte_carlo_aggregate(freq, sev_dist, nsim; rng=Random.default_rng()) -> Vector{Float64}

Simula `nsim` realizações da perda agregada S = Σ_{i=1}^{N} X_i, N ~ freq,
X_i ~ sev_dist iid, e retorna o vetor de `nsim` perdas simuladas.
"""
function monte_carlo_aggregate(freq::FrequencySpec, sev_dist, nsim::Int; rng = Random.default_rng())
    nd = sampler(freq)
    s = Vector{Float64}(undef, nsim)
    @inbounds for i in 1:nsim
        n = rand(rng, nd)
        acc = 0.0
        for _ in 1:n
            acc += rand(rng, sev_dist)
        end
        s[i] = acc
    end
    return s
end

# ---------------------------------------------------------------------------
# Métricas: VaR, ES e KS na cauda
# ---------------------------------------------------------------------------

"VaR e ES ao nível `alpha` (ex.: 0.999) a partir de uma distribuição discretizada (grid g, largura h)."
function var_es(g::Vector{Float64}, h::Float64; alpha::Float64 = 0.999)
    cum = cumsum(g)
    idx = findfirst(>=(alpha), cum)
    idx === nothing && (idx = length(g))
    var = (idx - 1) * h
    tail_mass = 1.0 - cum[idx] + g[idx]
    if tail_mass <= 0
        es = var
    else
        es_num = sum((k - 1) * h * g[k] for k in idx:length(g))
        es = es_num / sum(g[idx:end])
    end
    return (var = var, es = es)
end

"VaR e ES a partir de uma amostra de Monte Carlo (empírico)."
function var_es(sample::Vector{Float64}; alpha::Float64 = 0.999)
    srt = sort(sample)
    n = length(srt)
    idx = clamp(ceil(Int, alpha * n), 1, n)
    var = srt[idx]
    es = mean(srt[idx:end])
    return (var = var, es = es)
end

"""
    ks_tail(g, h, sample; p_lower=0.99) -> Float64

Distância de Kolmogorov-Smirnov entre a CDF discretizada `g` (grid de
largura h) e a CDF empírica de `sample`, restrita à cauda acima do
percentil `p_lower` da própria distribuição discretizada (ex.: p99).

As duas CDFs comparadas são as CDFs INCONDICIONAIS verdadeiras (escala 0 a
1, não uma versão renormalizada só dentro da cauda): na região de interesse
ambas já valem perto de `p_lower` (≈0,99), não perto de 0 — comparar versões
renormalizadas para [0,1] dentro da cauda infla artificialmente a distância
sempre para perto do máximo (1), mesmo quando os métodos batem bem.

Se `sample` já estiver ordenada, passe `presorted=true` para pular o
ordenamento (a amostra de referência, com 10⁷ pontos, não precisa ser
reordenada em toda chamada — o chamador ordena uma vez e reaproveita).
"""
function ks_tail(g::Vector{Float64}, h::Float64, sample::Vector{Float64}; p_lower::Float64 = 0.99, presorted::Bool = false)
    cum = cumsum(g)
    idx_lower = findfirst(>=(p_lower), cum)
    idx_lower === nothing && (idx_lower = 1)
    x_lower = (idx_lower - 1) * h

    sample_sorted = presorted ? sample : sort(sample)
    n_full = length(sample_sorted)
    tail_x = filter(>=(x_lower), sample_sorted)
    isempty(tail_x) && return NaN

    m = length(g)
    grid_x = [(k - 1) * h for k in 1:m]
    d = 0.0
    @inbounds for x in tail_x
        cdf_grid = x >= grid_x[end] ? 1.0 : cum[min(searchsortedlast(grid_x, x) + 1, m)]
        cdf_emp = searchsortedlast(sample_sorted, x) / n_full  # CDF incondicional verdadeira
        d = max(d, abs(cdf_grid - cdf_emp))
    end
    return d
end

"""
    ks_tail(sample, ref_sample; p_lower=0.99) -> Float64

Distância de Kolmogorov-Smirnov entre duas amostras (ex.: Monte Carlo rápido
contra a amostra de referência), restrita à cauda acima do percentil
`p_lower` DA AMOSTRA DE REFERÊNCIA — mesma convenção de CDFs incondicionais
da versão grid-vs-amostra, para que as duas métricas sejam comparáveis.

Se `ref_sample` já estiver ordenada, passe `ref_presorted=true` para pular
o ordenamento (ver nota de performance na outra assinatura de `ks_tail`).
"""
function ks_tail(sample::Vector{Float64}, ref_sample::Vector{Float64}; p_lower::Float64 = 0.99, ref_presorted::Bool = false)
    ref_sorted = ref_presorted ? ref_sample : sort(ref_sample)
    sample_sorted = sort(sample)
    n_ref, n_a = length(ref_sorted), length(sample_sorted)
    x_lower = ref_sorted[clamp(ceil(Int, p_lower * n_ref), 1, n_ref)]

    tail_pts = filter(>=(x_lower), vcat(ref_sorted, sample_sorted))
    isempty(tail_pts) && return NaN

    d = 0.0
    @inbounds for x in tail_pts
        cdf_a = searchsortedlast(sample_sorted, x) / n_a
        cdf_r = searchsortedlast(ref_sorted, x) / n_ref
        d = max(d, abs(cdf_a - cdf_r))
    end
    return d
end

end # module
