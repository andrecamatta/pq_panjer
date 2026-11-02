# test_sanity.jl — valida Panjer, FFT e Monte Carlo entre si num caso simples
# antes de rodar a grade fatorial completa.

include(joinpath(@__DIR__, "src", "AggregateLoss.jl"))
using .AggregateLoss
using Distributions
using Random
using Printf

function lognormal_from_mean_cv(mean_, cv)
    sigma2 = log(1 + cv^2)
    mu = log(mean_) - sigma2 / 2
    return LogNormal(mu, sqrt(sigma2))
end

function main()
    freq = PoissonFreq(10.0)
    sev = lognormal_from_mean_cv(1.0, 0.324)
    h = 0.05
    m = 4000

    f = discretize_severity(sev, h, m)
    @printf("Severidade discretizada: soma das massas = %.6f (deve ser ≈ 1)\n", sum(f))

    g_panjer, underflow = panjer_recursion(freq, f)
    @printf("Panjer: soma = %.6f, underflow reportado = %s\n", sum(g_panjer), underflow)

    g_fft = fft_aggregate(freq, sev, h, m; tilt = 0.0, pad_factor = 2)
    @printf("FFT (sem tilt): soma = %.6f\n", sum(g_fft))

    g_fft_tilt = fft_aggregate(freq, sev, h, m; auto_tilt = true, pad_factor = 2)
    @printf("FFT (com tilt): soma = %.6f\n", sum(g_fft_tilt))

    rng = MersenneTwister(42)
    mc = monte_carlo_aggregate(freq, sev, 2_000_000; rng = rng)

    ve_panjer = var_es(g_panjer, h)
    ve_fft = var_es(g_fft, h)
    ve_fft_tilt = var_es(g_fft_tilt, h)
    ve_mc = var_es(mc)

    @printf("\nVaR99.9%% / ES99.9%%:\n")
    @printf("  Panjer:        VaR=%.4f  ES=%.4f\n", ve_panjer.var, ve_panjer.es)
    @printf("  FFT sem tilt:  VaR=%.4f  ES=%.4f\n", ve_fft.var, ve_fft.es)
    @printf("  FFT com tilt:  VaR=%.4f  ES=%.4f\n", ve_fft_tilt.var, ve_fft_tilt.es)
    @printf("  Monte Carlo:   VaR=%.4f  ES=%.4f  (n=%d)\n", ve_mc.var, ve_mc.es, length(mc))

    @printf("\nMédia teórica E[S] = E[N]*E[X] = %.4f; médias simuladas/discretizadas:\n", 10.0 * 1.0)
    @printf("  Panjer: %.4f   FFT: %.4f   FFT+tilt: %.4f   MC: %.4f\n",
            sum((0:m-1) .* h .* g_panjer), sum((0:m-1) .* h .* g_fft),
            sum((0:m-1) .* h .* g_fft_tilt), mean(mc))
end

main()
