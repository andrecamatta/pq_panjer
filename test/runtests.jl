# test/runtests.jl — suíte de testes do pq_panjer.
#
# Roda test_sanity.jl (checagem de consistência Panjer/FFT/MC num cenário
# simples) como smoke test e acrescenta testes formais (@test) de massa,
# consistência entre métodos e equivalência var_es vs var_es_partial.

using Test
using Random
using Statistics
using Distributions

include(joinpath(@__DIR__, "..", "src", "AggregateLoss.jl"))
using .AggregateLoss

function lognormal_from_mean_cv(mean_, cv)
    sigma2 = log(1 + cv^2)
    mu = log(mean_) - sigma2 / 2
    return LogNormal(mu, sqrt(sigma2))
end

@testset "pq_panjer" begin

    @testset "test_sanity.jl roda sem erro" begin
        sanity_path = joinpath(@__DIR__, "..", "test_sanity.jl")
        # Roda num módulo próprio: test_sanity.jl faz seu próprio include+using
        # de AggregateLoss.jl, que conflitaria com o `using .AggregateLoss`
        # deste arquivo se compartilhassem o namespace de Main.
        @test_nowarn (@eval module SanityCheck
            include($sanity_path)
        end)
    end

    @testset "massa e consistência Panjer/FFT/MC" begin
        freq = PoissonFreq(10.0)
        sev = lognormal_from_mean_cv(1.0, 0.324)
        h = 0.05
        m = 4000

        f = discretize_severity(sev, h, m)
        @test sum(f) ≈ 1.0 atol = 1e-6

        g_panjer, _ = panjer_recursion(freq, f)
        @test sum(g_panjer) ≈ 1.0 atol = 1e-6

        g_fft = fft_aggregate(freq, sev, h, m; tilt = 0.0, pad_factor = 2)
        @test sum(g_fft) ≈ 1.0 atol = 1e-6

        g_fft_tilt = fft_aggregate(freq, sev, h, m; auto_tilt = true, pad_factor = 2)
        @test sum(g_fft_tilt) ≈ 1.0 atol = 1e-6

        rng = MersenneTwister(42)
        mc = monte_carlo_aggregate(freq, sev, 2_000_000; rng = rng)

        ve_panjer = var_es(g_panjer, h)
        ve_fft = var_es(g_fft, h)
        ve_fft_tilt = var_es(g_fft_tilt, h)
        ve_mc = var_es(mc)

        # Panjer e FFT resolvem a mesma recursão (a,b,0); devem bater entre
        # si bem mais de perto do que qualquer um bate com o ruído do MC.
        @test ve_panjer.var ≈ ve_fft.var atol = 5 * h
        @test ve_panjer.es ≈ ve_fft.es atol = 5 * h
        @test ve_fft_tilt.var ≈ ve_fft.var atol = 5 * h

        # Monte Carlo com 2e6 simulações deve ficar na vizinhança do VaR/ES
        # determinístico, com folga generosa para o ruído amostral do quantil 99,9%.
        @test ve_mc.var ≈ ve_panjer.var atol = 0.15 * ve_panjer.var
        @test ve_mc.es ≈ ve_panjer.es atol = 0.15 * ve_panjer.es
    end

    @testset "var_es_partial bate com var_es" begin
        rng = MersenneTwister(123)
        for n in (100, 1_000, 50_000)
            for alpha in (0.95, 0.99, 0.999)
                sample = randn(rng, n) .^ 2 .* 10 .+ 1  # amostra assimétrica, não trivial
                ve_full = var_es(copy(sample); alpha = alpha)
                ve_partial = var_es_partial(copy(sample); alpha = alpha)
                @test ve_full.var == ve_partial.var
                @test ve_full.es ≈ ve_partial.es atol = 1e-12
            end
        end
    end

end
