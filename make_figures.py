import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/workspace/pq/artigos/panjer/pq_panjer/data/output"
HERE = "/workspace/pq/artigos/panjer/saida"

ORANGE = "#FF6719"
NAVY = "#1a1a1a"
BLUE = "#1565c0"
GRAY = "#8a8a8a"
GREEN = "#2e7d32"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

df = pd.read_csv(f"{OUT}/grid_resultados.csv")
df["h_sobre_sd"] = df["h"] / df["cv"]  # severidade tem média 1, logo SD = CV
df["abs_erro_var_pct"] = df["erro_rel_var"].abs() * 100
df["abs_erro_es_pct"] = df["erro_rel_es"].abs() * 100

# ---------------------------------------------------------------------------
# Figura 1: erro do VaR 99,9% (Panjer/FFT) vs largura de discretização h,
# medida em unidades do desvio-padrão da severidade — a variável que de fato
# comanda a distorção do método do arredondamento, não m ou h isolados.
# ---------------------------------------------------------------------------
panjer = df[df.metodo == "panjer"].copy()

fig, ax = plt.subplots(figsize=(8, 5.5))
for freq, marker, color in (("poisson", "o", BLUE), ("bn", "s", ORANGE)):
    sub = panjer[panjer.freq == freq]
    ax.scatter(sub["h_sobre_sd"], sub["abs_erro_var_pct"].clip(lower=1e-3),
               s=34, alpha=0.75, color=color, marker=marker,
               label="Poisson" if freq == "poisson" else "Binomial negativa (r=0,72)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.axvline(0.3, color=GRAY, linestyle="--", linewidth=1.2)
ax.text(0.33, 0.002, "h ≈ 0,3×DP(severidade)", fontsize=9, color=GRAY, rotation=90, va="bottom")
ax.set_xlabel("Largura de discretização h, em múltiplos do desvio-padrão da severidade", fontsize=11)
ax.set_ylabel("Erro absoluto do VaR 99,9% (%, escala log)", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, fontsize=10, loc="upper left")
fig.tight_layout()
fig.savefig(f"{HERE}/fig_erro_vs_bandwidth.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figura 2: tempo de execução vs tamanho da grade (m) — Panjer O(m²) contra
# FFT O(m log m). Monte Carlo não depende de m (depende de E[N] e do número
# de simulações); mostrado como faixa de referência horizontal, não como
# curva em função de m, para não sugerir uma dependência que não existe.
# ---------------------------------------------------------------------------
bem = df[df.h_sobre_sd < 0.3]
labels = {"panjer": "Panjer", "fft_sem_tilt": "FFT (sem tilt)", "fft_com_tilt": "FFT (com tilt)",
          "mc_1e4": "Monte Carlo 10⁴", "mc_1e5": "Monte Carlo 10⁵"}
colors = {"panjer": ORANGE, "fft_sem_tilt": BLUE, "fft_com_tilt": "#6a1b9a",
          "mc_1e4": GRAY, "mc_1e5": GREEN}
fig, ax = plt.subplots(figsize=(8, 5.5))
for metodo in ("panjer", "fft_sem_tilt"):
    sub = bem[bem.metodo == metodo].groupby("m")["tempo_s"].median().reset_index()
    ax.plot(sub["m"], sub["tempo_s"], marker="o", linewidth=2.2, color=colors[metodo], label=labels[metodo])
mc_1e4_range = bem[bem.metodo == "mc_1e4"]["tempo_s"]
mc_1e5_range = bem[bem.metodo == "mc_1e5"]["tempo_s"]
ax.axhspan(mc_1e4_range.min(), mc_1e4_range.max(), color=GRAY, alpha=0.18,
           label="Monte Carlo 10⁴ (faixa, não depende de m)")
ax.axhspan(mc_1e5_range.min(), mc_1e5_range.max(), color=GREEN, alpha=0.18,
           label="Monte Carlo 10⁵ (faixa, não depende de m)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Tamanho da grade (m, escala log)", fontsize=11)
ax.set_ylabel("Tempo de execução, mediana (s, escala log)", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(f"{HERE}/fig_tempo_vs_m.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figura 3: precisão (erro absoluto do VaR 99,9%) por método, só nos
# cenários bem discretizados (h/DP < 0,3) — comparação direta do que cada
# método entrega pelo mesmo cenário.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
ordem = ["panjer", "fft_sem_tilt", "mc_1e4", "mc_1e5"]
data = [bem[bem.metodo == m]["abs_erro_var_pct"].values for m in ordem]
bp = ax.boxplot(data, tick_labels=[labels[m] for m in ordem], patch_artist=True, showfliers=True)
for patch, m in zip(bp["boxes"], ordem):
    patch.set_facecolor(colors[m])
    patch.set_alpha(0.55)
ax.set_ylabel("Erro absoluto do VaR 99,9% (%)", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("Cenários com discretização adequada (h < 0,3×DP da severidade)", fontsize=11)
fig.tight_layout()
fig.savefig(f"{HERE}/fig_precisao_bem_discretizado.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figura 4: domínio necessário (escala log) por E[N] e tipo de frequência —
# mostra por que a binomial negativa calibrada no SCR.data (r=0,72) explode
# o domínio exigido em relação à Poisson de mesma média.
# ---------------------------------------------------------------------------
dom = df.drop_duplicates(subset=["en", "freq"])[["en", "freq", "domain"]]
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for freq, marker, color in (("poisson", "o", BLUE), ("bn", "s", ORANGE)):
    sub = dom[dom.freq == freq].sort_values("en")
    ax.plot(sub["en"], sub["domain"], marker=marker, linewidth=2.0, color=color,
            label="Poisson" if freq == "poisson" else "Binomial negativa (r=0,72)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("E[N] (frequência esperada)", fontsize=11)
ax.set_ylabel("Domínio de discretização necessário (unidades de perda, escala log)", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(f"{HERE}/fig_dominio_vs_en.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("figuras geradas em", HERE)
