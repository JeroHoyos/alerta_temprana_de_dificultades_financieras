"""Paleta, fuentes y tamaños comunes a todos los gráficos del proyecto."""

import matplotlib.pyplot as plt
import seaborn as sns

from .utils import FIGURES_DIR

# Paleta: azul para lo que se quiere resaltar, gris azulado para el contexto, naranja para alertar
MAIN = "#2b5c8f"
MUTED = "#cfd8e6"
ALERT = "#d95f02"
NEUTRAL = "#555555"
DARK = "#333333"
EDGE = "#1f3b5c"
CMAP = "Blues"

RC_PARAMS = {
    "figure.dpi": 110,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "axes.labelsize": 11,
    "legend.frameon": False,
}


def apply_style() -> None:
    """Aplica el estilo común a todas las gráficas que se creen después."""
    plt.rcParams.update(RC_PARAMS)


def save_figure(fig, name: str) -> None:
    """Guarda la figura en figures/ con la resolución del póster."""
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / name, dpi=200, bbox_inches="tight")


def highlight_last(n: int) -> list:
    """Colores con la última barra resaltada (la mayor, si las barras van de menor a mayor)."""
    return [MUTED] * (n - 1) + [MAIN]


def highlight_above(values, threshold: float) -> list:
    """Naranja para las barras por encima del umbral y gris azulado para el resto."""
    return [ALERT if v > threshold else MUTED for v in values]


def pct_count_labels(pcts, counts) -> list:
    """Etiquetas del tipo '12.3 %  (1,234)'."""
    return [f"{p:.1f} %  ({n:,})" for p, n in zip(pcts, counts)]


def labeled_barh(ax, values, labels, colors, room: float = 1.3) -> None:
    """Barras horizontales con una etiqueta al final de cada una."""
    bars = ax.barh(values.index.astype(str), values.values, color=colors)
    ax.bar_label(bars, labels=labels, padding=4, fontsize=10)
    # Espacio a la derecha para que las etiquetas no se corten
    ax.set_xlim(0, values.max() * room)
    ax.grid(axis="x", alpha=0.3)


def confusion_heatmap(ax, counts, class_names) -> None:
    """Matriz de confusión con el conteo y el % sobre cada clase real; los aciertos llevan borde."""
    pct = counts / counts.sum(axis=1, keepdims=True)
    cells = [f"{n:,}\n({p:.1%})" for n, p in zip(counts.ravel(), pct.ravel())]
    # El color usa el % por clase real, así todas las matrices se comparan en la misma escala
    sns.heatmap(
        pct, vmin=0, vmax=1, cmap=CMAP, cbar=False, ax=ax,
        annot=[cells[:2], cells[2:]], fmt="", annot_kws={"fontsize": 13},
        linewidths=2, linecolor="white",
        xticklabels=class_names, yticklabels=class_names,
    )
    ax.grid(False)
    for k in range(2):
        ax.add_patch(plt.Rectangle((k, k), 1, 1, fill=False, edgecolor=EDGE, lw=2.5))
