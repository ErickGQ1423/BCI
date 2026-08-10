#!/usr/bin/env python3
"""Render a publication-friendly overview of the current online decoder."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


OUTPUT_DIR = Path("results")


COLORS = {
    "acq": "#DCEEFF",
    "model": "#E7F4E4",
    "control": "#FFE6C9",
    "decision": "#F8D7DA",
    "diag": "#E9DDF7",
    "output": "#FFF3BF",
    "recenter": "#D8F3DC",
    "neutral": "#F2F2F2",
}


def box(ax, x, y, w, h, text, color, fontsize=9, edge="#333333"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1f1f1f",
        zorder=4,
    )
    return (x, y, w, h)


def diamond(ax, cx, cy, w, h, text, color, fontsize=8.5):
    points = [
        (cx, cy + h / 2),
        (cx + w / 2, cy),
        (cx, cy - h / 2),
        (cx - w / 2, cy),
    ]
    patch = Polygon(
        points,
        closed=True,
        facecolor=color,
        edgecolor="#333333",
        linewidth=1.4,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, zorder=4)
    return (cx, cy, w, h)


def point(node, side):
    x, y, w, h = node
    if side == "top":
        return (x + w / 2, y + h)
    if side == "bottom":
        return (x + w / 2, y)
    if side == "left":
        return (x, y + h / 2)
    return (x + w, y + h / 2)


def diamond_point(node, side):
    cx, cy, w, h = node
    if side == "top":
        return (cx, cy + h / 2)
    if side == "bottom":
        return (cx, cy - h / 2)
    if side == "left":
        return (cx - w / 2, cy)
    return (cx + w / 2, cy)


def arrow(
    ax,
    start,
    end,
    label=None,
    color="#444444",
    style="-",
    connectionstyle="arc3,rad=0",
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.25,
        linestyle=style,
        color=color,
        connectionstyle=connectionstyle,
        zorder=2,
    )
    ax.add_patch(patch)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(
            mx,
            my + 0.12,
            label,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1),
            zorder=5,
        )


def main():
    fig, ax = plt.subplots(figsize=(24, 16))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 16)
    ax.axis("off")

    ax.text(
        12,
        15.55,
        "Arquitectura actual del decoder BCI online",
        ha="center",
        va="center",
        fontsize=22,
        weight="bold",
    )
    ax.text(
        12,
        15.12,
        "MDM controla · LDA/LR validan el fallback · cinco modelos se analizan en shadow",
        ha="center",
        fontsize=12,
        color="#444444",
    )

    # Acquisition row
    lsl = box(ax, 0.4, 13.8, 2.3, 0.75, "EEG por LSL\nStreamInlet", COLORS["acq"])
    channels = box(ax, 3.2, 13.8, 2.6, 0.75, "Selección\n9 canales", COLORS["acq"])
    filters = box(ax, 6.3, 13.8, 3.0, 0.75, "Notch 60 Hz\nBanda 0.1–1 Hz", COLORS["acq"])
    car = box(ax, 9.8, 13.8, 2.6, 0.75, "CAR + buffer\ncontinuo", COLORS["acq"])
    baseline = box(ax, 12.9, 13.8, 2.5, 0.75, "Baseline\n1 segundo", COLORS["acq"])
    m2 = box(ax, 15.9, 13.8, 3.0, 0.75, "Ventanas M2\ncada 250 ms", COLORS["acq"])
    quality = diamond(ax, 21.0, 14.18, 2.5, 1.1, "¿EEG\nválido?", COLORS["neutral"])
    bad = box(ax, 20.0, 12.7, 2.1, 0.62, "BAD_EEG\nomitir paso", COLORS["decision"], 8)

    for left, right in zip(
        [lsl, channels, filters, car, baseline],
        [channels, filters, car, baseline, m2],
    ):
        arrow(ax, point(left, "right"), point(right, "left"))
    arrow(ax, point(m2, "right"), diamond_point(quality, "left"))
    arrow(ax, diamond_point(quality, "bottom"), point(bad, "top"), "No")

    # Model bank
    features = box(ax, 10.6, 12.3, 3.0, 0.70, "Features del paso\nhasta −0.50 s", COLORS["neutral"])
    arrow(ax, diamond_point(quality, "left"), point(features, "right"), "Sí", connectionstyle="arc3,rad=-0.28")
    model_specs = [
        ("MDM\nRiemanniano", 1.0),
        ("LDA\nshrinkage", 5.0),
        ("LDA3\nFCz, C3, CP3", 9.0),
        ("Logistic\nRegression", 13.0),
        ("SVM", 17.0),
    ]
    models = []
    for label, x in model_specs:
        model = box(ax, x, 10.6, 3.0, 0.92, label + "\nP(MI)", COLORS["model"])
        models.append(model)
        arrow(ax, point(features, "bottom"), point(model, "top"), connectionstyle=f"arc3,rad={(x-9)/40:.2f}")

    mdm, lda, lda3, lr, svm = models

    # Real MDM control branch
    early = diamond(
        ax,
        2.5,
        8.9,
        3.4,
        1.35,
        "Early stop real MDM\n6 predicciones\n2 consecutivas\numbral 0.55",
        COLORS["control"],
        8,
    )
    endpoint = box(ax, 4.8, 8.45, 3.2, 0.85, "Fallback MDM\nendpoint −0.50 s", COLORS["control"])
    band = diamond(ax, 9.8, 8.88, 3.2, 1.3, "MDM endpoint\n≥0.60 MI\n≤0.40 REST\nintermedio AMBIGUOUS", COLORS["control"], 7.8)
    validate = diamond(ax, 14.0, 8.88, 3.2, 1.3, "¿LDA o LR\ncoincide con MDM?", COLORS["control"])
    original = box(ax, 4.1, 6.75, 4.1, 0.85, "mdm_operational_decision_original\n→ adaptive recentering", COLORS["recenter"], 8.5)
    final = box(ax, 11.6, 6.75, 4.0, 0.85, "final_validated_decision\n→ output operacional", COLORS["decision"], 8.7)

    arrow(ax, point(mdm, "bottom"), diamond_point(early, "top"))
    arrow(ax, diamond_point(early, "right"), point(endpoint, "left"), "No")
    arrow(ax, point(endpoint, "right"), diamond_point(band, "left"))
    arrow(ax, diamond_point(band, "right"), diamond_point(validate, "left"), "MDM MI/REST")
    arrow(ax, point(lda, "bottom"), diamond_point(validate, "top"), "clase ≥0.5", connectionstyle="arc3,rad=-0.28")
    arrow(ax, point(lr, "bottom"), diamond_point(validate, "top"), "clase ≥0.5", connectionstyle="arc3,rad=0.22")
    arrow(ax, diamond_point(early, "bottom"), point(original, "top"), "Sí")
    arrow(ax, diamond_point(band, "bottom"), point(original, "top"), "MDM original")
    arrow(ax, point(original, "right"), point(final, "left"), "early stop: sin validar")
    arrow(ax, diamond_point(validate, "bottom"), point(final, "top"), "acuerdo: MDM\ndesacuerdo: ambiguo")

    # Shadow branch
    records = box(ax, 19.4, 10.25, 3.8, 0.85, "Registro exacto por paso\n5 modelos", COLORS["diag"])
    shadow = box(ax, 18.0, 8.55, 2.7, 0.9, "SHADOW_\nEARLYSTOP", COLORS["diag"])
    stable = box(ax, 21.0, 8.55, 2.7, 0.9, "SHADOW_\nSTABILITY", COLORS["diag"])
    fastest = box(ax, 19.35, 6.9, 3.9, 0.8, "SHADOW_FASTEST_MODEL", COLORS["diag"])
    for model in models:
        arrow(
            ax,
            point(model, "right"),
            point(records, "left"),
            color="#7651A8",
            style="--",
            connectionstyle="arc3,rad=0.10",
        )
    arrow(ax, point(records, "bottom"), point(shadow, "top"), color="#7651A8", style="--")
    arrow(ax, point(records, "bottom"), point(stable, "top"), color="#7651A8", style="--")
    arrow(ax, point(shadow, "bottom"), point(fastest, "top"), color="#7651A8", style="--")
    arrow(ax, point(stable, "bottom"), point(fastest, "top"), color="#7651A8", style="--")

    # Recentring and operational output
    recenter = box(ax, 1.0, 4.9, 5.0, 0.95, "Adaptive recentering\nno ambigua · correcta · confianza ≥0.62\nsin BAD_EEG · actualización geodésica", COLORS["recenter"], 8.5)
    go = box(ax, 8.1, 4.95, 3.0, 0.85, "MI_BEGIN 200\no REST_BEGIN 100", COLORS["output"])
    result = diamond(ax, 13.2, 5.35, 3.0, 1.25, "¿Decisión final\ncorrecta?", COLORS["output"])
    reward = box(ax, 15.6, 5.0, 3.2, 0.9, "MI correcto\nfeedback + guante\nFES_MOTOR_GO", COLORS["output"], 8.5)
    rest = box(ax, 15.6, 3.65, 3.2, 0.75, "REST correcto\nmano estacionaria", COLORS["output"], 8.5)
    no_reward = box(ax, 11.4, 3.65, 3.2, 0.75, "Incorrecto/ambiguo\nsin recompensa", COLORS["neutral"], 8.5)
    robot = box(ax, 20.0, 5.0, 3.3, 0.9, "ROBOT_BEGIN 300\ntrayectoria + GO", COLORS["output"], 8.5)
    logs = box(ax, 4.9, 2.2, 6.0, 0.95, "Logs y CSV\nDECISION_LAYERS · probabilidades · calidad EEG\nresúmenes shadow y estabilidad", COLORS["diag"], 8.8)

    arrow(ax, point(original, "bottom"), point(recenter, "top"))
    arrow(ax, point(final, "bottom"), point(go, "top"))
    arrow(ax, point(go, "right"), diamond_point(result, "left"))
    arrow(ax, diamond_point(result, "right"), point(reward, "left"), "MI correcto")
    arrow(ax, diamond_point(result, "bottom"), point(rest, "top"), "REST correcto", connectionstyle="arc3,rad=-0.25")
    arrow(ax, diamond_point(result, "bottom"), point(no_reward, "top"), "No", connectionstyle="arc3,rad=0.25")
    arrow(ax, point(reward, "right"), point(robot, "left"))
    arrow(ax, point(original, "bottom"), point(logs, "top"), color="#7651A8", style="--", connectionstyle="arc3,rad=0.15")
    arrow(ax, point(final, "bottom"), point(logs, "top"), color="#7651A8", style="--", connectionstyle="arc3,rad=-0.10")
    arrow(ax, point(fastest, "bottom"), point(logs, "right"), color="#7651A8", style="--", connectionstyle="arc3,rad=0.20")

    # Preparation FES annotation
    prep_fes = box(
        ax,
        0.7,
        2.15,
        3.5,
        1.05,
        "Trial MI durante preparación\nMI_PREPARE 210\nFES_SENS_GO → FES_STOP",
        COLORS["output"],
        8.5,
    )
    arrow(ax, point(lsl, "bottom"), point(prep_fes, "top"), color="#B07A00", connectionstyle="arc3,rad=0.45")

    ax.plot([1.0, 2.0], [0.85, 0.85], color="#444444", linewidth=1.5)
    ax.text(2.15, 0.85, "flujo operacional", va="center", fontsize=9)
    ax.plot([5.0, 6.0], [0.85, 0.85], color="#7651A8", linewidth=1.5, linestyle="--")
    ax.text(6.15, 0.85, "flujo diagnóstico (no controla)", va="center", fontsize=9)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_DIR / "online_decoder_architecture.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        OUTPUT_DIR / "online_decoder_architecture.svg",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
