"""Create dependency-free SVG figures for the D4 manuscript."""

from csv import DictReader
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"


def text(x, y, value, size=12, anchor="start", fill="#222", weight="400", rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return f'<text x="{x}" y="{y}" font-size="{size}px" text-anchor="{anchor}" fill="{fill}" font-family="Arial, sans-serif" font-weight="{weight}"{transform}>{escape(str(value))}</text>'


def line(x1, y1, x2, y2, stroke="#b8b8b8", width=1, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'


def svg_start(width, height, title, desc):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(desc)}</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def write_svg(path, parts):
    parts.append("</svg>\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def load_csv(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(DictReader(handle))


def figure_d4a():
    rows = load_csv("d4a_repair_alignment.csv")
    width, height = 1100, 520
    left, top, plot_w, plot_h = 85, 65, 970, 350
    y_max = 15.0
    parts = svg_start(width, height, "D4a phase alignment", "Formal D4a q90 threshold error across eight cells.")
    parts.append(text(width / 2, 28, "D4a: utility alignment differs by phase marker", 18, "middle", weight="500"))
    for tick in (0, 5, 10, 15):
        y = top + plot_h - (tick / y_max) * plot_h
        parts.append(line(left, y, left + plot_w, y, "#dddddd", 1, "3 3"))
        parts.append(text(left - 10, y + 4, tick, 11, "end", "#444"))
    parts.append(line(left, top, left, top + plot_h, "#555", 1.2))
    parts.append(line(left, top + plot_h, left + plot_w, top + plot_h, "#555", 1.2))
    parts.append(text(20, top + plot_h / 2, "Mean q90 threshold error", 12, "middle", "#222", rotate=-90))
    colors = ["#2a6fbb", "#8c8c8c", "#d9772b"]
    keys = ["dyn_to_util_q90_err_mean", "dyn_null_to_util_q90_err_mean", "rank_to_util_q90_err_mean"]
    labels = ["C*_dyn vs utility", "null dynamics vs utility", "C*_rank vs utility"]
    group_w = plot_w / len(rows)
    bar_w = group_w * 0.22
    for i, row in enumerate(rows):
        cx = left + group_w * (i + 0.5)
        for j, (key, color) in enumerate(zip(keys, colors)):
            value = float(row[key])
            x = cx + (j - 1) * bar_w * 1.12 - bar_w / 2
            y = top + plot_h - value / y_max * plot_h
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{top + plot_h - y:.2f}" fill="{color}"/>')
        label = f'{row["split"]} / {row["variant"]} / K={row["K"]}'
        parts.append(text(cx, top + plot_h + 18, label, 9, "end", "#333", rotate=-35))
    legend_x = 165
    for color, label in zip(colors, labels):
        parts.append(f'<rect x="{legend_x}" y="455" width="14" height="14" fill="{color}"/>')
        parts.append(text(legend_x + 20, 467, label, 11))
        legend_x += 235
    parts.append(text(left, 505, "Formal D4a alignment table; each cell aggregates 20 seeds.", 10, "start", "#666"))
    write_svg(FIGURES / "d4a_phase_alignment.svg", parts)


def figure_d4b():
    rows = load_csv("d4b_formal_gpu/d4b_strength_summary.csv")
    width, height = 1120, 470
    panel_w, panel_gap = 330, 30
    top, plot_h = 75, 280
    lefts = [65 + i * (panel_w + panel_gap) for i in range(3)]
    splits = ["id", "ood_random", "ood_inverted"]
    parts = svg_start(width, height, "D4b residual desynchronization", "Formal D4b grouped summary recomputed from 3000 raw configurations.")
    parts.append(text(width / 2, 30, "D4b: residual bypass is associated with phase-marker desynchronization", 17, "middle", weight="500"))
    colors = {"sync_gap": "#2a6fbb", "utility_sensitivity": "#d9772b", "negative_delta_rate": "#2e8b57"}
    for panel, split in enumerate(splits):
        sub = sorted((r for r in rows if r["split"] == split), key=lambda r: float(r["residual_strength"]))
        left = lefts[panel]
        right = left + panel_w
        parts.append(text(left + panel_w / 2, 58, split.replace("ood_", "OOD ").replace("_", " ").title(), 13, "middle", weight="500"))
        for tick in (0, 4, 8, 12, 16):
            y = top + plot_h - tick / 16 * plot_h
            parts.append(line(left, y, right, y, "#dddddd", 1, "3 3"))
            if panel == 0:
                parts.append(text(left - 8, y + 4, tick, 10, "end", "#444"))
        parts.append(line(left, top, left, top + plot_h, "#555", 1.2))
        parts.append(line(left, top + plot_h, right, top + plot_h, "#555", 1.2))
        if panel == 0:
            parts.append(text(16, top + plot_h / 2, "Sync gap", 11, "middle", "#222", rotate=-90))
            parts.append(text(1100, top + plot_h / 2, "Rate / sensitivity", 11, "middle", "#222", rotate=90))
        xs = []
        for i, row in enumerate(sub):
            x = left + 35 + i * (panel_w - 55) / (len(sub) - 1)
            xs.append(x)
            parts.append(text(x, top + plot_h + 18, row["residual_strength"], 10, "middle"))
        points = {}
        for key, color, scale in (("sync_gap_mean", colors["sync_gap"], 16), ("utility_sensitivity_mean", colors["utility_sensitivity"], 1), ("negative_delta_rate_mean", colors["negative_delta_rate"], 1)):
            coords = []
            for x, row in zip(xs, sub):
                value = float(row[key])
                y = top + plot_h - value / scale * plot_h
                coords.append((x, y))
            points[key] = coords
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in coords)}"/>')
            marker = {"sync_gap_mean": "circle", "utility_sensitivity_mean": "square", "negative_delta_rate_mean": "triangle"}[key]
            for x, y in coords:
                if marker == "circle":
                    parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
                elif marker == "square":
                    parts.append(f'<rect x="{x - 3.5:.1f}" y="{y - 3.5:.1f}" width="7" height="7" fill="{color}"/>')
                else:
                    parts.append(f'<path d="M {x:.1f} {y - 4:.1f} L {x - 4:.1f} {y + 3:.1f} L {x + 4:.1f} {y + 3:.1f} Z" fill="{color}"/>')
    legend = [("#2a6fbb", "sync gap"), ("#d9772b", "utility sensitivity"), ("#2e8b57", "dynamics-first rate")]
    x = 245
    for color, label in legend:
        parts.append(f'<rect x="{x}" y="395" width="13" height="13" fill="{color}"/>')
        parts.append(text(x + 19, 406, label, 11))
        x += 205
    parts.append(text(65, 445, "Formal grouped summary from 3000 raw rows; 10 seeds, 5 strengths, 3 splits. Thresholds remain operational and per-C retrained.", 10, "start", "#666"))
    write_svg(FIGURES / "d4b_residual_desynchronization.svg", parts)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    figure_d4a()
    figure_d4b()
    print("Wrote SVG figures to", FIGURES)
