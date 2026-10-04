"""Create a self-contained interactive viewer and a GIF from real PPO episodes."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/number-grid-5m"
DATA = json.loads((OUTPUT / "replays.json").read_text())
SUMMARY = json.loads((OUTPUT / "summary.json").read_text())
font_root = Path(matplotlib.get_data_path()) / "fonts/ttf"


def font(size, bold=False):
    return ImageFont.truetype(str(font_root / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def render(frame, total, history):
    image = Image.new("RGB", (608, 742), "#0b1220")
    draw = ImageDraw.Draw(image)
    draw.text((26, 18), "Number Grid · PPO", fill="#e5edf8", font=font(25, True))
    draw.text((26, 54), f"Шаг {frame['step']}   Число {frame['strength']}   Награда {total:+g}", fill="#99abc5", font=font(16))
    size, cell, x0, y0 = DATA["size"], 34, 32, 98
    contact = [[0] * size for _ in range(size)]
    for index, (row, column) in enumerate(DATA["initial"]["opponent_positions"]):
        if not frame["alive"][index]:
            continue
        dangerous = frame["strength"] <= DATA["initial"]["opponent_strengths"][index]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                r, c = row + dr, column + dc
                if 0 <= r < size and 0 <= c < size:
                    contact[r][c] = max(contact[r][c], 2 if dangerous else 1)
    for row in range(size):
        for column in range(size):
            wall = DATA["initial"]["walls"][row][column]
            fill = "#35445c" if wall else "#3c2337" if contact[row][column] == 2 else "#153731" if contact[row][column] == 1 else "#17273d"
            x, y = x0 + column * cell, y0 + row * cell
            draw.rounded_rectangle((x + 1, y + 1, x + cell - 1, y + cell - 1), radius=3, fill=fill)
    points = [(x0 + (f["position"][1] + .5) * cell, y0 + (f["position"][0] + .5) * cell) for f in history]
    if len(points) > 1:
        draw.line(points, fill="#4178a7", width=2)

    def token(position, number, color, agent=False):
        row, column = position
        x, y = x0 + column * cell, y0 + row * cell
        draw.rounded_rectangle((x + 4, y + 4, x + cell - 4, y + cell - 4), radius=7,
                               fill=color, outline="#d8ecff" if agent else None, width=2)
        draw.text((x + cell / 2, y + cell / 2), str(number), anchor="mm", fill="#0b1220", font=font(20, True))
    for i, position in enumerate(DATA["initial"]["opponent_positions"]):
        if frame["alive"][i]:
            number = DATA["initial"]["opponent_strengths"][i]
            token(position, number, "#34d399" if frame["strength"] > number else "#fb7185")
    token(frame["position"], frame["strength"], "#60a5fa", True)
    labels = ["Игра идёт", "ПОБЕДА · Все оппоненты побеждены", "ПОРАЖЕНИЕ", "ТАЙМАУТ"]
    colors = ["#99abc5", "#34d399", "#fb7185", "#fbbf24"]
    draw.text((32, 660), labels[frame["outcome"]], fill=colors[frame["outcome"]], font=font(17, True))
    draw.text((32, 691), "Голубой — агент · Зелёный — слабее · Красный — опасен", fill="#99abc5", font=font(13))
    draw.text((32, 715), "Реальная запись обученного агента. Соседство: 8 клеток.", fill="#99abc5", font=font(13))
    return image


template = (ROOT / "number_grid_viewer.template.html").read_text()
viewer = template.replace("__DATA__", json.dumps(DATA, ensure_ascii=False)).replace(
    "__SUMMARY__", json.dumps(SUMMARY, ensure_ascii=False)).replace(
    "__SCRIPT__", (ROOT / "number_grid_viewer.js").read_text())
(OUTPUT / "viewer.html").write_text(viewer)
initial = DATA["initial"]
render(initial, 0, [initial]).save(OUTPUT / "preview.png")
episode = next((ep for ep in DATA["episodes"] if ep["outcome"] == 1), DATA["episodes"][0])
timeline = [initial] + episode["frames"]
images, durations, total = [], [], 0
for index, frame in enumerate(timeline):
    total += frame["reward"]
    images.append(render(frame, total, timeline[:index + 1]))
    durations.append(800 if index == 0 else 160)
durations[-1] = 1400
images[0].save(OUTPUT / "trained_agent.gif", save_all=True, append_images=images[1:], duration=durations, loop=0)
images[-1].save(OUTPUT / "victory.png")

rows = list(csv.DictReader((OUTPUT / "metrics.csv").open()))
steps = [int(row["timesteps"]) / 1e6 for row in rows]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
axes[0].plot(steps, [float(row["training_mean_return"]) for row in rows], label="Training", marker="o")
axes[0].plot(steps, [float(row["evaluation_mean_return"]) for row in rows], label="Evaluation", marker=".")
axes[0].axhline(SUMMARY["final_evaluations"]["random"]["mean_return"], color="gray", linestyle="--", label="Random policy")
axes[0].set(xlabel="Training transitions (million)", ylabel="Episode return", title="NumberGrid: episode return")
axes[0].legend()
axes[1].plot(steps, [float(row["evaluation_mean_length"]) for row in rows], marker="o")
axes[1].axhline(17, color="gray", linestyle="--", label="Shortest path: 17")
axes[1].set(xlabel="Training transitions (million)", ylabel="Steps per episode", title="NumberGrid: evaluation length")
axes[1].legend()
for axis in axes:
    axis.grid(alpha=.2)
fig.savefig(OUTPUT / "training_curve.png", dpi=160)
plt.close(fig)
print(json.dumps({"viewer": str(OUTPUT / "viewer.html"), "gif": str(OUTPUT / "trained_agent.gif"),
                  "replay_length": episode["length"], "replay_return": episode["return"]}, indent=2))
