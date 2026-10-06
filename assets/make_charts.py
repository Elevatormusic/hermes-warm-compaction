"""Make the benchmark charts of the README from evidence/lcm-bench.json.

Run from the repository root: python assets/make_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
ARMS = (("plugin", "warm_compaction", "url(#warm)"), ("lcm", "hermes-lcm", "#8EA4C8"), ("stock", "built-in", "#5E6878"))
BAR_X, BAR_W, BAR_H, STEP = 260, 640, 26, 34


def card(width, height, title, subtitle, body):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{title}" font-family="{FONT}">
  <defs>
    <linearGradient id="warm" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#FFB547"/><stop offset="0.6" stop-color="#FF6B3D"/><stop offset="1" stop-color="#E03A5A"/>
    </linearGradient>
  </defs>
  <rect width="{width}" height="{height}" rx="24" fill="#101218"/>
  <rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="23.5" fill="none" stroke="#262A33"/>
  <text x="40" y="54" font-size="26" font-weight="700" fill="#F7F4EF">{title}</text>
  <text x="40" y="82" font-size="15" fill="#A9A39A">{subtitle}</text>
{body}</svg>
"""


def group(y, label, values, scale, fmt, note=""):
    """One labelled group of three bars. values: arm -> number."""
    out = [f'  <text x="40" y="{y}" font-size="17" font-weight="700" fill="#EDE8E0">{label}</text>']
    if note:
        out.append(f'  <text x="1160" y="{y}" font-size="14" text-anchor="end" fill="#FF9A5C">{note}</text>')
    for i, (arm, name, fill) in enumerate(ARMS):
        top = y + 14 + i * STEP
        width = max(values[arm] * scale, 0)
        out.append(f'  <text x="{BAR_X - 16}" y="{top + 18}" font-size="15" text-anchor="end" fill="#CFC8BD">{name}</text>')
        out.append(f'  <rect x="{BAR_X}" y="{top}" width="{BAR_W}" height="{BAR_H}" rx="6" fill="#181B22"/>')
        if width > 0:
            out.append(f'  <rect x="{BAR_X}" y="{top}" width="{width:.1f}" height="{BAR_H}" rx="6" fill="{fill}"/>')
        weight = "700" if arm == "plugin" else "400"
        out.append(f'  <text x="{BAR_X + max(width, 0) + 10:.1f}" y="{top + 18}" font-size="15" font-weight="{weight}" '
                   f'fill="#F7F4EF">{fmt(arm)}</text>')
    return "\n".join(out) + "\n"


def main():
    summary = json.loads((ROOT / "evidence" / "lcm-bench.json").read_text(encoding="utf-8"))["summary"]
    compress = {arm: summary[arm]["median_seconds"]["compress"] for arm, _, _ in ARMS}
    workflow = {arm: summary[arm]["median_workflow_seconds"] for arm, _, _ in ARMS}

    def ratio(values):
        return (f"median {values['lcm'] / values['plugin']:.1f}× lower than hermes-lcm, "
                f"{values['stock'] / values['plugin']:.1f}× lower than built-in")

    scale = BAR_W / 100
    body = group(124, "Compaction", compress, scale, lambda a: f"{compress[a]:.1f} s", ratio(compress))
    body += group(268, "Compaction + next reply", workflow, scale, lambda a: f"{workflow[a]:.1f} s", ratio(workflow))
    speed = card(1200, 400, "Compaction speed",
                 "DGX, automatic compaction, 10 synthetic sessions of about 105K tokens. Median seconds, lower is better.",
                 body)

    def counts(get, total):
        return {arm: get(summary[arm]) for arm, _, _ in ARMS}, total

    groups = (("Compacted facts kept (6 per session)", *counts(lambda s: s["compacted_facts_correct"], 60)),
              ("Fact in the middle of a long message", *counts(lambda s: s["facts_correct"]["read_replica"], 10)),
              ("Standing rule kept in the next reply", *counts(lambda s: s["owner_rule"]["continuation"], 10)))
    body = ""
    for index, (label, values, total) in enumerate(groups):
        body += group(124 + index * 138, label, values, BAR_W / total, lambda a, v=values, t=total: f"{v[a]}/{t}")
    quality = card(1200, 530, "What the agent still knows after compaction",
                   "Same 10 sessions. One probe after one automatic compaction. Higher is better.", body)

    (ROOT / "assets" / "bench-speed.svg").write_text(speed, encoding="utf-8", newline="\n")
    (ROOT / "assets" / "bench-quality.svg").write_text(quality, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
