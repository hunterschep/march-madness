from __future__ import annotations

"""Build a small self-contained HTML tool for querying matchup probabilities by team name."""

import json
from datetime import datetime, timezone
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmmania.matchup_lookup import build_matchup_lookup_bundle


SEASON = 2026
OUTPUT_DIR = ROOT / "outputs" / "tools"
OUTPUT_PATH = OUTPUT_DIR / "matchup_lookup_2026.html"


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>March Madness Matchup Lookup</title>
    <style>
      :root {
        --bg: #f5efe2;
        --panel: #fffdf7;
        --ink: #1c2822;
        --muted: #607066;
        --line: rgba(28, 40, 34, 0.15);
        --accent: #1f6b52;
        --accent-soft: rgba(31, 107, 82, 0.12);
        --shadow: 0 18px 42px rgba(61, 53, 36, 0.12);
        --font-body: "Avenir Next", "Trebuchet MS", "Segoe UI", sans-serif;
        --font-display: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      }

      * { box-sizing: border-box; }
      body {
        margin: 0;
        min-height: 100vh;
        font-family: var(--font-body);
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(31, 107, 82, 0.10), transparent 35%),
          linear-gradient(180deg, #faf5ec 0%, var(--bg) 100%);
      }
      .shell {
        width: min(860px, calc(100vw - 24px));
        margin: 0 auto;
        padding: 28px 0 40px;
      }
      .card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 24px;
        box-shadow: var(--shadow);
      }
      .hero {
        padding: 28px;
      }
      .eyebrow {
        margin: 0 0 8px;
        color: var(--accent);
        font-size: 0.8rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
      }
      h1 {
        margin: 0;
        font-family: var(--font-display);
        font-size: clamp(2rem, 5vw, 3.6rem);
        line-height: 0.96;
      }
      .lede {
        margin: 14px 0 0;
        max-width: 60ch;
        color: var(--muted);
        line-height: 1.5;
      }
      .controls {
        display: grid;
        gap: 18px;
        margin-top: 18px;
        padding: 22px;
      }
      .toggle-row, .button-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
      }
      .toggle, .button {
        border: 1px solid var(--line);
        background: #fff;
        color: var(--ink);
        padding: 11px 15px;
        border-radius: 999px;
        font: inherit;
        cursor: pointer;
      }
      .toggle.active, .button.primary {
        background: var(--accent);
        color: #fffdf7;
        border-color: transparent;
      }
      .grid {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        gap: 12px;
        align-items: end;
      }
      .field {
        display: grid;
        gap: 8px;
      }
      .label {
        font-size: 0.8rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
      }
      input {
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 13px 14px;
        font: inherit;
        background: #fff;
      }
      .swap {
        margin-bottom: 1px;
      }
      .result {
        margin-top: 18px;
        padding: 22px;
      }
      .result h2 {
        margin: 0 0 10px;
        font-family: var(--font-display);
        font-size: 1.9rem;
      }
      .prob-row {
        display: grid;
        gap: 12px;
        margin-top: 14px;
      }
      .prob-card {
        border: 1px solid var(--line);
        border-radius: 18px;
        background: linear-gradient(180deg, #fff, #f8f3e8);
        padding: 16px;
      }
      .prob-team {
        font-weight: 700;
        font-size: 1rem;
      }
      .prob-value {
        margin-top: 6px;
        font-family: var(--font-display);
        font-size: 2rem;
      }
      .hint, .status {
        color: var(--muted);
        line-height: 1.5;
      }
      .status.error {
        color: #8e3614;
      }
      @media (max-width: 720px) {
        .grid {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <section class="card hero">
        <p class="eyebrow">live submission lookup</p>
        <h1>Ask the model about any matchup by team name.</h1>
        <p class="lede">
          Type two team names, pick men or women, and this tool will show the exact probability from the current
          live submission file. Aliases from Kaggle team spellings are supported, so inputs like Utah State will resolve.
        </p>
      </section>

      <section class="card controls">
        <div>
          <div class="label">Tournament</div>
          <div class="toggle-row" id="side-toggles"></div>
        </div>

        <div class="grid">
          <label class="field">
            <span class="label">Team A</span>
            <input id="team-a-input" list="team-options" placeholder="Arkansas">
          </label>
          <button class="button swap" id="swap-button" type="button">Swap</button>
          <label class="field">
            <span class="label">Team B</span>
            <input id="team-b-input" list="team-options" placeholder="Utah State">
          </label>
        </div>

        <div class="button-row">
          <button class="button primary" id="lookup-button" type="button">Lookup matchup</button>
          <button class="button" id="clear-button" type="button">Clear</button>
        </div>

        <datalist id="team-options"></datalist>
      </section>

      <section class="card result">
        <h2>Matchup probability</h2>
        <p class="status" id="status">
          Built __BUILD_TIMESTAMP__. Enter two teams from the same side to query the live submission.
        </p>
        <div class="prob-row" id="prob-row"></div>
      </section>
    </div>

    <script id="matchup-data" type="application/json">__MATCHUP_DATA__</script>
    <script>
      const bundle = JSON.parse(document.getElementById("matchup-data").textContent);
      const state = { sideCode: "M" };

      const sideToggles = document.getElementById("side-toggles");
      const teamOptions = document.getElementById("team-options");
      const teamAInput = document.getElementById("team-a-input");
      const teamBInput = document.getElementById("team-b-input");
      const status = document.getElementById("status");
      const probRow = document.getElementById("prob-row");

      renderSideToggles();
      refreshTeamList();

      document.getElementById("lookup-button").addEventListener("click", runLookup);
      document.getElementById("clear-button").addEventListener("click", () => {
        teamAInput.value = "";
        teamBInput.value = "";
        probRow.innerHTML = "";
        status.className = "status";
        status.textContent = "Cleared. Enter two teams to query the live submission.";
      });
      document.getElementById("swap-button").addEventListener("click", () => {
        const a = teamAInput.value;
        teamAInput.value = teamBInput.value;
        teamBInput.value = a;
      });
      teamAInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") runLookup();
      });
      teamBInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") runLookup();
      });

      function activeSide() {
        return bundle.sides[state.sideCode];
      }

      function renderSideToggles() {
        sideToggles.innerHTML = "";
        for (const [code, side] of Object.entries(bundle.sides)) {
          const button = document.createElement("button");
          button.type = "button";
          button.className = `toggle${code === state.sideCode ? " active" : ""}`;
          button.textContent = side.label;
          button.addEventListener("click", () => {
            state.sideCode = code;
            renderSideToggles();
            refreshTeamList();
            probRow.innerHTML = "";
            status.className = "status";
            status.textContent = `Switched to ${side.label}. Enter two teams to query the live submission.`;
          });
          sideToggles.appendChild(button);
        }
      }

      function refreshTeamList() {
        teamOptions.innerHTML = "";
        for (const team of activeSide().teams) {
          const option = document.createElement("option");
          option.value = team.teamName;
          teamOptions.appendChild(option);
        }
      }

      function normalize(value) {
        return value.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim().replace(/\\s+/g, " ");
      }

      function resolveTeamId(value) {
        const key = normalize(value);
        return activeSide().aliases[key] ?? null;
      }

      function resolveCanonicalName(teamId) {
        const team = activeSide().teams.find((row) => row.teamId === teamId);
        return team ? team.teamName : null;
      }

      function runLookup() {
        const teamAId = resolveTeamId(teamAInput.value);
        const teamBId = resolveTeamId(teamBInput.value);

        if (!teamAId || !teamBId) {
          probRow.innerHTML = "";
          status.className = "status error";
          status.textContent = "Could not resolve one or both team names on the selected side.";
          return;
        }

        if (teamAId === teamBId) {
          probRow.innerHTML = "";
          status.className = "status error";
          status.textContent = "Choose two different teams.";
          return;
        }

        const lowId = Math.min(teamAId, teamBId);
        const highId = Math.max(teamAId, teamBId);
        const key = `${lowId}_${highId}`;
        const pLow = activeSide().probabilities[key];
        if (pLow == null) {
          probRow.innerHTML = "";
          status.className = "status error";
          status.textContent = "No probability found for that matchup in the live submission.";
          return;
        }

        const teamAName = resolveCanonicalName(teamAId);
        const teamBName = resolveCanonicalName(teamBId);
        const teamAProb = teamAId === lowId ? pLow : 1 - pLow;
        const teamBProb = 1 - teamAProb;

        status.className = "status";
        status.textContent = `${teamAName} vs ${teamBName} from live_submission_2026.csv`;

        probRow.innerHTML = `
          <article class="prob-card">
            <div class="prob-team">${teamAName}</div>
            <div class="prob-value">${(teamAProb * 100).toFixed(1)}%</div>
            <div class="hint">Probability ${teamAName} wins this matchup.</div>
          </article>
          <article class="prob-card">
            <div class="prob-team">${teamBName}</div>
            <div class="prob-value">${(teamBProb * 100).toFixed(1)}%</div>
            <div class="hint">Probability ${teamBName} wins this matchup.</div>
          </article>
        `;
      }
    </script>
  </body>
</html>
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = build_matchup_lookup_bundle(SEASON)
    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = (
        HTML_TEMPLATE.replace("__MATCHUP_DATA__", json.dumps(bundle))
        .replace("__BUILD_TIMESTAMP__", built_at)
    )
    OUTPUT_PATH.write_text(html)


if __name__ == "__main__":
    main()
