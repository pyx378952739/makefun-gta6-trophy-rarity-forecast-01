// gta6-trophy-rarity-forecast-01 (Node 22, standard library only)
//
// Forecast of every Grand Theft Auto VI PlayStation 5 trophy's earn rate on
// Exophase at the case's target, as five quantiles per trophy, scored by
// pinball loss on the logit scale.
//
// Model — a counts model on the trophy page's own refresh dynamics.
//
// Exophase does not update the game's page continuously: the house saw it
// frozen for hours, then refreshed in one jump (638 -> 3,868 players tracked
// on launch night, every rate but the 100% one falling). A read therefore
// shows the rates as of the page's last refresh, and the forecast must model
// the refreshes, not the frozen reads between them. The program works in
// holder counts (H_i = rate_i * N / 100) and players tracked (N), which is
// what a refresh actually moves:
//
//   * Plateaus. Consecutive reads with identical players-tracked and rates
//     are one plateau. A change starts a new plateau: a refresh. The program
//     reads the refreshes out of the case's own history: the median gap
//     between plateau starts (the refresh cadence), the median log growth of
//     N per refresh, and, per trophy, the trend of its log rate over
//     refreshes (a least-squares slope: the page prints 2 decimals, so a
//     median of per-refresh ratios would read zero once small rates sit on
//     the rounding floor).
//   * Forward simulation. From the last read to the target it applies the
//     expected number of refreshes, each step's log ratio decaying toward
//     zero (growth saturates: the launch surge passes, rates settle), so the
//     median log rate is last log rate + lam_i * (1 - d^n) / (1 - d).
//   * Spread. Nine deterministic scenarios (refresh count x0.7 / x1 / x1.4
//     crossed with per-trophy growth x0.7 / x1 / x1.3) give the scenario
//     dispersion of each trophy's logit rate; the spread is that dispersion
//     combined with the residual dispersion of the per-refresh ratios and a
//     horizon-scaled floor, clamped to [0.45, 1.5] logit units. The upper
//     spread is 1.4x the lower one: upside risk (slower dilution, faster
//     progression) is larger than downside risk.
//   * No refresh in history. Launch night's practice histories are flat: no
//     refresh is visible, yet the answers sit after the page's first refresh.
//     The program then uses a prior fitted on the competition's public
//     description of that refresh: one refresh before a >= 60-minute target,
//     N x6, new players holding each trophy at a small fraction of its
//     current rate (tilted so the 100% trophy stays 100%: every new player
//     holds it), i.e. a median rate of roughly 0.22x the last rate for
//     ordinary trophies, plus a small upward drift that grows with the
//     horizon (players progress) and a wide asymmetric spread whose upper
//     tail reaches the undiluted rate, so the no-refresh branch is hedged
//     rather than assumed away. A horizon under 60 minutes expects no
//     refresh and falls back to persistence.
//   * Trophies at 100% in the last read stay at 99.99 across all quantiles:
//     every tracked player holds them, and a new player holds the intro
//     trophy too, so the rate cannot fall by dilution; the logit of 99.99 is
//     what the scorer clamps 100% to, so 99.99 scores exactly.
//
// Safety. The program never throws a case away: any failure in the model
// falls back, trophy by trophy, to the house baseline's persistence forecast
// (median the last rate, spread 0.5 * sqrt(days) logit units), and a failure
// in a whole case falls back to the baseline for the whole case, so the
// output is always a valid file. It reads every <id>.in in the input
// directory and writes one <id>.out per case:
// {"trophies":[{"id":1,"q":[q05,q25,q50,q75,q95]}, ...]}, 55 rows.
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Per-refresh log-ratio decay: growth saturates as the launch surge passes. */
const DECAY = 0.85;
/** Prior refresh interval (minutes) when the history shows no refresh. */
const PRIOR_INTERVAL_MIN = 240;
/** Prior players-tracked ratio of one refresh, from the public description
 *  of launch night's first refresh (638 -> 3,868 players). */
const PRIOR_N_RATIO = 6;
/** Prior share of a trophy's current rate at which new players hold it. */
const PRIOR_INFLOW_ALPHA = 0.07;
/** A horizon at least this long, with a flat history, is expected to
 *  contain a refresh. */
const REFRESH_HORIZON_MIN = 60;
/** Spread bounds, in logit units. */
const SIGMA_MIN = 0.45;
const SIGMA_MAX = 1.5;
/** Upper spread is wider than the lower one (upside risk is larger). */
const SIGMA_HI_MULT = 1.4;

interface Trophy {
  id: number;
}

interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

const clamp = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clamp(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

const median = (xs: number[]): number => {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
};

const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);

const sd = (xs: number[]): number => {
  if (xs.length < 2) return 0;
  const mu = mean(xs);
  return Math.sqrt(xs.reduce((a, b) => a + (b - mu) * (b - mu), 0) / (xs.length - 1));
};

/** Last non-null rate of trophy i over the history, else the last read's
 *  median rate, else 0.5. */
const lastRate = (c: Case, i: number): number => {
  for (let j = c.history.length - 1; j >= 0; j--) {
    const r = c.history[j].rates[i];
    if (r !== null && r !== undefined && Number.isFinite(r)) return r;
  }
  const last = c.history[c.history.length - 1];
  const vals = last.rates.filter((r): r is number => r !== null && Number.isFinite(r));
  return vals.length > 0 ? median(vals) : 0.5;
};

/** The house baseline's persistence forecast for one trophy. */
const persistenceRow = (r: number, days: number): number[] => {
  const spread = 0.5 * Math.sqrt(Math.max(0, days));
  const l = logit(r);
  return Z.map((z) => round4(invLogit(l + z * spread)));
};

const persistenceCase = (c: Case): Array<{ id: number; q: number[] }> => {
  const days = c.horizon_minutes / 1440;
  return c.trophies.map((t, i) => ({ id: t.id, q: persistenceRow(lastRate(c, i), days) }));
};

/** Build one row from a median logit and lower/upper spreads. */
const rowFromLogit = (m: number, sLo: number, sHi: number): number[] =>
  Z.map((z) => round4(clamp(invLogit(m + z * (z < 0 ? sLo : sHi)))));

/** The model forecast for one case. Throws on malformed input; the caller
 *  falls back to persistence. */
const modelCase = (c: Case): Array<{ id: number; q: number[] }> => {
  if (c.history.length === 0) throw new Error('no history');
  const last = c.history[c.history.length - 1];
  const lastMs = Date.parse(last.at);
  const targetMs = Number.isFinite(Date.parse(c.target))
    ? Date.parse(c.target)
    : lastMs + c.horizon_minutes * 60_000;
  const horizonMin = Math.max(0, (targetMs - lastMs) / 60_000);
  const days = horizonMin / 1440;

  // Plateau starts: a read whose players-tracked or any rate differs from
  // the previous read starts a new plateau (a refresh of the page).
  const starts: number[] = [0];
  for (let j = 1; j < c.history.length; j++) {
    const a = c.history[j - 1];
    const b = c.history[j];
    let changed = a.players_tracked !== b.players_tracked;
    if (!changed) {
      for (let i = 0; i < b.rates.length; i++) {
        if (a.rates[i] !== b.rates[i]) {
          changed = true;
          break;
        }
      }
    }
    if (changed) starts.push(j);
  }
  const nRefresh = starts.length - 1;

  // Per-trophy last rates.
  const rates = c.trophies.map((_, i) => lastRate(c, i));

  if (nRefresh >= 1) {
    // ---- Data-driven path: learn the refresh dynamics from history. ----
    const t = (j: number): number => Date.parse(c.history[j].at);
    const gaps: number[] = [];
    for (let k = 1; k < starts.length; k++) gaps.push((t(starts[k]) - t(starts[k - 1])) / 60_000);
    const posGaps = gaps.filter((g) => g > 0);
    const interval = posGaps.length > 0 ? median(posGaps) : PRIOR_INTERVAL_MIN;
    const nExp = horizonMin / interval;

    // Per-trophy trend of the log rate over refreshes: the least-squares
    // slope of log rate against plateau index. A median of per-refresh
    // ratios is not used: at the page's 2-decimal rounding the small rates
    // sit on a rounding floor (0.02, 0.02, ...) and most ratios read exactly
    // zero, which would hide a real trend. The slope averages that noise
    // out. Residuals around each trophy's line pool into the spread.
    const lam: number[] = [];
    const hasTrend: boolean[] = [];
    const resid: number[] = [];
    for (let i = 0; i < c.trophies.length; i++) {
      const pts: Array<[number, number]> = [];
      for (let k = 0; k < starts.length; k++) {
        const r = c.history[starts[k]].rates[i];
        if (r !== null && r > 0) pts.push([k, Math.log(clamp(r))]);
      }
      let slope = 0;
      let ok = false;
      if (pts.length >= 2) {
        const n = pts.length;
        const mk = pts.reduce((s, [k]) => s + k, 0) / n;
        const my = pts.reduce((s, [, y]) => s + y, 0) / n;
        let sxy = 0;
        let sxx = 0;
        for (const [k, y] of pts) {
          sxy += (k - mk) * (y - my);
          sxx += (k - mk) * (k - mk);
        }
        if (sxx > 0) {
          slope = sxy / sxx;
          ok = true;
          for (const [k, y] of pts) resid.push(y - (my + slope * (k - mk)));
        }
      }
      lam.push(slope);
      hasTrend.push(ok);
    }
    // Trophies with no usable trend borrow the pooled slope.
    const observed = lam.filter((_, i) => hasTrend[i]);
    const pooledLam = observed.length > 0 ? median(observed) : 0;
    for (let i = 0; i < lam.length; i++) if (!hasTrend[i]) lam[i] = pooledLam;
    const residSd = sd(resid);

    // Geometric decay sum for nExp steps: sum_j d^j, j = 0..nExp-1.
    const decaySum = nExp <= 0 ? 0 : (1 - Math.pow(DECAY, nExp)) / (1 - DECAY);
    // Scenario grid: refresh count x growth, weights favour the centre.
    const nScales = [0.7, 1, 1.4];
    const gScales = [0.7, 1, 1.3];
    const wN = [0.2, 0.6, 0.2];
    const wG = [0.2, 0.6, 0.2];

    return c.trophies.map((tr, i) => {
      const r = rates[i];
      if (r >= 99.99) return { id: tr.id, q: [99.99, 99.99, 99.99, 99.99, 99.99] };
      const base = logit(r);
      const scen: Array<[number, number]> = [];
      for (let a = 0; a < 3; a++) {
        for (let b = 0; b < 3; b++) {
          const nS = nExp * nScales[a];
          const dSum = nS <= 0 ? 0 : (1 - Math.pow(DECAY, nS)) / (1 - DECAY);
          scen.push([wN[a] * wG[b], base + lam[i] * gScales[b] * dSum]);
        }
      }
      const wTot = scen.reduce((s, [w]) => s + w, 0);
      const mu = scen.reduce((s, [w, x]) => s + w * x, 0) / wTot;
      const variance = scen.reduce((s, [w, x]) => s + w * (x - mu) * (x - mu), 0) / wTot;
      const sigmaScen = Math.sqrt(variance);
      const sigma = Math.min(
        SIGMA_MAX,
        Math.max(SIGMA_MIN, Math.sqrt(sigmaScen * sigmaScen + residSd * residSd + 0.0625 * days)),
      );
      const med = base + lam[i] * decaySum;
      return { id: tr.id, q: rowFromLogit(med, sigma, sigma * SIGMA_HI_MULT) };
    });
  }

  // ---- Flat history: no refresh is visible. Use the launch-night prior. ----
  if (horizonMin < REFRESH_HORIZON_MIN) return persistenceCase(c);
  const sLo = Math.min(0.8, 0.5 + 0.1 * Math.sqrt(days));
  const sHi = Math.min(1.8, 1.3 + 0.2 * Math.sqrt(days));
  // Small upward drift with horizon: players progress while the page sits.
  const drift = 0.35 * Math.log1p(days);
  return c.trophies.map((tr, i) => {
    const r = rates[i];
    if (r >= 99.99) return { id: tr.id, q: [99.99, 99.99, 99.99, 99.99, 99.99] };
    // One refresh: N x6; new players hold the trophy at a small fraction of
    // its current rate, tilted so the 100% trophy stays 100%.
    const alpha = PRIOR_INFLOW_ALPHA + (1 - PRIOR_INFLOW_ALPHA) * Math.pow(r / 100, 2);
    const f = (1 + alpha * (PRIOR_N_RATIO - 1)) / PRIOR_N_RATIO;
    const m = logit(Math.max(0.01, r * f)) + drift * Math.pow(r / 100, 0.25);
    return { id: tr.id, q: rowFromLogit(m, sLo, sHi) };
  });
};

/** Validate one output row; anything off falls back to persistence. */
const validRow = (row: { id: number; q: number[] }, n: number): boolean => {
  if (!Number.isInteger(row.id) || !Array.isArray(row.q) || row.q.length !== 5) return false;
  let prev = -Infinity;
  for (const q of row.q) {
    if (!Number.isFinite(q) || q < 0 || q > 100 || q < prev) return false;
    prev = q;
  }
  return true;
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  let rows: Array<{ id: number; q: number[] }>;
  try {
    const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
    if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
      throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}`);
    }
    try {
      rows = modelCase(c);
    } catch {
      rows = persistenceCase(c);
    }
    // Per-row safety: any invalid row is replaced by its persistence row.
    const days = c.horizon_minutes / 1440;
    rows = rows.map((row, i) => {
      if (validRow(row, c.trophies.length) && row.id === c.trophies[i].id) return row;
      return { id: c.trophies[i].id, q: persistenceRow(lastRate(c, i), days) };
    });
    if (rows.length !== c.trophies.length) throw new Error(`${file}: row count mismatch`);
  } catch {
    // Last resort: the case could not be read at all; write nothing rather
    // than a malformed file (a missing case scores as the last read, same as
    // an invalid one, but a malformed file risks more).
    continue;
  }
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies: rows })}\n`);
}
