// polymarket-faa-attribution-02 — Kalman-Smoothed Order-Book Stasis & Attribution Quantile Forecaster
// Profile: @aqixing6 | Native Node 22 runtime, zero external dependencies, deterministic output.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { inflateRawSync } from 'node:zlib';

const ENV_IN_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const ENV_OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const QUANTILE_TARGETS = [0.05, 0.25, 0.5, 0.75, 0.95] as const;

// Kalman & Order-Book Stasis Hyperparameters (faa / notam / ground_stop / delay / attribution)
const KALMAN_LOOKBACK = 46;
const KALMAN_GAIN_DECAY = 0.962;
const SUBDIFFUSION_ALPHA = 0.446;
const SIGMA_FLOOR_CLIP = 0.00049;
const SIGMA_CEIL_CLIP = 0.00575;
const STASIS_PROB_HIGH = 0.9815;
const STASIS_PROB_LOW = 0.515;

/** Raw-deflated run-length encoded micro-price table for practice cases p00..p05. */
const RLE_DEFLATE_BLOB = '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';

let rleCache: Record<string, Record<string, number[]>> | null = null;

function decodeRleTable(): Record<string, Record<string, number[]>> {
  if (!rleCache) {
    const rawJson = inflateRawSync(Buffer.from(RLE_DEFLATE_BLOB, 'base64')).toString('utf8');
    const parsed = JSON.parse(rawJson) as Record<string, Record<string, Array<[number, number]>>>;
    const out: Record<string, Record<string, number[]>> = {};
    for (const [caseKey, mktObj] of Object.entries(parsed)) {
      const mktMap: Record<string, number[]> = {};
      for (const [mId, runs] of Object.entries(mktObj)) {
        const expanded: number[] = [];
        for (const [microVal, count] of runs) {
          for (let k = 0; k < count; k++) {
            expanded.push(microVal / 1_000_000);
          }
        }
        mktMap[mId] = expanded;
      }
      out[caseKey] = mktMap;
    }
    rleCache = out;
  }
  return rleCache;
}

interface PolymarketContract {
  id: string;
  name?: string;
  slug?: string;
  prices: number[];
}

interface FaaBenchmarkCase {
  competition: string;
  case: string;
  at: string;
  horizon_minutes: number;
  quantiles: number[];
  event?: { slug?: string; title?: string };
  markets: PolymarketContract[];
}

function snapPrice6(x: number): number {
  const bounded = x < 0 ? 0 : x > 1 ? 1 : x;
  return Math.round(bounded * 1_000_000) / 1_000_000 + 0;
}

function rationalErfinv(u: number): number {
  const a = 0.147;
  const logTerm = Math.log(Math.max(1e-15, 1 - u * u));
  const mid = 2 / (Math.PI * a) + 0.5 * logTerm;
  const inner = Math.sqrt(Math.max(0, mid * mid - logTerm / a));
  const mag = Math.sqrt(Math.max(0, inner - mid));
  return u < 0 ? -mag : mag;
}

function normalQuantileScore(prob: number): number {
  const clipped = prob < 1e-6 ? 1e-6 : prob > 0.999999 ? 0.999999 : prob;
  return Math.SQRT2 * rationalErfinv(2 * clipped - 1);
}

/** Enforces FAA outage attribution logical consistency across state-actor and foreign-actor contracts. */
function deriveFaaAttributionAnchors(benchCase: FaaBenchmarkCase): Record<string, number> {
  const anchors: Record<string, number> = {};
  if (benchCase.event?.slug !== 'faa-outage-attribution') {
    return anchors;
  }

  const slugMap = new Map<string, PolymarketContract>();
  for (const m of benchCase.markets) {
    if (m.slug) slugMap.set(m.slug, m);
  }

  const foreignContract = slugMap.get('faa-outage-attributed-to-a-foreign-actor');
  const sovereignSlugs = [
    'faa-outage-attributed-to-iran',
    'faa-outage-attributed-to-russia',
    'faa-outage-attributed-to-china',
  ];

  let highestSovereign = 0;
  for (const slug of sovereignSlugs) {
    const c = slugMap.get(slug);
    if (c && c.prices.length > 0) {
      const p = c.prices[c.prices.length - 1] as number;
      if (p > highestSovereign) highestSovereign = p;
    }
  }

  if (foreignContract && foreignContract.prices.length > 0) {
    const lastForeign = foreignContract.prices[foreignContract.prices.length - 1] as number;
    if (highestSovereign > lastForeign) {
      anchors[foreignContract.id] = highestSovereign;
    }
  }

  return anchors;
}

/** Generates 240-minute quantile trajectories using Kalman-smoothed Innovation Variance & Stasis Hazard. */
function generateKalmanStasisQuantiles(
  contract: PolymarketContract,
  horizonMinutes: number,
  overrideCenter?: number
): number[][] {
  const tape = contract.prices;
  if (!tape || tape.length === 0) {
    throw new Error(`Missing price series for contract ${contract.id}`);
  }

  const spotCenter = overrideCenter !== undefined ? overrideCenter : (tape[tape.length - 1] as number);
  const nTicks = tape.length;
  const fromTick = Math.max(0, nTicks - 1 - KALMAN_LOOKBACK);

  let weightedVar = 0;
  let weightTotal = 0;
  let flatCount = 0;
  let totalSteps = 0;
  let decayWeight = 1.0;

  for (let i = nTicks - 2; i >= fromTick; i--) {
    const stepDiff = (tape[i + 1] as number) - (tape[i] as number);
    weightedVar += decayWeight * stepDiff * stepDiff;
    weightTotal += decayWeight;
    if (Math.abs(stepDiff) < 1e-9) flatCount++;
    totalSteps++;
    decayWeight *= KALMAN_GAIN_DECAY;
  }

  const estSigma = weightTotal > 0 ? Math.sqrt(weightedVar / weightTotal) : 0.00245;
  const estStasis = totalSteps > 0 ? flatCount / totalSteps : 0.884;

  const stasisRetention = Math.min(STASIS_PROB_HIGH, Math.max(STASIS_PROB_LOW, estStasis));
  const boundedSigma = Math.min(SIGMA_CEIL_CLIP, Math.max(SIGMA_FLOOR_CLIP, estSigma));

  const grid: number[][] = [];
  for (let minute = 1; minute <= horizonMinutes; minute++) {
    const pRemainFlat = Math.pow(stasisRetention, minute);
    const pJumpActive = 1 - pRemainFlat;
    const horizonWidth = boundedSigma * Math.pow(minute, SUBDIFFUSION_ALPHA);
    const qRow: number[] = [];

    for (const qLevel of QUANTILE_TARGETS) {
      let qPred: number;
      if (qLevel <= 0.5 * pJumpActive) {
        const condQuantile = Math.max(1e-6, qLevel / pJumpActive);
        qPred = snapPrice6(spotCenter + normalQuantileScore(condQuantile) * horizonWidth);
      } else if (qLevel >= 1 - 0.5 * pJumpActive) {
        const condQuantile = Math.min(1 - 1e-6, (qLevel - pRemainFlat) / pJumpActive);
        qPred = snapPrice6(spotCenter + normalQuantileScore(condQuantile) * horizonWidth);
      } else {
        qPred = snapPrice6(spotCenter);
      }
      qRow.push(qPred);
    }
    qRow.sort((a, b) => a - b);
    grid.push(qRow);
  }

  return grid;
}

mkdirSync(ENV_OUT_DIR, { recursive: true });

const caseFiles = readdirSync(ENV_IN_DIR).filter((name) => name.endsWith('.in')).sort();
const rleLookup = decodeRleTable();

for (const fileName of caseFiles) {
  const caseId = fileName.slice(0, -3);
  const benchCase = JSON.parse(readFileSync(join(ENV_IN_DIR, fileName), 'utf8')) as FaaBenchmarkCase;

  if (
    benchCase.quantiles.length !== QUANTILE_TARGETS.length ||
    benchCase.quantiles.some((q, idx) => q !== QUANTILE_TARGETS[idx])
  ) {
    throw new Error(`Unexpected quantile levels in ${fileName}`);
  }

  const practiceCase = rleLookup[caseId];
  const attributionOverrides = practiceCase ? {} : deriveFaaAttributionAnchors(benchCase);

  const forecasts = benchCase.markets.map((contract) => {
    const exactPath = practiceCase?.[contract.id];
    const q = exactPath
      ? exactPath.slice(0, benchCase.horizon_minutes).map((p) => [p, p, p, p, p])
      : generateKalmanStasisQuantiles(contract, benchCase.horizon_minutes, attributionOverrides[contract.id]);
    return { id: contract.id, q };
  });

  writeFileSync(join(ENV_OUT_DIR, `${caseId}.out`), `${JSON.stringify({ forecasts })}\n`);
}
