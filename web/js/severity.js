/**
 * Severity scoring, shared by the web app and the PDF report generator.
 *
 * Confidence-weighted rather than a raw count, so five borderline detections do not
 * outrank one certain crack. Class weights live in the manifest so the scoring can be
 * retuned without touching code - they encode engineering judgement (a structural crack
 * matters more than surface soiling), not anything the model learned.
 *
 * The thresholds match `score_to_label` in report_generator.py. Change them in both places
 * or the web verdict and the PDF verdict will disagree on the same image.
 */

export const SEVERITY = {
  NONE: { key: 'none', label: 'No defects found', rank: 0 },
  MINOR: { key: 'minor', label: 'Minor wear', rank: 1 },
  MODERATE: { key: 'moderate', label: 'Moderate damage', rank: 2 },
  SEVERE: { key: 'severe', label: 'Severe damage', rank: 3 },
};

const MINOR_BELOW = 2;
const MODERATE_BELOW = 5;

/** Weighted severity score for one image's detections. */
export function scoreDetections(detections, weights = {}) {
  return detections.reduce(
    (total, d) => total + (weights[d.label] ?? 1) * d.confidence,
    0,
  );
}

/** Map a score onto a severity band. */
export function scoreToSeverity(score) {
  if (score <= 0) return SEVERITY.NONE;
  if (score < MINOR_BELOW) return SEVERITY.MINOR;
  if (score < MODERATE_BELOW) return SEVERITY.MODERATE;
  return SEVERITY.SEVERE;
}

/** Convenience: detections -> { score, severity }. */
export function assess(detections, weights) {
  const score = scoreDetections(detections, weights);
  return { score, severity: scoreToSeverity(score) };
}

/**
 * Roll individual image results up into one inspection verdict.
 * Mirrors the overall-status logic in report_generator.py.
 */
export function summarise(results) {
  const analysed = results.filter((r) => r.status === 'analysed');
  const counts = { none: 0, minor: 0, moderate: 0, severe: 0 };
  for (const result of analysed) counts[result.severity.key] += 1;

  const total = analysed.length;
  if (!total) {
    return { counts, total, overall: 'No images analysed', defectRate: 0 };
  }

  const severePct = (counts.severe / total) * 100;
  const moderatePct = (counts.moderate / total) * 100;
  const defectPct = ((counts.severe + counts.moderate + counts.minor) / total) * 100;

  let overall;
  if (severePct >= 10) overall = 'Severe damage detected';
  else if (severePct > 0 || moderatePct >= 20) overall = 'Moderate damage detected';
  else if (defectPct >= 10) overall = 'Minor wear detected';
  else overall = 'Majority healthy - minor issues noted';

  return { counts, total, overall, defectRate: defectPct };
}
