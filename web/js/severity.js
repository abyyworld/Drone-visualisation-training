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
  NONE: { key: 'none', label: 'No findings in this image', rank: 0 },
  MINOR: { key: 'minor', label: 'Minor', rank: 1 },
  MODERATE: { key: 'moderate', label: 'Moderate', rank: 2 },
  SEVERE: { key: 'severe', label: 'Severe', rank: 3 },
};

/**
 * What each band is called, per subject.
 *
 * "Moderate damage" is the right words for a blade and the wrong ones for a crowd, which is
 * not damaged, or a fire, where the finding is what is burning and who is near it. The
 * bands and the arithmetic are shared; only the nouns change.
 *
 * The zero band is the one that matters most. It has to describe a null result about one
 * image and never the state of the thing photographed - a rule station/core/safety.py
 * enforces across this repository, and the reason the phrasing here is so careful.
 */
const LABELS = {
  turbine: { none: 'No defects found', minor: 'Minor wear', moderate: 'Moderate damage', severe: 'Severe damage' },
  solar: { none: 'No defects found', minor: 'Minor wear', moderate: 'Moderate damage', severe: 'Severe damage' },
  crowd: {
    none: 'No pressure pattern scored in this frame',
    minor: 'Worth watching',
    moderate: 'Under pressure',
    severe: 'Needs someone now',
  },
  wildfire: {
    none: 'Nothing visible in this frame',
    minor: 'Minor activity',
    moderate: 'Active',
    severe: 'Active, with people or property',
  },
};

/** The band's name for a given subject, falling back to the neutral one. */
export function severityLabel(severity, domain) {
  return LABELS[domain]?.[severity.key] ?? severity.label;
}

/**
 * The one-line verdict across a set. Mirrors OVERALL_LABELS in report_generator.py.
 *
 * Keyed by band rather than matched on words. The PDF used to pick the badge colour by
 * searching the sentence for "Severe", which stopped working the moment the wording became
 * subject-specific - the same trap that left the gate telling people to upload a turbine
 * long after it had learned two more subjects.
 */
const OVERALL = {
  turbine: {
    severe: 'Severe damage detected', moderate: 'Moderate damage detected',
    minor: 'Minor wear detected', none: 'Majority healthy - minor issues noted',
  },
  solar: {
    severe: 'Severe damage detected', moderate: 'Moderate damage detected',
    minor: 'Minor wear detected', none: 'Majority healthy - minor issues noted',
  },
  crowd: {
    severe: 'Crowd pressure needing someone now',
    moderate: 'Crowd under pressure in places',
    minor: 'Some areas worth watching',
    none: 'No pressure pattern scored in the frames reviewed',
  },
  wildfire: {
    severe: 'Active fire with people or property in the frames',
    moderate: 'Active fire in the frames reviewed',
    minor: 'Minor activity in the frames reviewed',
    none: 'Nothing visible in the frames reviewed',
  },
  default: {
    severe: 'Severe findings', moderate: 'Moderate findings',
    minor: 'Minor findings', none: 'No findings in the images reviewed',
  },
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

  let band;
  if (severePct >= 10) band = 'severe';
  else if (severePct > 0 || moderatePct >= 20) band = 'moderate';
  else if (defectPct >= 10) band = 'minor';
  else band = 'none';

  // One subject per set, taken from the results rather than assumed. A mixed batch falls
  // back to neutral wording rather than describing a fire as damage.
  const subjects = new Set(analysed.map((r) => r.domain).filter(Boolean));
  const domain = subjects.size === 1 ? [...subjects][0] : 'default';
  const overall = (OVERALL[domain] ?? OVERALL.default)[band];

  return { counts, total, overall, band, defectRate: defectPct };
}
