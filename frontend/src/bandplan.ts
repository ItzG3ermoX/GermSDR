// IARU / ITU Region 1 (Europe, Africa, Middle East, northern Asia) band plan.
// Frequencies in Hz. This drives the colored band-plan strip above the ruler so
// you can see at a glance which service (ham 40m/31m, FM broadcast, AM, airband,
// ...) you're tuned into -- the way KiwiSDR labels its waterfall.
//
// `kind` picks the colour; `label` is the short tag drawn in the segment.

export type BandKind =
  | 'ham'        // amateur radio
  | 'broadcast'  // shortwave / FM / LW-MW broadcast
  | 'aero'       // aeronautical
  | 'marine'     // maritime
  | 'utility'    // time signals, beacons, misc
  | 'cb';        // citizens band

export interface Band {
  lo: number;
  hi: number;
  label: string;
  kind: BandKind;
}

// Colour per kind (KiwiSDR-ish, muted but distinct). [fill, edge/text].
export const BAND_COLORS: Record<BandKind, { fill: string; text: string }> = {
  ham:       { fill: 'rgba(56, 189, 248, 0.30)', text: '#7dd3fc' }, // cyan
  broadcast: { fill: 'rgba(248, 113, 113, 0.28)', text: '#fca5a5' }, // red
  aero:      { fill: 'rgba(167, 139, 250, 0.28)', text: '#c4b5fd' }, // violet
  marine:    { fill: 'rgba(45, 212, 191, 0.28)', text: '#5eead4' }, // teal
  utility:   { fill: 'rgba(148, 163, 184, 0.26)', text: '#cbd5e1' }, // slate
  cb:        { fill: 'rgba(250, 204, 21, 0.26)', text: '#fde047' },  // yellow
};

const M = 1_000_000;
const k = 1_000;

// Sorted by lo. Amateur bands carry their classic metre-band names so "40m",
// "31m", "20m" etc. show up exactly as asked. Broadcast SW bands use the metre
// band the listener expects.
export const REGION1_BANDS: Band[] = [
  // --- Long / medium wave ---
  { lo: 148.5 * k, hi: 283.5 * k, label: 'LW AM', kind: 'broadcast' },
  { lo: 526.5 * k, hi: 1606.5 * k, label: 'MW AM', kind: 'broadcast' },

  // --- HF amateur + broadcast (metre bands) ---
  { lo: 1.81 * M, hi: 2.0 * M, label: '160m', kind: 'ham' },
  { lo: 2.3 * M, hi: 2.495 * M, label: '120m', kind: 'broadcast' },
  { lo: 3.2 * M, hi: 3.4 * M, label: '90m', kind: 'broadcast' },
  { lo: 3.5 * M, hi: 3.8 * M, label: '80m', kind: 'ham' },
  { lo: 3.95 * M, hi: 4.0 * M, label: '75m', kind: 'broadcast' },
  { lo: 4.75 * M, hi: 4.995 * M, label: '60m bc', kind: 'broadcast' },
  { lo: 5.351 * M, hi: 5.366 * M, label: '60m', kind: 'ham' },
  { lo: 5.9 * M, hi: 6.2 * M, label: '49m', kind: 'broadcast' },
  { lo: 7.0 * M, hi: 7.2 * M, label: '40m', kind: 'ham' },
  { lo: 7.2 * M, hi: 7.45 * M, label: '41m', kind: 'broadcast' },
  { lo: 9.4 * M, hi: 9.9 * M, label: '31m', kind: 'broadcast' },
  { lo: 10.1 * M, hi: 10.15 * M, label: '30m', kind: 'ham' },
  { lo: 11.6 * M, hi: 12.1 * M, label: '25m', kind: 'broadcast' },
  { lo: 13.57 * M, hi: 13.87 * M, label: '22m', kind: 'broadcast' },
  { lo: 14.0 * M, hi: 14.35 * M, label: '20m', kind: 'ham' },
  { lo: 15.1 * M, hi: 15.8 * M, label: '19m', kind: 'broadcast' },
  { lo: 17.48 * M, hi: 17.9 * M, label: '16m', kind: 'broadcast' },
  { lo: 18.068 * M, hi: 18.168 * M, label: '17m', kind: 'ham' },
  { lo: 21.0 * M, hi: 21.45 * M, label: '15m', kind: 'ham' },
  { lo: 21.45 * M, hi: 21.85 * M, label: '13m', kind: 'broadcast' },
  { lo: 24.89 * M, hi: 24.99 * M, label: '12m', kind: 'ham' },
  { lo: 25.67 * M, hi: 26.1 * M, label: '11m', kind: 'broadcast' },
  { lo: 26.965 * M, hi: 27.405 * M, label: 'CB', kind: 'cb' },
  { lo: 28.0 * M, hi: 29.7 * M, label: '10m', kind: 'ham' },

  // --- VHF ---
  { lo: 50.0 * M, hi: 52.0 * M, label: '6m', kind: 'ham' },
  { lo: 87.5 * M, hi: 108.0 * M, label: 'FM', kind: 'broadcast' },
  { lo: 108.0 * M, hi: 117.975 * M, label: 'VOR/ILS', kind: 'aero' },
  { lo: 118.0 * M, hi: 137.0 * M, label: 'Airband', kind: 'aero' },
  { lo: 144.0 * M, hi: 146.0 * M, label: '2m', kind: 'ham' },
  { lo: 156.0 * M, hi: 162.05 * M, label: 'Marine', kind: 'marine' },
  { lo: 430.0 * M, hi: 440.0 * M, label: '70cm', kind: 'ham' },
];

// Bands overlapping [lo, hi], for the visible window. Linear scan -- the list is
// tiny, so a binary search would be premature.
export function bandsInRange(lo: number, hi: number): Band[] {
  return REGION1_BANDS.filter((b) => b.hi >= lo && b.lo <= hi);
}
