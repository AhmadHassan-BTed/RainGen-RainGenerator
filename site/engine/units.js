/**
 * Human units and labels for the weather engine (MASTER_PLAN §6.3, §6.9).
 *
 * - Beaufort scale: WMO-No. 306 / WMO Manual on Marine Meteorological Services,
 *   land descriptions after the UK Met Office table. Continuous mapping uses the
 *   WMO empirical relation v = 0.836 * B^1.5 m/s.
 * - Rain intensity labels: WMO-No. 8 (CIMO Guide, slight < 2.5, moderate 2.5-10,
 *   heavy 10-50, intense > 50 mm/h) and the AMS Glossary (light <= 2.5,
 *   moderate 2.6-7.6, heavy > 7.6 mm/h).
 * - Flash-to-bang: c(T) = 331.3 * sqrt(1 + T/273.15) m/s (2.91 s/km at 20 C).
 */

export const BEAUFORT = [
  { force: 0, name: 'Calm', min: 0.0, max: 0.2, land: 'Calm; smoke rises vertically.' },
  { force: 1, name: 'Light air', min: 0.3, max: 1.5, land: 'Direction shown by smoke drift but not by wind vanes.' },
  { force: 2, name: 'Light breeze', min: 1.6, max: 3.3, land: 'Wind felt on face; leaves rustle; ordinary vanes moved by wind.' },
  { force: 3, name: 'Gentle breeze', min: 3.4, max: 5.4, land: 'Leaves and small twigs in constant motion; light flags extended.' },
  { force: 4, name: 'Moderate breeze', min: 5.5, max: 7.9, land: 'Raises dust and loose paper; small branches are moved.' },
  { force: 5, name: 'Fresh breeze', min: 8.0, max: 10.7, land: 'Small trees in leaf begin to sway; crested wavelets form on inland waters.' },
  { force: 6, name: 'Strong breeze', min: 10.8, max: 13.8, land: 'Large branches in motion; whistling heard in telegraph wires; umbrellas used with difficulty.' },
  { force: 7, name: 'Near gale', min: 13.9, max: 17.1, land: 'Whole trees in motion; inconvenience felt when walking against the wind.' },
  { force: 8, name: 'Gale', min: 17.2, max: 20.7, land: 'Twigs break off trees; progress generally impeded.' },
  { force: 9, name: 'Strong gale', min: 20.8, max: 24.4, land: 'Slight structural damage occurs; chimney pots and slates removed.' },
  { force: 10, name: 'Storm', min: 24.5, max: 28.4, land: 'Trees uprooted; considerable structural damage occurs.' },
  { force: 11, name: 'Violent storm', min: 28.5, max: 32.6, land: 'Very rarely experienced; accompanied by widespread damage.' },
  { force: 12, name: 'Hurricane', min: 32.7, max: Infinity, land: 'Devastation.' },
];

const COMPASS16 = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
const COMPASS16_NAMES = [
  'north', 'north-northeast', 'northeast', 'east-northeast', 'east', 'east-southeast', 'southeast', 'south-southeast',
  'south', 'south-southwest', 'southwest', 'west-southwest', 'west', 'west-northwest', 'northwest', 'north-northwest',
];

function num(x, d = 0) {
  const v = Number(x);
  return Number.isFinite(v) ? v : d;
}

/** Continuous Beaufort number (0..12+) from wind speed in m/s: B = (v / 0.836)^(2/3). */
export function beaufortFromMps(mps) {
  const v = Math.max(0, num(mps));
  return Math.min(17, Math.pow(v / 0.836, 2 / 3));
}

/** Wind speed in m/s for a (possibly fractional) Beaufort number: v = 0.836 * B^1.5. */
export function mpsFromBeaufort(b) {
  const B = Math.max(0, num(b));
  return 0.836 * Math.pow(B, 1.5);
}

/** Integer Beaufort force using the WMO class limits (m/s). */
export function beaufortForce(mps) {
  const v = Math.max(0, num(mps));
  for (const row of BEAUFORT) {
    if (v < row.max + 0.05) return row.force;
  }
  return 12;
}

/** The BEAUFORT table row for a wind speed. */
export function beaufortInfo(mps) {
  return BEAUFORT[beaufortForce(mps)];
}

/** Human rain-intensity label. system: 'wmo' (default) or 'ams'. The number is always shown alongside by the UI. */
export function rainLabel(mmh, system = 'wmo') {
  const r = Math.max(0, num(mmh));
  if (r <= 0.005) return 'No rain';
  if (String(system).toLowerCase() === 'ams') {
    if (r <= 2.5) return 'Light rain';
    if (r <= 7.6) return 'Moderate rain';
    if (r < 50) return 'Heavy rain';
    if (r < 100) return 'Violent rain';
    return 'Cloudburst';
  }
  if (r < 0.25) return 'Very slight rain';
  if (r < 2.5) return 'Slight rain';
  if (r < 10) return 'Moderate rain';
  if (r < 50) return 'Heavy rain';
  if (r < 100) return 'Intense rain';
  return 'Cloudburst';
}

/** Speed of sound in air (m/s) at temperature tempC. */
export function speedOfSound(tempC = 20) {
  return 331.3 * Math.sqrt(Math.max(0.05, 1 + num(tempC, 20) / 273.15));
}

/** Seconds between flash and thunder for a strike km away at tempC (default 20 C). */
export function flashToBangSeconds(km, tempC = 20) {
  return (Math.max(0, num(km)) * 1000) / speedOfSound(tempC);
}

/** Distance in km from a counted flash-to-bang delay. */
export function kmFromFlashToBang(seconds, tempC = 20) {
  return (Math.max(0, num(seconds)) * speedOfSound(tempC)) / 1000;
}

function wrap360(deg) {
  const d = num(deg) % 360;
  return d < 0 ? d + 360 : d;
}

/** 16-point compass abbreviation ('N', 'NNE', ... 'NNW') for a bearing in degrees true. */
export function compass16(deg) {
  return COMPASS16[Math.round(wrap360(deg) / 22.5) % 16];
}

/** 16-point compass name in words ('west', 'north-northeast', ...). */
export function compassName(deg) {
  return COMPASS16_NAMES[Math.round(wrap360(deg) / 22.5) % 16];
}

/** Bearing in degrees for a compass abbreviation ('WSW' -> 247.5); returns NaN if unknown. */
export function degFromCompass(label) {
  const i = COMPASS16.indexOf(String(label || '').trim().toUpperCase());
  return i < 0 ? NaN : i * 22.5;
}

/**
 * Format a wind speed. system: 'metric' (km/h, default), 'si'|'mps' (m/s),
 * 'imperial'|'mph', 'knots'|'kn', 'beaufort'.
 */
export function formatSpeed(mps, system = 'metric') {
  const v = Math.max(0, num(mps));
  switch (String(system).toLowerCase()) {
    case 'si':
    case 'mps':
    case 'm/s':
      return `${v < 10 ? v.toFixed(1) : Math.round(v)} m/s`;
    case 'imperial':
    case 'us':
    case 'uk':
    case 'mph':
      return `${Math.round(v * 2.236936)} mph`;
    case 'knots':
    case 'kn':
    case 'kt':
    case 'aviation':
      return `${Math.round(v * 1.943844)} kn`;
    case 'beaufort':
    case 'bft':
      return `Force ${beaufortForce(v)}`;
    default:
      return `${Math.round(v * 3.6)} km/h`;
  }
}

/** Format a distance given in km. system 'imperial' uses miles. */
export function formatDistance(km, system = 'metric') {
  const d = Math.max(0, num(km));
  const s = String(system).toLowerCase();
  if (s === 'imperial' || s === 'us' || s === 'mph') {
    const mi = d * 0.621371;
    return mi < 0.2 ? `${Math.round(mi * 5280)} ft` : `${mi < 10 ? mi.toFixed(1) : Math.round(mi)} mi`;
  }
  return d < 1 ? `${Math.round(d * 1000)} m` : `${d < 10 ? d.toFixed(1) : Math.round(d)} km`;
}

/** Format a temperature given in C. system 'imperial' uses F. */
export function formatTemperature(c, system = 'metric') {
  const t = num(c, 12);
  const s = String(system).toLowerCase();
  if (s === 'imperial' || s === 'us' || s === 'f') return `${Math.round(t * 1.8 + 32)} °F`;
  return `${Math.round(t)} °C`;
}

/** Dew point (C) by the Magnus formula (Alduchov & Eskridge 1996 constants). */
export function dewPoint(tempC, rhPct) {
  const T = num(tempC, 12);
  const rh = Math.min(100, Math.max(1, num(rhPct, 90)));
  const a = 17.625;
  const b = 243.04;
  const g = Math.log(rh / 100) + (a * T) / (b + T);
  return (b * g) / (a - g);
}

/** Spoken/caption description: "Force 5, fresh breeze, 30 km/h from the west". */
export function describeWind(mps, fromDeg, system = 'metric') {
  const info = beaufortInfo(mps);
  const dir = Number.isFinite(Number(fromDeg)) ? ` from the ${compassName(fromDeg)}` : '';
  return `Force ${info.force}, ${info.name.toLowerCase()}, ${formatSpeed(mps, system)}${dir}`;
}

/** Spoken/caption description of rain: "Moderate rain, 5 mm/h". */
export function describeRain(mmh, system = 'wmo') {
  const r = Math.max(0, num(mmh));
  const n = r < 1 ? r.toFixed(2) : r < 10 ? r.toFixed(1) : Math.round(r);
  return `${rainLabel(r, system)}, ${n} mm/h`;
}
