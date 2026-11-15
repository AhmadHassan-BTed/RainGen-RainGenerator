/**
 * "Describe your weather": a deterministic parser from plain words to a
 * rainscene patch (MASTER_PLAN 6.7). Known vocabulary only, no network, no ML:
 * rain words and WMO intensities, mm/h and in/h, Beaufort names, "force 6",
 * "30 mph gusts", "from the west" / "westerly", thunder distance words, km/miles,
 * flash-to-bang seconds (distance = t * c(T)), surfaces, shelters, places, time,
 * season, temperature, humidity, listening mode and a METAR subset
 * (dddffGggKT, +TSRA, VCTS, TT/TdTd; ASOS: TS <= 5 NM, VCTS 5-10 NM).
 */
import { mpsFromBft, bftFromMps, bftInfo, compassWord, rainWord, flashBang, fmtNum, speedOfSound } from './a11y.js';

const DIRS = {
  n: 0, nne: 22.5, ne: 45, ene: 67.5, e: 90, ese: 112.5, se: 135, sse: 157.5,
  s: 180, ssw: 202.5, sw: 225, wsw: 247.5, w: 270, wnw: 292.5, nw: 315, nnw: 337.5,
};

function dirKey(word) {
  const w = word.toLowerCase().replace(/[\s-]+/g, '').replace(/erly$|ern$/, '');
  const map = {
    north: 'n', south: 's', east: 'e', west: 'w',
    northeast: 'ne', northwest: 'nw', southeast: 'se', southwest: 'sw',
    northnortheast: 'nne', eastnortheast: 'ene', eastsoutheast: 'ese', southsoutheast: 'sse',
    southsouthwest: 'ssw', westsouthwest: 'wsw', westnorthwest: 'wnw', northnorthwest: 'nnw',
  };
  if (map[w]) return map[w];
  if (DIRS[w] !== undefined) return w;
  return null;
}

const DIR_WORD = '(north[\\s-]?north[\\s-]?east|east[\\s-]?north[\\s-]?east|east[\\s-]?south[\\s-]?east|south[\\s-]?south[\\s-]?east|south[\\s-]?south[\\s-]?west|west[\\s-]?south[\\s-]?west|west[\\s-]?north[\\s-]?west|north[\\s-]?north[\\s-]?west|north[\\s-]?east|north[\\s-]?west|south[\\s-]?east|south[\\s-]?west|north|south|east|west|nne|ene|ese|sse|ssw|wsw|wnw|nnw|ne|nw|se|sw|n|s|e|w)';

const num = (s) => parseFloat(String(s).replace(',', '.'));

/* ------------------------------------------------------------------ vocab */

const PRECIP_RULES = [
  [/\b(no rain|not raining|rain (?:has )?stopped|stopped raining|dry weather|after the (?:rain|storm))\b/, { type: 'none', rate: 0.02 }],
  [/\bcloudburst\b/, { type: 'rain', rate: 110 }],
  [/\b(torrential|deluge|biblical|monsoon(?:al)?|bucketing|sheeting)\b/, { type: 'rain', rate: 60 }],
  [/\b(very heavy|extreme|violent)\s+(rain|showers?|downpour)\b/, { type: 'rain', rate: 60 }],
  [/\b(downpour|pouring|lashing|hammering)\b/, { type: 'rain', rate: 35 }],
  [/\bheavy\s+showers?\b/, { type: 'showers', rate: 20, character: 'showery' }],
  [/\bheavy\s+rain\b/, { type: 'rain', rate: 20 }],
  [/\bfreezing\s+(rain|drizzle)\b/, { type: 'freezing_rain', rate: 2 }],
  [/\b(sleet|ice pellets)\b/, { type: 'sleet', rate: 2 }],
  [/\bgraupel\b/, { type: 'graupel', rate: 2 }],
  [/\bheavy\s+snow|blizzard\b/, { type: 'snow', rate: 4 }],
  [/\blight\s+snow|flurr(?:y|ies)\b/, { type: 'snow', rate: 0.5 }],
  [/\bsnow\w*\b/, { type: 'snow', rate: 1.5 }],
  [/\bhail\w*\b/, { type: 'hail', rate: 30 }],
  [/\bheavy\s+drizzle\b/, { type: 'drizzle', rate: 0.9 }],
  [/\blight\s+drizzle\b/, { type: 'drizzle', rate: 0.1 }],
  [/\bdrizzl\w*\b/, { type: 'drizzle', rate: 0.4 }],
  [/\b(mizzle|spitting|sprinkl\w*|misty rain|mist)\b/, { type: 'drizzle', rate: 0.2 }],
  [/\b(moderate|steady)\s+rain\b/, { type: 'rain', rate: 5 }],
  [/\b(light|slight|soft|gentle|fine)\s+rain\b/, { type: 'rain', rate: 1.5 }],
  [/\b(light|scattered)\s+showers?\b/, { type: 'showers', rate: 2, character: 'showery' }],
  [/\bshowers?\b/, { type: 'showers', rate: 6, character: 'showery' }],
  [/\bthunderstorm\w*|\btstorm\b/, { type: 'rain', rate: 25, character: 'showery', drops: 'fat' }],
  [/\b(rain|raining|rainy|rainfall|wet)\b/, { type: 'rain', rate: 4 }],
];

const BFT_NAMES = [
  ['violent storm', 11], ['hurricane', 12], ['whole gale', 10], ['storm force', 10], ['strong gale', 9], ['severe gale', 9],
  ['near gale', 7], ['moderate gale', 7], ['fresh gale', 8], ['gale force', 8], ['gale-force', 8], ['gales?', 8],
  ['strong breeze', 6], ['fresh breeze', 5], ['moderate breeze', 4], ['gentle breeze', 3], ['light breeze', 2],
  ['light air', 1], ['dead calm', 0], ['calm (?:wind|air|conditions)', 0], ['no wind', 0], ['windless', 0], ['still air', 0],
  ['howling', 9], ['very windy', 7], ['blustery', 6], ['windy', 5], ['breezy', 4], ['light wind', 2], ['gentle wind', 3],
];

const SURFACE_RULES = [
  [/\b(tin|metal|corrugated|iron|steel)\s+roof\b/, 'metal'],
  [/\b(metal|tin|steel|bucket|bins?)\b/, 'metal'],
  [/\b(leaves|leafy|foliage|canopy|broadleaf|oak|beech|maple)\b/, 'leaves'],
  [/\b(pine|needles|spruce|fir|conifer\w*)\b/, 'needles'],
  [/\b(grass|lawn|meadow|field|moor)\b/, 'grass'],
  [/\b(mud|muddy|soil|earth|dirt)\b/, 'soil'],
  [/\b(puddles?|pond|lake|water|river|stream|sea|ocean)\b/, 'puddle'],
  [/\b(asphalt|tarmac|road|street)\b/, 'asphalt'],
  [/\b(pavement|sidewalk|concrete|patio|parking)\b/, 'concrete'],
  [/\b(gravel|shingle|stones|scree)\b/, 'gravel'],
  [/\b(tiles?|tiled)\b/, 'tile'],
  [/\b(glass|window(?:s|pane)?|skylight|windscreen|windshield|greenhouse)\b/, 'glass'],
  [/\b(tent|tarp|canvas|fabric|umbrella|awning|hood)\b/, 'fabric'],
  [/\b(wood|wooden|deck|decking|porch|cabin|shed)\b/, 'wood'],
  [/\b(car|cars|bonnet|hood of the car)\b/, 'car_body'],
  [/\bsnow\w*\b/, 'snow'],
];

const SHELTER_RULES = [
  [/\b(window|windows) (?:is |slightly |half )?(?:ajar|cracked)\b|\bcracked window\b/, { shelter: 'room_open', window: 0.25 }],
  [/\b(window|windows) (?:wide )?open\b|\bopen window\b/, { shelter: 'room_open', window: 0.7 }],
  [/\b(window|windows) (?:shut|closed)\b|\bclosed window\b/, { shelter: 'room_closed', window: 0 }],
  [/\b(?:in|inside) (?:a|the|my) tent\b|\btent\b/, { shelter: 'tent' }],
  [/\b(?:in|inside) (?:a|the|my) (?:car|van|truck)\b|\bwipers?\b/, { shelter: 'car' }],
  [/\bumbrella\b/, { shelter: 'umbrella' }],
  [/\b(porch|veranda|verandah|balcony|awning|gazebo|stoop)\b/, { shelter: 'porch' }],
  [/\bunder (?:a|an|the) (?:tree|oak|pine|trees|canopy)\b/, { shelter: 'tree' }],
  [/\b(attic|loft|cabin|tin roof|metal roof|shed|barn|caravan)\b/, { shelter: 'attic' }],
  [/\bcave\b/, { shelter: 'cave' }],
  [/\b(underpass|under (?:a|the) bridge|tunnel)\b/, { shelter: 'underpass' }],
  [/\b(indoors|inside|bedroom|living room|at home|in bed|my room|the room|kitchen|study)\b/, { shelter: 'room_closed' }],
  [/\b(outside|outdoors|out in it|in the open|open field|exposed|standing in the rain)\b/, { shelter: 'open' }],
];

const PLACE_RULES = [
  [/\b(pine|spruce|fir|conifer\w*|boreal|taiga)\b.*\b(forest|woods?|woodland)\b|\b(pine|conifer)\s+(forest|woods?)\b/, 'conifer_forest'],
  [/\b(forest|woods|woodland|grove)\b/, 'deciduous_forest'],
  [/\b(jungle|rain ?forest|tropical|tropics)\b/, 'jungle'],
  [/\b(beach|coast\w*|seaside|harbou?r|shore|cliff)\b/, 'coast'],
  [/\b(lake|pond|loch|lakeside)\b/, 'lake'],
  [/\b(alley|alleyway|back street|lane)\b/, 'alley'],
  [/\b(city|downtown|urban|street|avenue|tokyo|london|new york|paris)\b/, 'city_street'],
  [/\b(suburb\w*|cul-de-sac|neighbou?rhood)\b/, 'suburb'],
  [/\b(mountain|ridge|highlands?|moor\w*|summit|fell)\b/, 'mountain_ridge'],
  [/\b(desert|dunes|canyon)\b/, 'desert'],
  [/\b(farm|barn|paddock|orchard)\b/, 'farm'],
  [/\b(meadow|field|prairie|plain)\b/, 'meadow'],
  [/\b(garden|backyard|yard)\b/, 'garden'],
];

const TIME_RULES = [
  [/\bmidnight\b/, 0], [/\blate (?:at )?night\b|\bsmall hours\b/, 2], [/\b(dawn|sunrise|daybreak|first light)\b/, 6],
  [/\bearly morning\b/, 6.5], [/\bmorning\b/, 9], [/\b(noon|midday|lunchtime)\b/, 12], [/\bafternoon\b/, 15],
  [/\b(dusk|sunset|twilight)\b/, 19], [/\bevening\b/, 20.5], [/\b(night|tonight|nighttime|night-time)\b/, 23],
];

const SEASON_RULES = [
  [/\bspring(?:time)?\b/, 'spring'], [/\bsummer(?:time|y)?\b/, 'summer'],
  [/\bautumn\w*\b|\bin the fall\b|\bfall (?:season|evening|night|day|afternoon)\b/, 'autumn'], [/\bwinter\w*\b/, 'winter'],
];

const MODE_RULES = [
  [/\b(sleep|sleepy|bedtime|nap|fall asleep)\b/, 'sleep'],
  [/\b(focus|study|studying|concentrat\w*|deep work)\b/, 'focus'],
  [/\b(asmr|tingl\w*)\b/, 'asmr'],
  [/\b(relax\w*|unwind|calm(?:ing)?|cozy|cosy|soothing)\b/, 'calm'],
];

/* ------------------------------------------------------------------ parser */

/**
 * @returns {{patch: Object<string, any>, chips: Array<{label:string, text:string}>, understood: boolean}}
 */
export function parseDescription(input, scene) {
  const raw = String(input || '').trim();
  const text = ` ${raw.toLowerCase().replace(/[“”"']/g, '').replace(/\s+/g, ' ')} `;
  const patch = {};
  const chips = [];
  const chip = (label, t) => chips.push({ label, text: t });
  const tempNow = () => patch['weather.temperature_c'] ?? scene?.weather?.temperature_c ?? 15;
  const thunderCtx = /\b(thunder\w*|lightning|storm\w*|tstorm|electrical)\b/.test(text);

  /* METAR subset (uppercase tokens in the original text) */
  const metarWind = /\b(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?(KT|MPS)\b/.exec(raw);
  if (metarWind) {
    const toMps = metarWind[4] === 'MPS' ? 1 : 0.514444;
    const ff = num(metarWind[2]) * toMps;
    patch['weather.wind.speed_mps'] = ff;
    if (metarWind[1] !== 'VRB') patch['weather.wind.from_deg'] = num(metarWind[1]);
    if (metarWind[3]) patch['weather.wind.gust_factor'] = Math.min(2, Math.max(1, (num(metarWind[3]) * toMps) / Math.max(0.5, ff)));
    chip('METAR wind', `${metarWind[1]}° ${metarWind[2]}${metarWind[3] ? ' G' + metarWind[3] : ''} ${metarWind[4]}`);
  }
  const metarWx = /(?:^|\s)([+-]|VC)?(TS|SH|FZ)?(RA|DZ|SN|GR|GS|PL|SG)?(RA|DZ|SN)?(?=\s|$)/g;
  let mw;
  while ((mw = metarWx.exec(raw))) {
    if (mw[0].length === 0) metarWx.lastIndex++;
    const [tok, inten, desc, p1] = mw;
    if (!desc && !p1) continue;
    if (tok.trim().length < 2) continue;
    if (desc === 'TS' || (inten === 'VC' && desc === 'TS')) {
      patch['weather.thunder.enabled'] = true;
      patch['weather.thunder.distance_km'] = inten === 'VC' ? 14 : 6;
    }
    const rates = { RA: [1.5, 5, 20], DZ: [0.1, 0.4, 0.9], SN: [0.5, 1.5, 4], GR: [15, 30, 50], GS: [1, 2, 4], PL: [1, 2, 4], SG: [0.2, 0.5, 1] };
    if (p1 && rates[p1]) {
      const idx = inten === '-' ? 0 : inten === '+' ? 2 : 1;
      patch['weather.precipitation.rate_mm_h'] = rates[p1][idx];
      patch['weather.precipitation.type'] = { RA: desc === 'FZ' ? 'freezing_rain' : desc === 'SH' ? 'showers' : 'rain', DZ: desc === 'FZ' ? 'freezing_rain' : 'drizzle', SN: 'snow', GR: 'hail', GS: 'graupel', PL: 'sleet', SG: 'snow' }[p1];
      chip('METAR weather', tok.trim());
    }
  }
  const metarT = /\b(M?\d{2})\/(M?\d{2})\b/.exec(raw);
  if (metarT && metarWind) {
    const t = num(metarT[1].replace('M', '-'));
    const td = num(metarT[2].replace('M', '-'));
    patch['weather.temperature_c'] = t;
    const rh = 100 * Math.exp((17.625 * td) / (243.04 + td)) / Math.exp((17.625 * t) / (243.04 + t));
    patch['weather.humidity_pct'] = Math.round(Math.max(10, Math.min(100, rh)));
  }

  /* precipitation */
  let precipSet = false;
  for (const [re, r] of PRECIP_RULES) {
    if (!re.test(text)) continue;
    if (patch['weather.precipitation.type'] === undefined) patch['weather.precipitation.type'] = r.type;
    if (patch['weather.precipitation.rate_mm_h'] === undefined) patch['weather.precipitation.rate_mm_h'] = r.rate;
    if (r.character && patch['weather.precipitation.character'] === undefined) patch['weather.precipitation.character'] = r.character;
    if (r.drops && patch['weather.precipitation.drops'] === undefined) patch['weather.precipitation.drops'] = r.drops;
    precipSet = true;
    break;
  }
  const mmh = /(\d+(?:[.,]\d+)?)\s*(?:mm\s*\/\s*h(?:r|our)?\b|mm\s*(?:per|an|a)\s*h(?:ou)?r\b|mmh\b|millimet(?:er|re)s?\s*(?:per|an|a)\s*hour)/.exec(text);
  const inh = /(\d+(?:[.,]\d+)?)\s*(?:in(?:ch(?:es)?)?\s*(?:\/|per|an|a)\s*h(?:ou)?r)\b/.exec(text);
  if (mmh || inh) {
    const r = mmh ? num(mmh[1]) : num(inh[1]) * 25.4;
    patch['weather.precipitation.rate_mm_h'] = Math.max(0.02, Math.min(150, r));
    if (!precipSet || patch['weather.precipitation.type'] === 'none') patch['weather.precipitation.type'] = r < 0.6 ? 'drizzle' : 'rain';
    precipSet = true;
  }
  if (precipSet) {
    const r = patch['weather.precipitation.rate_mm_h'];
    const t = patch['weather.precipitation.type'];
    chip('Precipitation', t === 'none' ? 'dry, drips only' : `${t.replace('_', ' ')}, ${fmtNum(r)} mm/h (${rainWord(r)})`);
  }

  if (/\b(fat|big|large|huge|heavy|plump)\s+(rain)?\s*drops?\b/.test(text)) patch['weather.precipitation.drops'] = 'fat';
  else if (/\b(fine|small|tiny|misty)\s+(rain|drops?|drizzle)\b/.test(text)) patch['weather.precipitation.drops'] = 'fine';
  if (patch['weather.precipitation.drops']) chip('Drops', patch['weather.precipitation.drops']);

  if (/\b(intermittent|on and off|off and on|patchy|now and then)\b/.test(text)) patch['weather.precipitation.character'] = 'intermittent';
  else if (/\b(showery|squall\w*|in bursts|bursts of)\b/.test(text)) patch['weather.precipitation.character'] = 'showery';
  else if (/\b(steady|constant|continuous|persistent|relentless|all night)\b/.test(text)) patch['weather.precipitation.character'] = 'continuous';
  if (patch['weather.precipitation.character']) chip('Character', patch['weather.precipitation.character']);

  const hailNum = /(\d+(?:[.,]\d+)?)\s*(?:mm|millimet\w+)\s+hail/.exec(text);
  const hailWord = [[/\bpea\b/, 6], [/\bmarble\b/, 13], [/\b(grape|cherry)\b/, 19], [/\bwalnut\b/, 25], [/\bping[\s-]?pong\b/, 38], [/\bgolf[\s-]?ball\b/, 44]].find(([re]) => re.test(text));
  if (hailNum || (hailWord && /\bhail/.test(text))) {
    patch['weather.precipitation.hail_size_mm'] = Math.max(6, Math.min(44, hailNum ? num(hailNum[1]) : hailWord[1]));
    patch['weather.precipitation.type'] = 'hail';
    chip('Hail', `${fmtNum(patch['weather.precipitation.hail_size_mm'])} mm stones`);
  }

  /* wind: beaufort words and numbers */
  let bft = null;
  const forceNum = /\b(?:force|beaufort|bft)\s*(\d{1,2}(?:[.,]\d)?)\b/.exec(text);
  if (forceNum) bft = Math.min(12, num(forceNum[1]));
  if (bft === null) {
    for (const [name, b] of BFT_NAMES) {
      if (new RegExp(`\\b${name}\\b`).test(text)) {
        bft = b;
        break;
      }
    }
  }
  if (/\b(gusty|blustery|gusting)\b/.test(text)) patch['weather.wind.gust_factor'] = 1.75;

  const speedRe = /(\d+(?:[.,]\d+)?)\s*(km\s*\/\s*h|kmh|kph|km per hour|kilomet\w+ (?:per|an) hour|mph|miles (?:per|an) hour|m\s*\/\s*s|mps|met(?:er|re)s (?:per|a) second|knots?|kts?|kt)\b/g;
  let mean = null;
  let gust = null;
  let m;
  while ((m = speedRe.exec(text))) {
    const v = num(m[1]);
    const u = m[2];
    let mps;
    if (/^k(m|ph)|kilomet/.test(u)) mps = v / 3.6;
    else if (/mph|miles/.test(u)) mps = v * 0.44704;
    else if (/kn|kt/.test(u)) mps = v * 0.514444;
    else mps = v;
    const before = text.slice(Math.max(0, m.index - 22), m.index);
    const after = text.slice(m.index + m[0].length, m.index + m[0].length + 12);
    if (/gust/.test(before) || /^\s*gusts?/.test(after)) gust = mps;
    else if (mean === null) mean = mps;
  }
  if (mean !== null) patch['weather.wind.speed_mps'] = mean;
  else if (bft !== null) patch['weather.wind.speed_mps'] = mpsFromBft(bft);
  if (gust !== null) {
    const base = patch['weather.wind.speed_mps'] ?? gust / 1.6;
    if (patch['weather.wind.speed_mps'] === undefined) patch['weather.wind.speed_mps'] = base;
    patch['weather.wind.gust_factor'] = Math.max(1, Math.min(2, gust / Math.max(0.5, base)));
  }

  /* direction: wind vs thunder */
  const fromRe = new RegExp(`\\b(?:from|out of)\\s+(?:the\\s+)?${DIR_WORD}\\b`, 'g');
  let fm;
  while ((fm = fromRe.exec(text))) {
    const k = dirKey(fm[1]);
    if (!k) continue;
    const before = text.slice(Math.max(0, fm.index - 30), fm.index);
    if (/\b(thunder|storm|lightning|rumble)\w*[^.,;]*$/.test(before) && !/\bwind\w*[^.,;]*$/.test(before)) {
      patch['weather.thunder.bearing_deg'] = DIRS[k];
    } else if (patch['weather.wind.from_deg'] === undefined) {
      patch['weather.wind.from_deg'] = DIRS[k];
    }
  }
  const erly = new RegExp(`\\b${DIR_WORD.replace(/\)$/, '')})(?:erly|ern)?\\s+(?:wind|winds|breeze|gale|gusts|airflow)\\b`).exec(text)
    || /\b(north|south|east|west|north[\s-]?east|north[\s-]?west|south[\s-]?east|south[\s-]?west)erly\b/.exec(text);
  if (erly && patch['weather.wind.from_deg'] === undefined) {
    const k = dirKey(erly[1]);
    if (k) patch['weather.wind.from_deg'] = DIRS[k];
  }
  const degFrom = /\bwind\w*\s+(?:from\s+)?(\d{1,3})\s*(?:°|deg\w*)/.exec(text);
  if (degFrom && patch['weather.wind.from_deg'] === undefined) patch['weather.wind.from_deg'] = num(degFrom[1]) % 360;
  const toDir = new RegExp(`\\b(?:storm|thunder|lightning|cell)\\w*\\b[^.,;]{0,30}?\\b(?:to|in|towards|over)\\s+the\\s+${DIR_WORD}\\b`).exec(text);
  if (toDir && patch['weather.thunder.bearing_deg'] === undefined) {
    const k = dirKey(toDir[1]);
    if (k) patch['weather.thunder.bearing_deg'] = DIRS[k];
  }
  if (patch['weather.wind.speed_mps'] !== undefined || patch['weather.wind.from_deg'] !== undefined) {
    const v = patch['weather.wind.speed_mps'] ?? scene?.weather?.wind?.speed_mps ?? 4;
    const b = bftFromMps(v);
    const from = patch['weather.wind.from_deg'] ?? scene?.weather?.wind?.from_deg ?? 270;
    chip('Wind', `force ${fmtNum(b, 1)} ${bftInfo(b).name.toLowerCase()}, ${Math.round(v * 3.6)} km/h from the ${compassWord(from)}`);
  }
  if (patch['weather.wind.gust_factor'] !== undefined) chip('Gusts', `${fmtNum(patch['weather.wind.gust_factor'], 2)} × mean`);

  /* thunder */
  if (/\bno thunder|without thunder|thunder-free\b/.test(text)) {
    patch['weather.thunder.enabled'] = false;
    chip('Thunder', 'off');
  } else if (thunderCtx && /\b(thunder\w*|lightning|electrical storm|tstorm|storm\b)/.test(text)) {
    patch['weather.thunder.enabled'] = true;
    let km = null;
    const kmRe = /(\d+(?:[.,]\d+)?)\s*(km|kilomet\w+|mi|miles?)\b(?!\s*(?:\/|per|an hour|a hour|ph))/.exec(text);
    if (kmRe) km = num(kmRe[1]) * (/^mi/.test(kmRe[2]) ? 1.609 : 1);
    const secRe = /(\d+(?:[.,]\d+)?)\s*(?:s|sec|secs|seconds?)\b/.exec(text);
    if (km === null && secRe && /\b(flash|bang|count|between|after|later|delay)\b/.test(text)) {
      const T = tempNow();
      km = num(secRe[1]) / flashBang(1, T);
    }
    if (km === null) {
      if (/\b(overhead|right above|directly above|on top of (?:us|me)|right here)\b/.test(text)) km = 0.8;
      else if (/\b(very close|very near)\b/.test(text)) km = 1;
      else if (/\b(close|nearby|near)\b/.test(text)) km = 2;
      else if (/\b(distant|far off|far away|faraway|far-off|remote|horizon)\b/.test(text)) km = 15;
      else if (/\brumbl\w*\b/.test(text)) km = 12;
    }
    if (km !== null) patch['weather.thunder.distance_km'] = Math.max(0.3, Math.min(25, km));
    if (/\b(approaching|coming (?:closer|in|our way)|moving in|rolling in|building|arriving|closing in)\b/.test(text)) patch['weather.thunder.approach_kmh'] = 20;
    else if (/\b(receding|moving away|passing|departing|leaving|fading|moving off)\b/.test(text)) patch['weather.thunder.approach_kmh'] = -20;
    if (/\b(frequent|constant|continuous|non-stop) (?:lightning|flashes)\b/.test(text)) patch['weather.thunder.flashes_per_min'] = 8;
    else if (/\b(occasional|rare|odd) (?:lightning|flash\w*|thunder)\b/.test(text)) patch['weather.thunder.flashes_per_min'] = 0.8;
    const fpm = /(\d+(?:[.,]\d+)?)\s*flash\w*\s*(?:per|a|an|\/)\s*min/.exec(text);
    if (fpm) patch['weather.thunder.flashes_per_min'] = Math.min(20, num(fpm[1]));
    const km2 = patch['weather.thunder.distance_km'] ?? scene?.weather?.thunder?.distance_km ?? 8;
    chip('Thunder', `${fmtNum(km2)} km, flash-to-bang ${fmtNum(flashBang(km2, tempNow()), 1)} s`);
  }
  if (patch['weather.thunder.bearing_deg'] !== undefined) chip('Storm bearing', compassWord(patch['weather.thunder.bearing_deg']));

  /* storm arc */
  if (/\b(arriving|approaching|building|first drops|coming)\b/.test(text) && thunderCtx) patch['weather.storm_arc'] = 'arriving';
  else if (/\b(clearing|after the storm|dying down|easing|tailing off|passed)\b/.test(text)) patch['weather.storm_arc'] = 'clearing';
  else if (/\bpassing\b/.test(text)) patch['weather.storm_arc'] = 'passing';
  if (patch['weather.storm_arc']) chip('Storm arc', patch['weather.storm_arc']);

  /* shelter */
  for (const [re, r] of SHELTER_RULES) {
    if (!re.test(text)) continue;
    patch['place.shelter'] = r.shelter;
    if (r.window !== undefined) patch['place.window_open'] = r.window;
    break;
  }
  if (patch['place.shelter']) chip('Shelter', patch['place.shelter'].replace('_', ', window ').replace('room, window open', 'room, window open'));
  if (/\bwipers? (?:on|going|running)|\bslow wipers?\b/.test(text)) patch['place.wipers'] = 'slow';
  if (/\bfast wipers?|wipers? (?:on )?fast\b/.test(text)) patch['place.wipers'] = 'fast';
  if (/\bwipers? off\b/.test(text)) patch['place.wipers'] = 'off';
  if (patch['place.wipers']) chip('Wipers', patch['place.wipers']);

  /* place */
  for (const [re, p] of PLACE_RULES) {
    if (re.test(text)) {
      patch['place.preset'] = p;
      chip('Place', p.replace(/_/g, ' '));
      break;
    }
  }

  /* surfaces */
  const mentioned = [];
  for (const [re, k] of SURFACE_RULES) {
    if (re.test(text) && !mentioned.includes(k)) mentioned.push(k);
  }
  if (mentioned.length) {
    const cur = { ...(scene?.place?.surfaces || {}) };
    for (const k of mentioned) delete cur[k];
    const rest = Object.values(cur).reduce((a, b) => a + b, 0);
    const out = {};
    const share = 75 / mentioned.length;
    mentioned.forEach((k) => { out[k] = share; });
    for (const [k, v] of Object.entries(cur)) if (rest > 0) out[k] = (v / rest) * 25;
    if (rest <= 0) mentioned.forEach((k) => { out[k] = 100 / mentioned.length; });
    for (const k of Object.keys(out)) out[k] = Math.round(out[k] * 10) / 10;
    patch['place.surfaces'] = out;
    chip('Surfaces', mentioned.map((k) => k.replace('_', ' ')).join(', '));
  }

  /* time */
  const clock = /\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b|\b(\d{1,2}):(\d{2})\b/.exec(text);
  if (clock) {
    let h;
    if (clock[3]) {
      h = num(clock[1]) % 12 + (clock[3] === 'pm' ? 12 : 0) + (clock[2] ? num(clock[2]) / 60 : 0);
    } else {
      h = num(clock[4]) + num(clock[5]) / 60;
    }
    if (h >= 0 && h < 24) patch['weather.hour'] = h;
  }
  if (patch['weather.hour'] === undefined) {
    for (const [re, h] of TIME_RULES) {
      if (re.test(text)) {
        patch['weather.hour'] = h;
        break;
      }
    }
  }
  if (patch['weather.hour'] !== undefined) {
    const h = patch['weather.hour'];
    chip('Time', `${String(Math.floor(h)).padStart(2, '0')}:${String(Math.round((h % 1) * 60)).padStart(2, '0')}`);
  }

  for (const [re, s] of SEASON_RULES) {
    if (re.test(text)) {
      patch['weather.season'] = s;
      chip('Season', s);
      break;
    }
  }

  /* temperature */
  const tNum = /(-?\d+(?:[.,]\d+)?)\s*(?:°\s*|deg(?:rees?)?\s*)(c|celsius|f|fahrenheit)?\b/.exec(text);
  if (tNum && !/wind\w*\s+(?:from\s+)?$/.test(text.slice(Math.max(0, tNum.index - 12), tNum.index))) {
    let t = num(tNum[1]);
    if (tNum[2] && tNum[2].startsWith('f')) t = ((t - 32) * 5) / 9;
    if (t >= -40 && t <= 50) patch['weather.temperature_c'] = Math.round(t * 10) / 10;
  } else {
    const tw = [[/\b(scorching|sweltering|hot)\b/, 30], [/\b(warm|balmy|muggy|sultry)\b/, 23], [/\bmild\b/, 15], [/\bcool\b/, 10], [/\bchilly\b/, 6], [/\b(cold|bitter)\b/, 3], [/\b(icy|frosty|freezing(?! rain| drizzle))\b/, -3]]
      .find(([re]) => re.test(text));
    if (tw) patch['weather.temperature_c'] = tw[1];
  }
  if (patch['weather.temperature_c'] !== undefined) {
    const T = patch['weather.temperature_c'];
    chip('Temperature', `${fmtNum(T, 1)} °C (sound ${fmtNum(speedOfSound(T), 0)} m/s)`);
    const ty = patch['weather.precipitation.type'] ?? scene?.weather?.precipitation?.type;
    if (T <= -1 && (ty === 'rain' || ty === 'drizzle' || ty === 'showers') && !/\brain\b/.test(text)) patch['weather.precipitation.type'] = 'snow';
  }

  /* humidity */
  const rh = /(\d{1,3})\s*%\s*(?:humidity|rh|relative humidity)|humidity\s*(?:of|at)?\s*(\d{1,3})\s*%?/.exec(text);
  if (rh) patch['weather.humidity_pct'] = Math.max(10, Math.min(100, num(rh[1] ?? rh[2])));
  else if (/\b(humid|muggy|sultry|steamy|clammy)\b/.test(text)) patch['weather.humidity_pct'] = 96;
  else if (/\bdry air\b/.test(text)) patch['weather.humidity_pct'] = 45;
  if (patch['weather.humidity_pct'] !== undefined) chip('Humidity', `${fmtNum(patch['weather.humidity_pct'], 0)} %`);

  /* listener posture */
  if (/\b(lying|lie down|in bed|in my sleeping bag|on the floor)\b/.test(text)) patch['listener.height_m'] = 0.5;
  else if (/\b(sitting|seated|sat)\b/.test(text)) patch['listener.height_m'] = 1.2;
  if (patch['listener.height_m'] !== undefined) chip('Ears at', `${fmtNum(patch['listener.height_m'], 1)} m`);

  for (const [re, md] of MODE_RULES) {
    if (re.test(text)) {
      patch.mode = md;
      chip('Mode', md);
      break;
    }
  }

  return { patch, chips, understood: chips.length > 0 };
}
