// Drop-size parity: site/engine/physics.js dsdFromRate against raingen/physics/dsd.py dsd_from_scene.
// Reference values come from the Python library at the default cutoffs (0.3-6 mm, gamma, mu = 3):
//   dsd_from_scene(SimpleNamespace(drops=d), R) -> [Dm mm, Nw m^-3 mm^-1, total_flux m^-2 s^-1]
// Run: node tests/web/dsd_parity.mjs
import { dsdFromRate, classFlux, totalFlux } from '../../site/engine/physics.js';

const REF = {
  'fine:0.5': [0.736839, 13118.0412, 922.615503],
  'fine:4': [1.128222, 13145.7258, 2450.379539],
  'fine:25': [1.581752, 16679.0107, 6064.324555],
  'fine:100': [1.798268, 36851.751, 16995.278119],
  'normal:0.5': [0.86574, 5919.782, 616.455993],
  'normal:4': [1.28314, 7124.9784, 1726.760196],
  'normal:25': [1.716605, 11414.2372, 4833.862015],
  'normal:100': [1.987112, 23326.7305, 12878.399355],
  'fat:0.5': [1.235324, 1066.379, 239.446874],
  'fat:4': [1.755406, 1647.2824, 726.899617],
  'fat:25': [2.343072, 2769.5954, 2038.608285],
  'fat:100': [2.717095, 5754.2287, 5432.250327],
};
const TOL = 0.01;

let fail = 0;
const rel = (a, b) => Math.abs(a - b) / Math.abs(b);
for (const [key, [dmRef, nwRef, fluxRef]] of Object.entries(REF)) {
  const [drops, r] = key.split(':');
  const dsd = dsdFromRate(Number(r), { drops });
  const flux = totalFlux(classFlux(dsd));
  const eDm = rel(dsd.dm, dmRef);
  const eNw = rel(dsd.nw, nwRef);
  const eFlux = rel(flux, fluxRef);
  const ok = eDm <= TOL && eFlux <= TOL && eNw <= TOL;
  if (!ok) fail++;
  console.log(
    `${ok ? 'ok  ' : 'FAIL'} ${drops.padEnd(6)} R=${String(r).padEnd(4)} ` +
      `Dm ${dsd.dm.toFixed(4)} (py ${dmRef.toFixed(4)}, ${(eDm * 100).toFixed(3)}%)  ` +
      `flux ${flux.toFixed(1)} (py ${fluxRef.toFixed(1)}, ${(eFlux * 100).toFixed(3)}%)  ` +
      `Nw ${(eNw * 100).toFixed(3)}%`,
  );
}
if (fail) {
  console.error(`${fail} case(s) outside ${TOL * 100}%`);
  process.exit(1);
}
console.log(`all ${Object.keys(REF).length} cases within ${TOL * 100}%`);
