/**
 * Unlock audio inside a user gesture.
 *
 * Safari and every iOS browser only let an AudioContext start (and only route it past the
 * ring/silent switch) when it is created or resumed synchronously inside the click/tap handler.
 * Call unlockAudio() as the first statement of the handler, before any await, and hand the
 * returned context to WeatherEngine({ context }).
 */

let silentEl = null;

function silentWavUrl() {
  // 0.25 s of 8 kHz 8-bit mono silence (unsigned 8-bit PCM: 128 is zero).
  const n = 2000;
  const buf = new ArrayBuffer(44 + n);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 36 + n, true); str(8, 'WAVE');
  str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, 8000, true); v.setUint32(28, 8000, true); v.setUint16(32, 1, true); v.setUint16(34, 8, true);
  str(36, 'data'); v.setUint32(40, n, true);
  new Uint8Array(buf, 44).fill(128);
  return URL.createObjectURL(new Blob([buf], { type: 'audio/wav' }));
}

/** Create (or resume) an AudioContext inside the current gesture; returns it, or null without Web Audio. */
export function unlockAudio(existing = null) {
  // iOS 17+: play through the ring/silent switch like a media app.
  try {
    if (navigator.audioSession) navigator.audioSession.type = 'playback';
  } catch (e) { /* not supported */ }
  // Older iOS: a playing <audio> element moves the page into the playback audio session.
  try {
    if (!silentEl) {
      silentEl = new Audio(silentWavUrl());
      silentEl.loop = true;
      silentEl.setAttribute('playsinline', '');
    }
    const p = silentEl.play();
    if (p && p.catch) p.catch(() => {});
  } catch (e) { /* media playback unavailable */ }

  const AC = globalThis.AudioContext || globalThis.webkitAudioContext;
  if (!AC) return null;
  let ctx = existing && existing.state !== 'closed' ? existing : null;
  if (!ctx) {
    try {
      ctx = new AC({ latencyHint: 'playback' });
    } catch (e) {
      ctx = new AC();
    }
  }
  // Starting a one-sample buffer inside the gesture unlocks older WebKit.
  try {
    const src = ctx.createBufferSource();
    src.buffer = ctx.createBuffer(1, 1, ctx.sampleRate);
    src.connect(ctx.destination);
    src.start(0);
  } catch (e) { /* ignore */ }
  try {
    const r = ctx.resume();
    if (r && r.catch) r.catch(() => {});
  } catch (e) { /* ignore */ }
  return ctx;
}

/** Stop the silent media element (call when the user pauses, so the OS media controls clear). */
export function releaseAudio() {
  try { silentEl && silentEl.pause(); } catch (e) { /* ignore */ }
}
