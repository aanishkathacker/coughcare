"""Small HTML components used by the Streamlit app."""

from __future__ import annotations

import base64
import json


def waveform_player(wav_bytes: bytes, peaks: list[float], duration_s: float) -> str:
    """Return a self-contained scrolling-waveform audio player.

    Uses only the embedded WAV and vanilla canvas/JS, so it works without a
    CDN or network access. The waveform scrolls under a fixed playhead while
    the native audio element plays.
    """
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    peaks_json = json.dumps(peaks)
    duration = max(duration_s, 0.01)
    return f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  :root {{
    --ink: #16231F;
    --muted: #5E6B64;
    --accent: #0E6F5C;
    --idle: #CFC7B7;
    --line: #E3DBCB;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: transparent; }}
  body {{ font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); }}
  .player {{ width: 100%; padding: 2px 0 0; }}
  .canvas-wrap {{ position: relative; width: 100%; height: 92px; }}
  canvas {{ width: 100%; height: 92px; display: block; }}
  .meta {{ display: flex; justify-content: space-between; font-size: 12px; letter-spacing: .02em; color: var(--muted); margin: 2px 2px 8px; font-variant-numeric: tabular-nums; }}
  audio {{ width: 100%; height: 38px; display: block; }}
  @media (prefers-reduced-motion: reduce) {{ canvas {{ scroll-behavior: auto; }} }}
</style>
</head>
<body>
  <div class="player">
    <div class="canvas-wrap"><canvas id="wave" aria-label="Cough waveform"></canvas></div>
    <div class="meta"><span id="elapsed">0:00</span><span id="total">{duration_s:.1f}s clip</span></div>
    <audio id="audio" controls preload="metadata" src="data:audio/wav;base64,{audio_b64}"></audio>
  </div>
<script>
(function () {{
  const audio = document.getElementById('audio');
  const canvas = document.getElementById('wave');
  const elapsed = document.getElementById('elapsed');
  const ctx = canvas.getContext('2d');
  const peaks = {peaks_json};
  const DURATION = {duration};
  const VISIBLE = 46;          // bars in the scrolling viewport
  const PLAYHEAD = 0.38;       // fixed playhead position (0–1)
  let width = 0, height = 0, dpr = 1;

  function resize() {{
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    width = Math.max(rect.width, 240);
    height = 92;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }}

  function draw() {{
    if (!width) return;
    const progress = DURATION > 0 ? Math.min(audio.currentTime / DURATION, 1) : 0;
    ctx.clearRect(0, 0, width, height);

    // Subtle centre line
    ctx.strokeStyle = '#E3DBCB';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2 + 0.5);
    ctx.lineTo(width, height / 2 + 0.5);
    ctx.stroke();

    const gap = 3;
    const barW = Math.max(2, (width - gap * (VISIBLE - 1)) / VISIBLE);
    const step = peaks.length / VISIBLE;
    const centreIndex = progress * peaks.length;
    const startIndex = centreIndex - VISIBLE * PLAYHEAD;
    const playheadX = width * PLAYHEAD;

    for (let i = 0; i < VISIBLE; i++) {{
      const srcIndex = startIndex + i;
      if (srcIndex < 0 || srcIndex >= peaks.length) continue;
      const idx = Math.max(0, Math.min(peaks.length - 1, Math.round(srcIndex)));
      const amp = Math.max(0.045, peaks[idx]);
      const barH = Math.max(3, amp * (height - 18));
      const x = i * (barW + gap);
      const played = x <= playheadX;
      ctx.fillStyle = played ? '#0E6F5C' : '#CFC7B7';
      const y = (height - barH) / 2;
      const radius = Math.min(barW / 2, 3);
      ctx.beginPath();
      if (ctx.roundRect) {{
        ctx.roundRect(x, y, barW, barH, radius);
      }} else {{
        ctx.rect(x, y, barW, barH);
      }}
      ctx.fill();
    }}

    // Fixed playhead
    ctx.strokeStyle = '#16231F';
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(playheadX, 6);
    ctx.lineTo(playheadX, height - 6);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }}

  let raf = null;
  function tick() {{
    draw();
    const t = Math.floor(audio.currentTime || 0);
    elapsed.textContent = '0:' + String(t).padStart(2, '0');
    if (!audio.paused && !audio.ended) {{
      raf = requestAnimationFrame(tick);
    }}
  }}

  audio.addEventListener('play', () => {{
    if (raf) cancelAnimationFrame(raf);
    tick();
  }});
  audio.addEventListener('pause', () => {{
    if (raf) cancelAnimationFrame(raf);
    draw();
  }});
  audio.addEventListener('timeupdate', draw);
  audio.addEventListener('ended', () => {{
    if (raf) cancelAnimationFrame(raf);
    draw();
  }});
  audio.addEventListener('error', () => {{
    elapsed.textContent = 'playback error';
  }});

  window.addEventListener('resize', resize);
  resize();
  if (window.ResizeObserver) {{
    new ResizeObserver(resize).observe(canvas);
  }}
}})();
</script>
</body>
</html>
"""
