/* Shape Brain Trainer: all the AI code.
   Drawing pad -> 28x28 pixel grid -> k-nearest-neighbour guess -> imagination.
   Plain JavaScript, no libraries, nothing leaves the browser. */

(function(){
  'use strict';

  /* ============================================================
     1. Constants
     ============================================================ */
  const CANVAS_SIZE   = 450;   // drawing pad, px
  const GRID          = 28;    // downsampled grid is 28 x 28
  const VEC_LEN       = GRID * GRID;  // 784 features
  const INK_ALPHA     = 40;    // canvas alpha above this counts as ink
  const CELL_FILL_MIN = 0.10;  // share of a block that must be inked to become a 1
  const BRUSH         = 15;    // stroke width, px
  const BOX_PAD       = 1.14;  // breathing room around the bounding box
  const K_MAX         = 5;     // k in k-nearest neighbours
  const SPEAK_AT      = 70;    // announce out loud above this confidence
  const PIX_VIEW      = 280;   // pixel-view canvas size, px
  const PIX_CELL      = PIX_VIEW / GRID;  // 10 px per grid cell
  const DREAM_VIEW    = 336;   // imagination canvas size, px
  const DREAM_CELL    = DREAM_VIEW / GRID;  // 12 px per grid cell
  const DREAM_MS      = 1300;  // how long a shape takes to crystallise out of static
  const DREAM_FRAME   = 45;    // ms per animation frame (a TV-static flicker rate)
  const DREAM_KEEP    = 8;     // thumbnails kept in "Recent dreams"
  const EMPTY_CELL    = 'rgba(30,43,69,0.55)';
  const STATIC_CELL   = 'rgba(138,154,181,0.5)';
  const STATIC_TONE   = '#8a9ab5';

  const SHAPES = ['Circle', 'Triangle', 'Square', 'Star'];
  const META = {
    Circle:   { glyph: '◯', tone: '#22d3ee', key: '1' },
    Triangle: { glyph: '△', tone: '#fb7185', key: '2' },
    Square:   { glyph: '□', tone: '#a78bfa', key: '3' },
    Star:     { glyph: '☆', tone: '#fbbf24', key: '4' }
  };
  const LEVELS = [
    { min: 0,  name: 'Blank Slate' },
    { min: 4,  name: 'First Sparks' },
    { min: 8,  name: 'Pattern Spotter' },
    { min: 16, name: 'Shape Scout' },
    { min: 28, name: 'Sharp Eye' },
    { min: 40, name: 'Shape Genius' }
  ];

  /* ============================================================
     2. State
     ============================================================ */
  const trainingData = [];                  // [{ id, label, vector: [784 x 0|1] }]
  const counts = { Circle: 0, Triangle: 0, Square: 0, Star: 0 };
  const strokes = [];                       // [[{x,y}, ...], ...]

  let sampleId       = 0;
  let drawing        = false;
  let currentStroke  = null;
  let lastVector     = null;   // most recent 784-vector, for the hover read-out
  let lastSpoken     = null;   // label last announced out loud
  let voiceOn        = true;
  let previewTimer   = 0;
  let resetArmed     = false;
  let resetTimer     = 0;
  let dreamRun       = 0;      // bumping this cancels any dream still animating
  let dreamTimer     = 0;

  /* ============================================================
     3. Element handles
     ============================================================ */
  const pad        = document.getElementById('pad');
  const ctx        = pad.getContext('2d', { willReadFrequently: true });
  const padFrame   = document.getElementById('padFrame');
  const padHint    = document.getElementById('padHint');
  const pixels     = document.getElementById('pixels');
  const pctx       = pixels.getContext('2d');
  const barsEl     = document.getElementById('bars');
  const trainBtnsEl= document.getElementById('trainBtns');
  const logEl      = document.getElementById('log');
  const dataCountEl= document.getElementById('dataCount');
  const levelNameEl= document.getElementById('levelName');
  const levelNumEl = document.getElementById('levelNum');
  const xpFillEl   = document.getElementById('xpFill');
  const xpBarEl    = document.getElementById('xpBar');
  const xpHintEl   = document.getElementById('xpHint');
  const verdictEl  = document.getElementById('verdict');
  const verdictText= document.getElementById('verdictText');
  const verdictSub = document.getElementById('verdictSub');
  const statOn     = document.getElementById('statOn');
  const statBox    = document.getElementById('statBox');
  const statStep   = document.getElementById('statStep');
  const statCursor = document.getElementById('statCursor');
  const kShownEl   = document.getElementById('kShown');
  const liveDot    = document.getElementById('liveDot');
  const soundBtn   = document.getElementById('soundBtn');
  const resetBtn   = document.getElementById('resetBtn');
  const undoBtn    = document.getElementById('undoBtn');
  const clearBtn   = document.getElementById('clearBtn');
  const autoClear  = document.getElementById('autoClear');
  const logClear   = document.getElementById('logClear');
  const dreamCv    = document.getElementById('dream');
  const dctx       = dreamCv.getContext('2d');
  const dreamCard  = document.getElementById('dreamCard');
  const dreamTitle = document.getElementById('dreamTitle');
  const dreamSub   = document.getElementById('dreamSub');
  const dreamBtnsEl= document.getElementById('dreamBtns');
  const dreamSurprise = document.getElementById('dreamSurprise');
  const dreamFromEl  = document.getElementById('dreamFrom');
  const dreamNoiseEl = document.getElementById('dreamNoise');
  const dreamInkEl   = document.getElementById('dreamInk');
  const dreamCheckEl = document.getElementById('dreamCheck');
  const dreamGallery = document.getElementById('dreamGallery');
  const dreamEmpty   = document.getElementById('dreamEmpty');

  const barFills = {};   // label -> fill element
  const barPcts  = {};   // label -> percentage text element
  const barRows  = {};   // label -> row element
  const tallyEls = {};   // label -> tally element on the train button
  const dreamBtnEls   = {};   // label -> dream button
  const dreamNoiseEls = {};   // label -> "% noise" tag on the dream button

  /* ============================================================
     4. Brain log
     ============================================================ */
  function clock() {
    const d = new Date();
    const p = function (n) { return n < 10 ? '0' + n : String(n); };
    return p(d.getHours()) + ':' + p(d.getMinutes()) + ':' + p(d.getSeconds());
  }

  function log(msg, tone) {
    const line = document.createElement('div');
    line.className = 'log-line';
    line.style.color =
      tone === 'ok'   ? 'var(--emerald)' :
      tone === 'warn' ? 'var(--amber)'   :
      tone === 'math' ? 'var(--cyan)'    :
      tone === 'bad'  ? 'var(--rose)'    : 'var(--muted)';
    const stamp = document.createElement('span');
    stamp.className = 'stamp';
    stamp.textContent = '[' + clock() + '] ';
    line.appendChild(stamp);
    line.appendChild(document.createTextNode(msg));
    logEl.appendChild(line);
    while (logEl.children.length > 90) { logEl.removeChild(logEl.firstChild); }
    logEl.scrollTop = logEl.scrollHeight;
  }

  /* ============================================================
     5. Speech
     ============================================================ */
  const canSpeak = typeof window.speechSynthesis !== 'undefined' &&
                   typeof window.SpeechSynthesisUtterance !== 'undefined';

  function speak(text) {
    if (!voiceOn || !canSpeak) { return; }
    try {
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 0.98;
      u.pitch = 1.15;
      u.volume = 1;
      window.speechSynthesis.speak(u);
    } catch (err) { /* voice is a bonus, never a blocker */ }
  }

  function toggleVoice() {
    voiceOn = !voiceOn;
    soundBtn.textContent = 'Voice: ' + (voiceOn ? 'on' : 'off');
    soundBtn.setAttribute('aria-pressed', voiceOn ? 'true' : 'false');
    if (!voiceOn && canSpeak) {
      try { window.speechSynthesis.cancel(); } catch (err) {}
    }
    log('Voice ' + (voiceOn ? 'on' : 'off') + '.', 'info');
  }

  /* ============================================================
     6. Drawing the pad
     ============================================================ */
  function pointFrom(e) {
    const r = pad.getBoundingClientRect();
    const sx = CANVAS_SIZE / r.width;
    const sy = CANVAS_SIZE / r.height;
    return {
      x: Math.max(0, Math.min(CANVAS_SIZE, (e.clientX - r.left) * sx)),
      y: Math.max(0, Math.min(CANVAS_SIZE, (e.clientY - r.top) * sy))
    };
  }

  function drawStroke(pts) {
    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = BRUSH;
    ctx.strokeStyle = '#eaf6ff';
    ctx.fillStyle = '#eaf6ff';

    if (pts.length === 1) {
      // a single tap is a dot
      ctx.beginPath();
      ctx.arc(pts[0].x, pts[0].y, BRUSH / 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      return;
    }

    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    if (pts.length === 2) {
      ctx.lineTo(pts[1].x, pts[1].y);
    } else {
      // anti-aliased smoothing: curve through the midpoint of each pair of samples,
      // using the sample itself as the quadratic control point
      for (let i = 1; i < pts.length - 1; i++) {
        const mx = (pts[i].x + pts[i + 1].x) / 2;
        const my = (pts[i].y + pts[i + 1].y) / 2;
        ctx.quadraticCurveTo(pts[i].x, pts[i].y, mx, my);
      }
      const last = pts[pts.length - 1];
      ctx.lineTo(last.x, last.y);
    }
    ctx.stroke();
    ctx.restore();
  }

  function redrawPad() {
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    for (let i = 0; i < strokes.length; i++) { drawStroke(strokes[i]); }
  }

  function setPadHint() {
    if (strokes.length === 0) { padHint.classList.remove('hidden'); }
    else { padHint.classList.add('hidden'); }
  }

  /* ============================================================
     7. Pixel maths: 450 x 450 canvas  ->  bounding-boxed 784-vector
     ============================================================ */
  function extractFeatures() {
    const img = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE).data;

    // 7a. find the ink bounding box
    let minX = CANVAS_SIZE, minY = CANVAS_SIZE, maxX = -1, maxY = -1, inkPixels = 0;
    for (let y = 0; y < CANVAS_SIZE; y++) {
      const row = y * CANVAS_SIZE;
      for (let x = 0; x < CANVAS_SIZE; x++) {
        if (img[(row + x) * 4 + 3] > INK_ALPHA) {
          inkPixels++;
          if (x < minX) { minX = x; }
          if (x > maxX) { maxX = x; }
          if (y < minY) { minY = y; }
          if (y > maxY) { maxY = y; }
        }
      }
    }
    if (maxX < 0) { return null; }   // nothing drawn

    // 7b. square the box around the drawing's centre, so a small shape in the corner
    //     produces the same vector as a big one in the middle
    const w = maxX - minX + 1;
    const h = maxY - minY + 1;
    let side = Math.max(w, h) * BOX_PAD;
    if (side < GRID) { side = GRID; }           // keep at least 1 source px per block
    const cx = (minX + maxX + 1) / 2;
    const cy = (minY + maxY + 1) / 2;
    const x0 = cx - side / 2;
    const y0 = cy - side / 2;
    const step = side / GRID;

    // 7c. average each block and threshold it into 1 or 0
    const vec = new Array(VEC_LEN).fill(0);
    let onCells = 0;
    for (let gy = 0; gy < GRID; gy++) {
      for (let gx = 0; gx < GRID; gx++) {
        const sx0 = Math.floor(x0 + gx * step);
        const sy0 = Math.floor(y0 + gy * step);
        let sx1 = Math.ceil(x0 + (gx + 1) * step);
        let sy1 = Math.ceil(y0 + (gy + 1) * step);
        if (sx1 <= sx0) { sx1 = sx0 + 1; }
        if (sy1 <= sy0) { sy1 = sy0 + 1; }

        let hit = 0;
        let total = 0;
        for (let py = sy0; py < sy1; py++) {
          if (py < 0 || py >= CANVAS_SIZE) { total += (sx1 - sx0); continue; }
          const row = py * CANVAS_SIZE;
          for (let px = sx0; px < sx1; px++) {
            total++;
            if (px < 0 || px >= CANVAS_SIZE) { continue; }
            if (img[(row + px) * 4 + 3] > INK_ALPHA) { hit++; }
          }
        }
        if (total > 0 && hit / total >= CELL_FILL_MIN) {
          vec[gy * GRID + gx] = 1;
          onCells++;
        }
      }
    }

    return {
      vec: vec,
      onCells: onCells,
      inkPixels: inkPixels,
      box: { x: minX, y: minY, w: w, h: h },
      step: step
    };
  }

  /* ============================================================
     8. Pixel view
     ============================================================ */
  function renderPixels(vec) {
    lastVector = vec;
    pctx.clearRect(0, 0, PIX_VIEW, PIX_VIEW);
    for (let gy = 0; gy < GRID; gy++) {
      for (let gx = 0; gx < GRID; gx++) {
        const on = vec ? vec[gy * GRID + gx] === 1 : false;
        pctx.fillStyle = on ? '#22d3ee' : 'rgba(30,43,69,0.55)';
        pctx.fillRect(gx * PIX_CELL + 0.5, gy * PIX_CELL + 0.5, PIX_CELL - 1, PIX_CELL - 1);
      }
    }
  }

  function updatePixelStats(f) {
    if (!f) {
      statOn.textContent = '0 / 784';
      statBox.textContent = '—';
      statStep.textContent = '—';
      return;
    }
    statOn.textContent = f.onCells + ' / 784';
    statBox.textContent = f.box.w + ' × ' + f.box.h + ' px at (' + f.box.x + ',' + f.box.y + ')';
    statStep.textContent = f.step.toFixed(1) + ' px per cell';
  }

  /* ============================================================
     9. k-nearest neighbours over the 784-vectors
     ============================================================ */
  function euclidean(a, b) {
    let sum = 0;
    for (let i = 0; i < VEC_LEN; i++) {
      const d = a[i] - b[i];
      sum += d * d;
    }
    return Math.sqrt(sum);
  }

  function knownClasses() {
    let n = 0;
    for (let i = 0; i < SHAPES.length; i++) {
      if (counts[SHAPES[i]] > 0) { n++; }
    }
    return n;
  }

  function setBars(pctByLabel, leader) {
    for (let i = 0; i < SHAPES.length; i++) {
      const name = SHAPES[i];
      const pct = pctByLabel ? pctByLabel[name] : 0;
      barFills[name].style.width = pct.toFixed(1) + '%';
      barPcts[name].textContent = Math.round(pct) + '%';
      barRows[name].setAttribute('aria-valuenow', String(Math.round(pct)));
      if (leader === name && pct > 0) { barRows[name].classList.add('lead'); }
      else { barRows[name].classList.remove('lead'); }
    }
  }

  function setVerdict(headline, sub, mood) {
    verdictText.textContent = headline;
    verdictSub.textContent = sub;
    verdictEl.classList.remove('sure', 'unsure', 'empty');
    if (mood) { verdictEl.classList.add(mood); }
  }

  const ZERO = { Circle: 0, Triangle: 0, Square: 0, Star: 0 };

  // k-nearest-neighbour vote for any 784-vector against the memory bank.
  // Only call this when trainingData has at least one memory.
  function knnScore(vec) {
    // distance from this vector to every stored memory
    const scored = [];
    for (let i = 0; i < trainingData.length; i++) {
      scored.push({
        label: trainingData[i].label,
        id: trainingData[i].id,
        d: euclidean(vec, trainingData[i].vector)
      });
    }
    scored.sort(function (a, b) { return a.d - b.d; });

    // let the k closest memories vote, each weighted by 1 / (1 + distance)
    const k = Math.min(K_MAX, scored.length);
    const neighbours = scored.slice(0, k);
    const weight = { Circle: 0, Triangle: 0, Square: 0, Star: 0 };
    let totalWeight = 0;
    for (let i = 0; i < neighbours.length; i++) {
      const w = 1 / (1 + neighbours[i].d);
      weight[neighbours[i].label] += w;
      totalWeight += w;
    }

    const pctByLabel = {};
    let leader = SHAPES[0];
    let leaderPct = -1;
    for (let i = 0; i < SHAPES.length; i++) {
      const name = SHAPES[i];
      const pct = totalWeight > 0 ? (weight[name] / totalWeight) * 100 : 0;
      pctByLabel[name] = pct;
      if (pct > leaderPct) { leaderPct = pct; leader = name; }
    }
    return { k: k, neighbours: neighbours, pctByLabel: pctByLabel, leader: leader, leaderPct: leaderPct };
  }

  function classify(quiet) {
    const f = extractFeatures();
    renderPixels(f ? f.vec : null);
    updatePixelStats(f);

    if (!f) {
      setBars(ZERO, null);
      lastSpoken = null;
      setVerdict('Pad is empty.', 'Draw a shape and I will start comparing pixels.', null);
      return;
    }

    if (trainingData.length === 0) {
      setBars(ZERO, null);
      setVerdict("I haven't learned anything yet!",
                 'I have 0 memories, so I have nothing to compare your ' + f.onCells + ' ink pixels against. Teach me a shape.',
                 'empty');
      if (!quiet) {
        log('Comparing 784 pixels to 0 stored memories... nothing to compare with. Confidence: 0%.', 'warn');
      }
      return;
    }

    // 9a + 9b. distance to every memory, then the k closest vote
    const s = knnScore(f.vec);
    const k = s.k;
    const neighbours = s.neighbours;
    const pctByLabel = s.pctByLabel;
    const leader = s.leader;
    const leaderPct = s.leaderPct;

    setBars(pctByLabel, leader);
    kShownEl.textContent = String(k);

    // 9c. report the maths
    if (!quiet) {
      log('Comparing 784 pixels to ' + trainingData.length + ' stored memories... Closest match: ' +
          neighbours[0].label + ' (Distance: ' + neighbours[0].d.toFixed(1) + ')', 'math');
      const votes = [];
      for (let i = 0; i < neighbours.length; i++) {
        votes.push(neighbours[i].label + ' ' + neighbours[i].d.toFixed(1));
      }
      log('k=' + k + ' nearest neighbours voted -> ' + votes.join(' | '), 'info');
    }

    // 9d. verdict
    const rounded = Math.round(leaderPct);
    if (leaderPct >= SPEAK_AT) {
      setVerdict('I think that is a ' + leader.toLowerCase() + '!',
                 rounded + '% confident, from ' + counts[leader] + ' remembered ' +
                 (counts[leader] === 1 ? 'drawing' : 'drawings') + ' of a ' + leader.toLowerCase() + '.',
                 'sure');
      const key = leader + ':sure';
      if (key !== lastSpoken) {
        lastSpoken = key;
        speak('I think that is a ' + leader.toLowerCase() + '!');
      }
    } else {
      setVerdict('Maybe a ' + leader.toLowerCase() + '? I am not sure.',
                 'Only ' + rounded + '% confident. Train me on a few more shapes and ask again.',
                 'unsure');
      const key = leader + ':unsure';
      if (key !== lastSpoken) {
        lastSpoken = key;
        speak("I think it's a " + leader.toLowerCase() + ", but I'm not sure.");
      }
    }

    if (!quiet && knownClasses() === 1) {
      log('Careful: ' + leader + ' is the only shape I know, so everything looks like a ' +
          leader.toLowerCase() + ' to me.', 'warn');
    }
  }

  function schedulePreview() {
    if (previewTimer) { return; }
    previewTimer = window.setTimeout(function () {
      previewTimer = 0;
      const f = extractFeatures();
      renderPixels(f ? f.vec : null);
      updatePixelStats(f);
    }, 90);
  }

  /* ============================================================
     10. Training
     ============================================================ */
  function levelFor(n) {
    let idx = 0;
    for (let i = 0; i < LEVELS.length; i++) {
      if (n >= LEVELS[i].min) { idx = i; }
    }
    return idx;
  }

  function updateStats() {
    const n = trainingData.length;
    dataCountEl.textContent = String(n);
    for (let i = 0; i < SHAPES.length; i++) {
      const name = SHAPES[i];
      tallyEls[name].textContent = counts[name] > 0 ? '× ' + counts[name] : '—';
      if (dreamNoiseEls[name]) {
        dreamNoiseEls[name].textContent = Math.round(noiseLevelFor(counts[name]) * 100) + '% noise';
      }
    }
    kShownEl.textContent = String(Math.min(K_MAX, n));

    const idx = levelFor(n);
    levelNameEl.textContent = LEVELS[idx].name;
    levelNumEl.textContent = String(idx);

    let pct;
    if (idx >= LEVELS.length - 1) {
      pct = 100;
      xpHintEl.textContent = 'Maxed out with ' + n + ' memories. Press Reset brain to start a fresh demo.';
    } else {
      const from = LEVELS[idx].min;
      const to = LEVELS[idx + 1].min;
      pct = ((n - from) / (to - from)) * 100;
      const need = to - n;
      xpHintEl.textContent = 'Feed me ' + need + ' more ' + (need === 1 ? 'drawing' : 'drawings') +
                             ' to reach level ' + (idx + 1) + ': ' + LEVELS[idx + 1].name + '.';
    }
    xpFillEl.style.width = Math.max(0, Math.min(100, pct)).toFixed(1) + '%';
    xpBarEl.setAttribute('aria-valuenow', String(Math.round(pct)));

    if (n > 0) { liveDot.classList.add('live'); }
    else { liveDot.classList.remove('live'); }
  }

  function trainAs(label) {
    const f = extractFeatures();
    if (!f) {
      log('Training refused: the pad is empty. Draw a ' + label.toLowerCase() + ' first.', 'bad');
      setVerdict('Nothing to learn from.', 'Draw a ' + label.toLowerCase() + ' on the pad, then press the button again.', 'empty');
      speak('Draw something first!');
      return;
    }

    sampleId++;
    trainingData.push({ id: sampleId, label: label, vector: f.vec });
    counts[label]++;

    log('Stored memory #' + sampleId + ' as ' + label + ': 784 numbers, ' + f.onCells +
        ' of them set to 1. Memory bank: ' + trainingData.length + '.', 'ok');
    updateStats();

    const btn = document.getElementById('train-' + label);
    if (btn) {
      btn.classList.remove('just-learned');
      void btn.offsetWidth;          // restart the animation
      btn.classList.add('just-learned');
    }

    speak('Got it. That is a ' + label.toLowerCase() + '.');
    lastSpoken = label + ':sure';    // suppress an immediate echo if we re-classify below

    if (autoClear.checked) {
      strokes.length = 0;
      redrawPad();
      setPadHint();
      renderPixels(null);
      updatePixelStats(null);
      setBars(ZERO, null);
      lastSpoken = null;
      setVerdict('Ready for the next drawing.',
                 'Memory bank holds ' + trainingData.length + ' ' +
                 (trainingData.length === 1 ? 'drawing' : 'drawings') + '. Draw another shape.',
                 null);
    } else {
      classify(true);
    }
  }

  /* ============================================================
     11. Pad controls
     ============================================================ */
  function clearPad() {
    strokes.length = 0;
    redrawPad();
    setPadHint();
    lastSpoken = null;
    classify(true);
    log('Pad cleared. Memories kept: ' + trainingData.length + '.', 'info');
  }

  function undo() {
    if (strokes.length === 0) { return; }
    strokes.pop();
    redrawPad();
    setPadHint();
    classify();
  }

  function resetBrain() {
    if (!resetArmed) {
      resetArmed = true;
      resetBtn.classList.add('btn-armed');
      resetBtn.textContent = 'Tap again to wipe!';
      log('Reset armed. Press again within 4 seconds to erase every memory.', 'warn');
      resetTimer = window.setTimeout(disarmReset, 4000);
      return;
    }

    disarmReset();
    const had = trainingData.length;
    trainingData.length = 0;
    for (let i = 0; i < SHAPES.length; i++) { counts[SHAPES[i]] = 0; }
    sampleId = 0;
    lastSpoken = null;
    strokes.length = 0;
    redrawPad();
    setPadHint();
    renderPixels(null);
    updatePixelStats(null);
    setBars(ZERO, null);
    updateStats();
    setVerdict("I haven't learned anything yet!",
               'Every memory is gone. I am exactly as clever as an empty list.',
               'empty');
    resetDreams();
    log('BRAIN WIPED. ' + had + ' ' + (had === 1 ? 'memory' : 'memories') +
        ' deleted. Back to knowing absolutely nothing.', 'bad');
    speak('My brain is empty. I do not know any shapes any more.');
  }

  function disarmReset() {
    if (resetTimer) { window.clearTimeout(resetTimer); resetTimer = 0; }
    resetArmed = false;
    resetBtn.classList.remove('btn-armed');
    resetBtn.textContent = 'Reset brain';
  }

  /* ============================================================
     12. Imagination: invent a brand-new shape from memories
     ============================================================ */
  // How much randomness to stir in. 0 memories -> 1.0 (pure static);
  // every extra memory calms it down: 1 -> 40%, 3 -> 18%, 10 -> 6%.
  function noiseLevelFor(memories) {
    return 1 / (1 + 1.5 * memories);
  }

  function dreamPlan(label) {
    const mems = [];
    for (let i = 0; i < trainingData.length; i++) {
      if (trainingData[i].label === label) { mems.push(trainingData[i].vector); }
    }
    const noise = noiseLevelFor(mems.length);
    const prob = new Array(VEC_LEN).fill(0.5);   // with no memories, every pixel is a coin flip
    if (mems.length === 0) {
      return { prob: prob, noise: noise, used: 0 };
    }

    // 12a. random blend weights, so each dream is a new mix of the memories
    //      (-ln(U) weights, normalised, give an evenly random mix)
    const w = [];
    let wTotal = 0;
    for (let j = 0; j < mems.length; j++) {
      const x = -Math.log(1 - Math.random());
      w.push(x);
      wTotal += x;
    }

    for (let i = 0; i < VEC_LEN; i++) {
      // 12b. blended memory: how likely is ink at this pixel?
      let p = 0;
      for (let j = 0; j < mems.length; j++) { p += w[j] * mems[j][i]; }
      p /= wTotal;
      // 12c. sharpen, so pixels most memories agree on stay strong
      const a = p * p;
      const b = (1 - p) * (1 - p);
      p = a / (a + b);
      // 12d. stir in randomness: fewer memories, more static
      prob[i] = (1 - noise) * p + noise * 0.5;
    }
    return { prob: prob, noise: noise, used: mems.length };
  }

  // Paint one animation frame. Pixels whose random reveal order is below
  // `reveal` have locked in to their final value; the rest still flicker.
  function renderDreamFrame(reveal, final, order, tone) {
    dctx.clearRect(0, 0, DREAM_VIEW, DREAM_VIEW);
    for (let gy = 0; gy < GRID; gy++) {
      for (let gx = 0; gx < GRID; gx++) {
        const i = gy * GRID + gx;
        let fill;
        if (final && order[i] < reveal) {
          fill = final[i] === 1 ? tone : EMPTY_CELL;
        } else {
          fill = Math.random() < 0.5 ? STATIC_CELL : EMPTY_CELL;
        }
        dctx.fillStyle = fill;
        dctx.fillRect(gx * DREAM_CELL + 0.5, gy * DREAM_CELL + 0.5, DREAM_CELL - 1, DREAM_CELL - 1);
      }
    }
  }

  function prefersReducedMotion() {
    return !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  }

  function setDreamCard(title, sub, mood) {
    dreamTitle.textContent = title;
    dreamSub.textContent = sub;
    dreamCard.classList.remove('sure', 'unsure', 'empty');
    if (mood) { dreamCard.classList.add(mood); }
  }

  function markActiveDream(label) {
    for (let i = 0; i < SHAPES.length; i++) {
      const b = dreamBtnEls[SHAPES[i]];
      if (!b) { continue; }
      if (SHAPES[i] === label) { b.classList.add('active'); }
      else { b.classList.remove('active'); }
    }
  }

  function dream(label) {
    const run = ++dreamRun;
    if (dreamTimer) { window.clearTimeout(dreamTimer); dreamTimer = 0; }

    // 12e. build the probability map, then flip a weighted coin per pixel
    const plan = dreamPlan(label);
    const final = new Array(VEC_LEN);
    const order = new Array(VEC_LEN);
    let ink = 0;
    for (let i = 0; i < VEC_LEN; i++) {
      final[i] = Math.random() < plan.prob[i] ? 1 : 0;
      ink += final[i];
      order[i] = Math.random();
    }
    const isStatic = plan.used === 0;
    const tone = isStatic ? STATIC_TONE : META[label].tone;
    const lower = label.toLowerCase();

    markActiveDream(label);
    setDreamCard(isStatic ? 'Trying to imagine a ' + lower + '...' : 'Dreaming up a ' + lower + '...',
                 isStatic ? 'I have no ' + lower + ' memories to pull the pixels into shape, so they stay random.'
                          : 'Starting from pure static and letting my memories pull the pixels into place.', null);
    dreamFromEl.textContent = plan.used + ' ' + lower + (plan.used === 1 ? '' : 's');
    dreamNoiseEl.textContent = Math.round(plan.noise * 100) + '%';
    dreamInkEl.textContent = '...';
    dreamCheckEl.textContent = '...';
    dreamCheckEl.style.color = '';

    if (isStatic) {
      log('Imagining a ' + label + ': 0 ' + lower + ' memories, so every one of the 784 pixels is a 50/50 coin flip...', 'warn');
    } else {
      log('Imagining a ' + label + ': blending a random mix of ' + plan.used + ' ' + lower +
          (plan.used === 1 ? ' memory' : ' memories') + ' + ' + Math.round(plan.noise * 100) +
          '% noise, then flipping 784 weighted coins...', 'math');
    }

    const finish = function () {
      if (run !== dreamRun) { return; }
      dreamTimer = 0;
      renderDreamFrame(1.01, final, order, tone);
      finishDream(label, plan, final, ink, tone);
    };

    if (prefersReducedMotion()) { finish(); return; }

    const start = performance.now();
    const step = function () {
      if (run !== dreamRun) { return; }          // a newer dream took over
      const t = Math.min(1, (performance.now() - start) / DREAM_MS);
      const reveal = t * t * (3 - 2 * t);       // smoothstep: slow start, fast middle, gentle landing
      renderDreamFrame(reveal, final, order, tone);
      if (t < 1) { dreamTimer = window.setTimeout(step, DREAM_FRAME); }
      else { finish(); }
    };
    step();
  }

  function finishDream(label, plan, final, ink, tone) {
    const lower = label.toLowerCase();
    dreamInkEl.textContent = ink + ' / 784';

    // 12f. self-check: run the dream through the same k-NN classifier
    let check = null;
    if (trainingData.length > 0) {
      check = knnScore(final);
      dreamCheckEl.textContent = check.leader + ' ' + Math.round(check.leaderPct) + '%';
      dreamCheckEl.style.color = check.leader === label ? 'var(--emerald)' : 'var(--amber)';
    } else {
      dreamCheckEl.textContent = 'no memories';
      dreamCheckEl.style.color = 'var(--rose)';
    }

    if (plan.used === 0) {
      if (trainingData.length === 0) {
        setDreamCard('Pure static. I have zero memories.',
                     'With nothing to remember, all 784 pixels were coin flips. Teach me a shape, then ask me to dream again.',
                     'empty');
        speak("I don't know any shapes yet, so all I can imagine is static.");
      } else {
        setDreamCard('Static. I have never seen a ' + lower + '.',
                     'I know other shapes, but not this one, so every pixel was a coin flip. Train me on a ' + lower + ' and try again.',
                     'empty');
        speak('I have never seen a ' + lower + ', so all I can imagine is static.');
      }
      log('Dream result: pure noise, ' + ink + ' random ink pixels.' +
          (check ? ' Even static gets a guess: my classifier calls it ' + check.leader + ' (' +
                   Math.round(check.leaderPct) + '%).' : ''), 'warn');
    } else {
      const fuzzy = plan.noise > 0.25;
      const agrees = check && check.leader === label;
      setDreamCard('Here is a ' + lower + ' I imagined!',
                   'Invented from ' + plan.used + (plan.used === 1 ? ' memory' : ' memories') + '. ' +
                   (fuzzy ? 'Still fuzzy, because I have only seen a few. More drawings will sharpen my imagination.'
                          : 'Nice and sharp, because you gave me plenty of examples.'),
                   fuzzy || !agrees ? 'unsure' : 'sure');
      if (agrees) {
        speak('Here is a ' + lower + ' I imagined. Even I think it looks like a ' + lower + '.');
      } else {
        speak('I tried to imagine a ' + lower + ', but it came out looking more like a ' +
              check.leader.toLowerCase() + '.');
      }
      log('Dream result: ' + ink + ' ink pixels. Self-check: my own classifier calls it ' + check.leader +
          ' (' + Math.round(check.leaderPct) + '%)' + (agrees ? '.' : ', not a ' + label + '!'), agrees ? 'ok' : 'warn');
    }

    addDreamThumb(final, plan.used === 0 ? 'static' : label, tone);
  }

  function addDreamThumb(final, caption, tone) {
    const fig = document.createElement('figure');
    fig.className = 'dream-thumb fresh';
    const c = document.createElement('canvas');
    c.width = GRID * 2;
    c.height = GRID * 2;
    c.setAttribute('aria-label', 'Dreamed ' + caption);
    const g = c.getContext('2d');
    g.fillStyle = '#040b18';
    g.fillRect(0, 0, c.width, c.height);
    g.fillStyle = tone;
    for (let gy = 0; gy < GRID; gy++) {
      for (let gx = 0; gx < GRID; gx++) {
        if (final[gy * GRID + gx] === 1) { g.fillRect(gx * 2, gy * 2, 2, 2); }
      }
    }
    const cap = document.createElement('figcaption');
    cap.textContent = caption;
    fig.appendChild(c);
    fig.appendChild(cap);
    dreamGallery.insertBefore(fig, dreamGallery.firstChild);
    while (dreamGallery.children.length > DREAM_KEEP) {
      dreamGallery.removeChild(dreamGallery.lastChild);
    }
    dreamEmpty.hidden = true;
  }

  function dreamIdle() {
    dreamFromEl.textContent = '—';
    dreamNoiseEl.textContent = '—';
    dreamInkEl.textContent = '—';
    dreamCheckEl.textContent = '—';
    dreamCheckEl.style.color = '';
    markActiveDream(null);
    renderDreamFrame(0, null, null, STATIC_TONE);   // a still frame of static
    setDreamCard('Nothing imagined yet.',
                 'Pick a shape and I will invent a brand-new one from my memories. No memories means pure static.',
                 null);
  }

  function resetDreams() {
    dreamRun++;                                    // cancel any dream mid-animation
    if (dreamTimer) { window.clearTimeout(dreamTimer); dreamTimer = 0; }
    dreamGallery.innerHTML = '';
    dreamEmpty.hidden = false;
    dreamIdle();
  }

  function dreamSomething() {
    dream(SHAPES[Math.floor(Math.random() * SHAPES.length)]);
  }

  /* ============================================================
     13. Build the buttons and bars
     ============================================================ */
  function buildTrainButtons() {
    for (let i = 0; i < SHAPES.length; i++) {
      const name = SHAPES[i];
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.id = 'train-' + name;
      btn.className = 'train-btn';
      btn.style.setProperty('--tone', META[name].tone);
      btn.setAttribute('aria-label', 'Train the AI: this drawing is a ' + name);

      const glyph = document.createElement('span');
      glyph.className = 'glyph';
      glyph.setAttribute('aria-hidden', 'true');
      glyph.textContent = META[name].glyph;

      const text = document.createElement('span');
      text.textContent = 'Train: ' + name;

      const tally = document.createElement('span');
      tally.className = 'tally';
      tally.textContent = '—';
      tallyEls[name] = tally;

      btn.appendChild(glyph);
      btn.appendChild(text);
      btn.appendChild(tally);
      btn.addEventListener('click', function () { trainAs(name); });
      trainBtnsEl.appendChild(btn);
    }
  }

  function buildBars() {
    for (let i = 0; i < SHAPES.length; i++) {
      const name = SHAPES[i];
      const row = document.createElement('div');
      row.className = 'bar-row grid items-center gap-3';
      row.style.gridTemplateColumns = 'minmax(92px,118px) minmax(0,1fr) 64px';
      row.style.setProperty('--tone', META[name].tone);
      row.setAttribute('role', 'progressbar');
      row.setAttribute('aria-valuemin', '0');
      row.setAttribute('aria-valuemax', '100');
      row.setAttribute('aria-valuenow', '0');
      row.setAttribute('aria-label', name + ' confidence');

      const label = document.createElement('div');
      label.className = 'flex items-center gap-2 text-[13px] font-bold';
      const g = document.createElement('span');
      g.style.color = META[name].tone;
      g.setAttribute('aria-hidden', 'true');
      g.textContent = META[name].glyph;
      const nm = document.createElement('span');
      nm.className = 'bar-name';
      nm.textContent = name;
      label.appendChild(g);
      label.appendChild(nm);

      const track = document.createElement('div');
      track.className = 'bar-track';
      const fill = document.createElement('div');
      fill.className = 'bar-fill';
      track.appendChild(fill);

      const right = document.createElement('div');
      right.className = 'flex items-center justify-end gap-1 font-mono text-[12px] tnum';
      const pct = document.createElement('span');
      pct.textContent = '0%';
      const crown = document.createElement('span');
      crown.className = 'crown';
      crown.style.color = META[name].tone;
      crown.setAttribute('aria-hidden', 'true');
      crown.textContent = '◀';
      right.appendChild(pct);
      right.appendChild(crown);

      row.appendChild(label);
      row.appendChild(track);
      row.appendChild(right);
      barsEl.appendChild(row);

      barRows[name] = row;
      barFills[name] = fill;
      barPcts[name] = pct;
    }
  }

  function buildDreamButtons() {
    for (let i = 0; i < SHAPES.length; i++) {
      const name = SHAPES[i];
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'dream-btn';
      btn.style.setProperty('--tone', META[name].tone);
      btn.setAttribute('aria-label', 'Ask the AI to imagine a brand-new ' + name);

      const glyph = document.createElement('span');
      glyph.setAttribute('aria-hidden', 'true');
      glyph.style.fontSize = '17px';
      glyph.textContent = META[name].glyph;

      const text = document.createElement('span');
      text.textContent = 'Dream: ' + name;

      const noise = document.createElement('span');
      noise.className = 'noise';
      noise.textContent = '100% noise';

      btn.appendChild(glyph);
      btn.appendChild(text);
      btn.appendChild(noise);
      btn.addEventListener('click', function () { dream(name); });
      dreamBtnsEl.appendChild(btn);

      dreamBtnEls[name] = btn;
      dreamNoiseEls[name] = noise;
    }
  }

  /* ============================================================
     14. Events
     ============================================================ */
  function onPointerDown(e) {
    if (e.pointerType === 'mouse' && e.button !== 0) { return; }
    drawing = true;
    padFrame.classList.add('is-drawing');
    try { pad.setPointerCapture(e.pointerId); } catch (err) {}
    currentStroke = [pointFrom(e)];
    strokes.push(currentStroke);
    redrawPad();
    setPadHint();
    e.preventDefault();
  }

  function onPointerMove(e) {
    if (!drawing || !currentStroke) { return; }
    const p = pointFrom(e);
    const last = currentStroke[currentStroke.length - 1];
    if (Math.abs(p.x - last.x) + Math.abs(p.y - last.y) < 2) { return; }
    currentStroke.push(p);
    redrawPad();
    schedulePreview();
    e.preventDefault();
  }

  function onPointerUp() {
    if (!drawing) { return; }
    drawing = false;
    currentStroke = null;
    padFrame.classList.remove('is-drawing');
    if (previewTimer) { window.clearTimeout(previewTimer); previewTimer = 0; }
    classify();          // one fresh inference per finished stroke
  }

  pad.addEventListener('pointerdown', onPointerDown);
  window.addEventListener('pointermove', onPointerMove);
  window.addEventListener('pointerup', onPointerUp);
  window.addEventListener('pointercancel', onPointerUp);

  pixels.addEventListener('pointermove', function (e) {
    const r = pixels.getBoundingClientRect();
    const gx = Math.floor(((e.clientX - r.left) / r.width) * GRID);
    const gy = Math.floor(((e.clientY - r.top) / r.height) * GRID);
    if (gx < 0 || gy < 0 || gx >= GRID || gy >= GRID) { return; }
    const v = lastVector ? lastVector[gy * GRID + gx] : 0;
    statCursor.textContent = 'row ' + gy + ', col ' + gx + ' = ' + v;
  });
  pixels.addEventListener('pointerleave', function () {
    statCursor.textContent = '—';
  });

  undoBtn.addEventListener('click', undo);
  clearBtn.addEventListener('click', clearPad);
  resetBtn.addEventListener('click', resetBrain);
  soundBtn.addEventListener('click', toggleVoice);
  dreamSurprise.addEventListener('click', dreamSomething);
  logClear.addEventListener('click', function () {
    logEl.innerHTML = '';
    log('Log cleared. Memories untouched: ' + trainingData.length + '.', 'info');
  });

  window.addEventListener('keydown', function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) { return; }
    const t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) { return; }
    const k = String(e.key).toLowerCase();
    if (k === '1' || k === '2' || k === '3' || k === '4') {
      trainAs(SHAPES[Number(k) - 1]);
      e.preventDefault();
    } else if (k === 'u') {
      undo();
    } else if (k === 'c') {
      clearPad();
    } else if (k === 'd') {
      dreamSomething();
    } else if (k === 'v') {
      toggleVoice();
    } else if (k === 'r') {
      resetBrain();
    }
  });

  /* ============================================================
     15. Boot
     ============================================================ */
  buildTrainButtons();
  buildBars();
  buildDreamButtons();
  dreamIdle();
  redrawPad();
  setPadHint();
  renderPixels(null);
  updatePixelStats(null);
  setBars(ZERO, null);
  updateStats();

  if (!canSpeak) {
    voiceOn = false;
    soundBtn.textContent = 'Voice: n/a';
    soundBtn.disabled = true;
    soundBtn.setAttribute('aria-pressed', 'false');
  }

  log('Brain online. Plain JavaScript k-nearest neighbours, 0 memories, 784 features per drawing.', 'ok');
  log('Nothing is pre-trained. I genuinely cannot tell a circle from a star yet.', 'warn');
  log('Draw a shape, then press a Train button to give me my first memory.', 'info');
})();
