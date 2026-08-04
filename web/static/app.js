/* Sperm-analysis research demo -- frontend.
 *
 * Rules this file obeys, and why:
 *
 * 1. It contains no product logic. The 60% rule, the health rule, the aspect
 *    order, the 0/1 label convention, the motility grades and the slider ranges
 *    all arrive from the server (`/aspects`, `/config`, `/decide`). A second
 *    copy of any of them in JavaScript is a second source of truth, and the one
 *    that is easiest to change without a test noticing. Where this file appears
 *    to make a decision, look again: it is choosing which server-returned rows
 *    to display, never what they say.
 * 2. No external requests. No framework, no bundler, no CDN, no web font. The
 *    device runs offline; a demo that needs the internet is not a demo of it.
 * 3. The prediction is labelled untrained wherever it is visible -- the top
 *    banner, the comparison caveat, the button note and the footer.
 * 4. Nothing is signalled by colour alone. Every coloured cell also carries a
 *    symbol and a word.
 */

"use strict";

(function () {
  // ---------------------------------------------------------------- helpers

  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  };

  const REDUCED_MOTION =
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /** Application state. `meta` and `config` are server truth, fetched once. */
  const S = {
    meta: null,
    config: null,
    generated: null,
    classified: null,
    knobOverrides: Object.create(null),
    forcedAspects: null, // array of four 0/1 once forcing is enabled
    ladderFor: null,
    anim: { raf: 0, points: null, index: 0, lastTime: 0 },
  };

  function showError(message) {
    const toast = $("error-toast");
    toast.textContent = message;
    toast.classList.remove("hidden");
    window.clearTimeout(showError.timer);
    showError.timer = window.setTimeout(() => toast.classList.add("hidden"), 9000);
  }

  async function request(url, body) {
    const options = body === undefined
      ? { method: "GET" }
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        };
    const response = await fetch(url, options);
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const payload = await response.json();
        detail = typeof payload.detail === "string"
          ? payload.detail
          : JSON.stringify(payload.detail || payload);
      } catch (ignored) {
        /* the body was not JSON; the status text is all we have */
      }
      throw new Error(`${url} -> ${response.status}: ${detail}`);
    }
    return response.json();
  }

  const fmt = (value, digits) =>
    value === null || value === undefined || Number.isNaN(value)
      ? "n/a"
      : Number(value).toFixed(digits === undefined ? 2 : digits);

  function debounce(fn, ms) {
    let timer = 0;
    return function debounced() {
      window.clearTimeout(timer);
      timer = window.setTimeout(fn, ms);
    };
  }

  /** Symbol + word + colour class for a comparison outcome. */
  function agreementPill(kind) {
    if (kind === "agree") return { mark: "✓", word: "agree", cls: "pill-ok" };
    if (kind === "disagree") return { mark: "✗", word: "DISAGREE", cls: "pill-bad" };
    return { mark: "?", word: "not comparable", cls: "pill-warn" };
  }

  function pill(kind, extra) {
    const spec = agreementPill(kind);
    const node = el("span", "pill " + spec.cls);
    node.appendChild(el("span", "pill-mark", spec.mark));
    node.appendChild(el("span", null, extra === undefined ? spec.word : extra));
    return node;
  }

  function labelPill(labelValue) {
    const names = S.meta.label_names;
    const isNormal = Number(labelValue) === S.meta.label_normal;
    const node = el("span", "pill " + (isNormal ? "pill-ok" : "pill-bad"));
    node.appendChild(el("span", "pill-mark", isNormal ? "○" : "●"));
    node.appendChild(el("span", null, `${names[String(labelValue)]} (${labelValue})`));
    return node;
  }

  // ------------------------------------------------------------ canvas: crop

  /** Size a canvas' backing store to its rendered size, in device pixels. */
  function fitCanvas(canvas) {
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
    const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    return { width, height };
  }

  /**
   * Draw the crop at an *integer* magnification with smoothing disabled.
   *
   * An integer factor matters: a fractional one resamples, and a resampled
   * pixel is a pixel the generator never produced. The whole claim of this
   * panel is that each visible block is one real sample.
   */
  function drawCrop(image) {
    const canvas = $("image-canvas");
    const box = fitCanvas(canvas);
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, box.width, box.height);
    const factor = Math.max(1, Math.floor(Math.min(box.width / image.width, box.height / image.height)));
    const drawWidth = image.width * factor;
    const drawHeight = image.height * factor;
    ctx.drawImage(
      image,
      Math.floor((box.width - drawWidth) / 2),
      Math.floor((box.height - drawHeight) / 2),
      drawWidth,
      drawHeight
    );
    return factor;
  }

  /**
   * Decode the returned PNG and paint it. Callback-driven, not awaited.
   *
   * `HTMLImageElement.decode()` is the tidier API but its promise can stay
   * pending indefinitely while the document is hidden, which would stall the
   * whole generate/classify/decide chain behind an image nobody is looking at.
   * `onload` fires regardless of visibility, and nothing downstream needs the
   * decoded bitmap, so the pipeline no longer waits for it.
   */
  function renderCrop(payload) {
    const image = new Image();
    image.onload = () => {
      S.generated.decodedImage = image;
      const factor = drawCrop(image);
      $("image-meta").textContent =
        `${payload.image_shape[0]}x${payload.image_shape[1]} px, ` +
        `${fmt(payload.image_um_per_px, 3)} um/px, ` +
        `${fmt(payload.image_field_um, 1)} um field, x${factor} nearest-neighbour`;
    };
    image.onerror = () => {
      showError("the PNG returned by /generate could not be decoded by the browser");
    };
    image.src = "data:image/png;base64," + payload.image;
  }

  // ------------------------------------------------------ canvas: trajectory

  function trackTransform(points, box) {
    const pad = 18 * (window.devicePixelRatio || 1);
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const [x, y] of points) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const spanX = Math.max(maxX - minX, 1e-6);
    const spanY = Math.max(maxY - minY, 1e-6);
    // One isotropic scale for both axes: an anisotropic fit would make a
    // straight swim look curved and a circling cell look linear, which is
    // precisely the distinction the panel exists to show.
    const scale = Math.min((box.width - 2 * pad) / spanX, (box.height - 2 * pad) / spanY);
    return {
      scale,
      toX: (x) => pad + (x - minX) * scale + (box.width - 2 * pad - spanX * scale) / 2,
      toY: (y) => pad + (y - minY) * scale + (box.height - 2 * pad - spanY * scale) / 2,
    };
  }

  function drawTrack(upTo) {
    const canvas = $("track-canvas");
    const box = fitCanvas(canvas);
    const ctx = canvas.getContext("2d");
    const points = S.anim.points;
    if (!points || points.length < 2) return;
    const t = trackTransform(points, box);
    const ratio = window.devicePixelRatio || 1;
    const style = getComputedStyle(document.body);
    const ink = style.getPropertyValue("--ink").trim() || "#000";
    const faint = style.getPropertyValue("--ink-faint").trim() || "#888";
    const accent = style.getPropertyValue("--accent").trim() || "#1d4ed8";

    ctx.clearRect(0, 0, box.width, box.height);

    // Whole path, faint: the trail says "when", this says "where in total".
    ctx.strokeStyle = faint;
    ctx.globalAlpha = 0.28;
    ctx.lineWidth = 1 * ratio;
    ctx.beginPath();
    ctx.moveTo(t.toX(points[0][0]), t.toY(points[0][1]));
    for (let i = 1; i < points.length; i++) ctx.lineTo(t.toX(points[i][0]), t.toY(points[i][1]));
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Fading trail behind the head.
    const trail = Math.max(8, Math.round(points.length / 3));
    const start = Math.max(0, upTo - trail);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (let i = start; i < upTo; i++) {
      const age = (upTo - i) / trail;
      ctx.globalAlpha = Math.max(0.05, 1 - age);
      ctx.lineWidth = (1 + 2.2 * (1 - age)) * ratio;
      ctx.strokeStyle = accent;
      ctx.beginPath();
      ctx.moveTo(t.toX(points[i][0]), t.toY(points[i][1]));
      ctx.lineTo(t.toX(points[i + 1][0]), t.toY(points[i + 1][1]));
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Start marker and current head.
    ctx.fillStyle = faint;
    ctx.beginPath();
    ctx.arc(t.toX(points[0][0]), t.toY(points[0][1]), 3 * ratio, 0, Math.PI * 2);
    ctx.fill();

    const head = points[Math.min(upTo, points.length - 1)];
    ctx.fillStyle = accent;
    ctx.beginPath();
    ctx.arc(t.toX(head[0]), t.toY(head[1]), 4.5 * ratio, 0, Math.PI * 2);
    ctx.fill();

    // Scale bar, in micrometres, derived from the sampling the server used.
    const umPerPx = S.generated ? S.generated.payload.track_um_per_px : null;
    if (umPerPx) {
      const targetPx = box.width * 0.22;
      const rawUm = (targetPx / t.scale) * umPerPx;
      const step = Math.pow(10, Math.floor(Math.log10(rawUm)));
      const niceUm = Math.max(step, Math.round(rawUm / step) * step);
      const barPx = (niceUm / umPerPx) * t.scale;
      const y = box.height - 12 * ratio;
      const x = 12 * ratio;
      ctx.strokeStyle = ink;
      ctx.globalAlpha = 0.75;
      ctx.lineWidth = 2 * ratio;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + barPx, y);
      ctx.stroke();
      ctx.fillStyle = ink;
      ctx.font = `${11 * ratio}px ui-monospace, monospace`;
      ctx.fillText(`${niceUm} um`, x, y - 5 * ratio);
      ctx.globalAlpha = 1;
    }
  }

  function animateTrack(points) {
    window.cancelAnimationFrame(S.anim.raf);
    S.anim.points = points;
    S.anim.index = 0;
    S.anim.lastTime = 0;
    if (REDUCED_MOTION) {
      S.anim.index = points.length - 1;
      drawTrack(S.anim.index);
      return;
    }
    // Paint the whole path once, synchronously, before handing over to the
    // animation. requestAnimationFrame does not fire in a hidden tab, so a page
    // loaded in the background would otherwise show an empty panel until it was
    // brought to the front -- and a blank canvas next to a populated CASA table
    // reads as a broken demo rather than as a paused one.
    drawTrack(points.length - 1);
    // Four seconds for the whole track regardless of its length, so a 512-point
    // track is not four times slower to watch than a 128-point one.
    const perMs = points.length / 4000;
    const step = (now) => {
      if (!S.anim.lastTime) S.anim.lastTime = now;
      S.anim.index += (now - S.anim.lastTime) * perMs;
      S.anim.lastTime = now;
      if (S.anim.index >= points.length - 1) S.anim.index = 0;
      drawTrack(Math.floor(S.anim.index));
      S.anim.raf = window.requestAnimationFrame(step);
    };
    S.anim.raf = window.requestAnimationFrame(step);
  }

  // ------------------------------------------------------------- control DOM

  function sliderRow(id, labelText, min, max, step, value, formatter) {
    const row = el("div", "ctl-row slider-row");
    row.id = id + "_row";
    const label = el("label", null, labelText);
    label.htmlFor = id;
    const input = document.createElement("input");
    input.type = "range";
    input.id = id;
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    input.value = String(value);
    const out = el("output", null, formatter(value));
    out.id = id + "_out";
    input.addEventListener("input", () => {
      out.textContent = formatter(Number(input.value));
    });
    row.append(label, input, out);
    return row;
  }

  function buildMotilitySelect() {
    const select = $("motility");
    select.innerHTML = "";
    const sampled = el("option", null, "sample from P(progressive)");
    sampled.value = "";
    select.appendChild(sampled);
    for (const name of S.meta.motility_classes) {
      const option = el(
        "option",
        null,
        name.replace(/_/g, " ") +
          (S.meta.progressive_classes.indexOf(name) >= 0 ? "  (progressive)" : "")
      );
      option.value = name;
      select.appendChild(option);
    }
    select.value = "rapid_progressive";
  }

  function buildPrevalenceSliders() {
    const host = $("prevalence-sliders");
    host.innerHTML = "";
    for (const aspect of S.meta.aspects) {
      host.appendChild(
        sliderRow(
          "prev_" + aspect,
          aspect,
          0,
          1,
          0.01,
          S.meta.default_prevalences[aspect],
          (v) => fmt(v, 2)
        )
      );
    }
  }

  function buildAspectToggles() {
    const host = $("aspect-toggles");
    host.innerHTML = "";
    S.forcedAspects = S.meta.aspects.map(() => S.meta.label_normal);
    S.meta.aspects.forEach((aspect, index) => {
      const row = el("div", "aspect-toggle");
      row.appendChild(el("span", null, aspect));
      const seg = el("div", "seg");
      [S.meta.label_normal, S.meta.label_abnormal].forEach((value) => {
        const button = el("button", null, `${S.meta.label_names[String(value)]} (${value})`);
        button.type = "button";
        button.dataset.aspect = String(index);
        button.dataset.value = String(value);
        button.addEventListener("click", () => {
          S.forcedAspects[index] = value;
          syncAspectToggles();
          scheduleGenerate();
        });
        seg.appendChild(button);
      });
      row.appendChild(seg);
      host.appendChild(row);
    });
    syncAspectToggles();

    const quick = $("aspect-quick");
    quick.innerHTML = "";
    const allNormal = el("button", null, "all four normal");
    allNormal.type = "button";
    allNormal.addEventListener("click", () => {
      S.forcedAspects = S.meta.aspects.map(() => S.meta.label_normal);
      syncAspectToggles();
      scheduleGenerate();
    });
    quick.appendChild(allNormal);
    S.meta.aspects.forEach((aspect, index) => {
      const button = el("button", null, `only ${aspect} abnormal`);
      button.type = "button";
      button.addEventListener("click", () => {
        S.forcedAspects = S.meta.aspects.map((_, i) =>
          i === index ? S.meta.label_abnormal : S.meta.label_normal
        );
        syncAspectToggles();
        scheduleGenerate();
      });
      quick.appendChild(button);
    });
  }

  function syncAspectToggles() {
    for (const button of $("aspect-toggles").querySelectorAll("button")) {
      const index = Number(button.dataset.aspect);
      const value = Number(button.dataset.value);
      button.classList.toggle("is-active", S.forcedAspects[index] === value);
    }
  }

  function buildKnobSliders() {
    const host = $("knob-sliders");
    host.innerHTML = "";
    for (const knob of S.meta.knobs) {
      const mid = (knob.min + knob.max) / 2;
      const row = sliderRow(
        "knob_" + knob.name,
        knob.label,
        knob.min,
        knob.max,
        knob.step,
        mid,
        (v) => `${fmt(v, 2)} ${knob.unit || ""}`.trim()
      );
      const input = row.querySelector("input");
      input.addEventListener("input", () => {
        S.knobOverrides[knob.name] = Number(input.value);
        row.classList.add("is-overridden");
        scheduleGenerate();
      });
      const title = [];
      if (knob.driven_by) title.push(`normally caused by the '${knob.driven_by}' label`);
      if (knob.normal_band) title.push(`normal band ${knob.normal_band[0]} to ${knob.normal_band[1]}`);
      if (knob.note) title.push(knob.note);
      if (!knob.breaks_label_link) title.push("nuisance parameter: overriding it does not affect any label");
      row.title = title.join("; ");
      host.appendChild(row);
    }
    $("clear-knobs").addEventListener("click", () => {
      S.knobOverrides = Object.create(null);
      for (const row of host.querySelectorAll(".slider-row")) row.classList.remove("is-overridden");
      scheduleGenerate();
    });
  }

  function buildPresetCases() {
    const host = $("preset-cases");
    host.innerHTML = "";
    for (const testCase of S.meta.mandated_decision_cases) {
      const button = el("button", null, testCase.caption);
      button.type = "button";
      button.title = testCase.why;
      button.addEventListener("click", () => {
        $("trackable").value = String(testCase.trackable_count);
        $("eligible").value = String(testCase.ai_eligible_count);
        syncDecisionInputs();
        runDecision(true);
      });
      host.appendChild(button);
    }
  }

  // ------------------------------------------------------------- generation

  function readParams() {
    const params = {
      seed: Number($("seed").value) || 0,
      prevalences: {},
      motility: $("motility").value === "" ? null : $("motility").value,
      progressive_rate: Number($("progressive_rate").value),
      aspects: $("force_aspects").checked ? S.forcedAspects.slice() : null,
      knobs: Object.assign({}, S.knobOverrides),
      image_size: Number($("image_size").value),
      blur_px: Number($("blur_px").value),
      noise_sigma: Number($("noise_sigma").value),
      n_points: Number($("n_points").value),
      fps: Number($("fps").value),
      track_um_per_px: Number($("track_um_per_px").value),
      flow_vx_px_s: Number($("flow_vx_px_s").value),
      flow_vy_px_s: Number($("flow_vy_px_s").value),
    };
    for (const aspect of S.meta.aspects) {
      params.prevalences[aspect] = Number($("prev_" + aspect).value);
    }
    return params;
  }

  function renderOverrideWarning(payload) {
    const node = $("override-warning");
    const breaking = payload.overridden_knobs.filter((name) => {
      const knob = S.meta.knobs.find((k) => k.name === name);
      return knob && knob.breaks_label_link;
    });
    if (!payload.overridden_knobs.length) {
      node.classList.add("hidden");
      node.textContent = "";
      return;
    }
    node.classList.remove("hidden");
    node.textContent = breaking.length
      ? `⚠ Manual override active on ${breaking.join(", ")}. ` +
        "These knobs are normally caused by the binary labels, so the label shown " +
        "below is the label that was drawn, not a description of what you now see. " +
        "The generator's label/pixel guarantee is suspended for this cell."
      : `Override active on ${payload.overridden_knobs.join(", ")}. ` +
        "These are nuisance parameters, so no label is affected.";
  }

  function renderCasa(payload) {
    const body = $("casa-table").querySelector("tbody");
    body.innerHTML = "";
    const units = { vcl: "um/s", vsl: "um/s", vap: "um/s", lin: "", str: "", wob: "", alh: "um", bcf: "Hz" };
    for (const name of S.meta.casa_feature_names) {
      const row = el("tr");
      row.appendChild(el("th", null, name.toUpperCase()));
      const corrected = payload.casa[name];
      const observed = payload.casa_observed[name];
      const drifted = payload.flow_correction_applied && Math.abs(corrected - observed) > 1e-6;
      row.appendChild(
        el(
          "td",
          null,
          `${fmt(corrected, 2)} ${units[name]}`.trim() +
            (drifted ? `   (uncorrected ${fmt(observed, 2)})` : "")
        )
      );
      body.appendChild(row);
    }
    const row = el("tr");
    row.appendChild(el("th", null, "duration"));
    row.appendChild(el("td", null, `${fmt((payload.trajectory.length - 1) * payload.dt_s, 3)} s`));
    body.appendChild(row);
  }

  /**
   * Move every un-overridden knob slider to the value the generator produced.
   *
   * Without this the slider sits at its midpoint and reads as the value in
   * force, which would invert the panel's message: these knobs are *outputs* of
   * the label until you touch one, and the control should show that.
   */
  function syncKnobSliders(payload) {
    for (const knob of S.meta.knobs) {
      if (knob.name in S.knobOverrides) continue;
      const value = payload.state[knob.name];
      if (value === undefined) continue;
      const input = $("knob_" + knob.name);
      const out = $("knob_" + knob.name + "_out");
      if (!input || !out) continue;
      input.value = String(Math.min(knob.max, Math.max(knob.min, value)));
      out.textContent = `${fmt(value, 2)} ${knob.unit || ""}`.trim();
    }
  }

  function renderState(payload) {
    const body = $("state-table").querySelector("tbody");
    body.innerHTML = "";
    for (const knob of S.meta.knobs) {
      const value = payload.state[knob.name];
      if (value === undefined) continue;
      const row = el("tr");
      const offBand =
        knob.normal_band && (value < knob.normal_band[0] || value > knob.normal_band[1]);
      if (offBand) row.className = "is-off-band";
      row.appendChild(
        el("th", null, knob.label + (payload.overridden_knobs.indexOf(knob.name) >= 0 ? " *" : ""))
      );
      row.appendChild(
        el("td", null, `${fmt(value, 3)} ${knob.unit || ""}`.trim() + (offBand ? "  (off band)" : ""))
      );
      body.appendChild(row);
    }
    const note = el("tr");
    note.appendChild(el("th", null, "* = manually overridden"));
    note.appendChild(el("td", null, payload.overridden_knobs.length ? "yes" : "none"));
    body.appendChild(note);
  }

  async function runGenerate() {
    const params = readParams();
    let payload;
    try {
      payload = await request("/generate", params);
    } catch (error) {
      showError(String(error.message || error));
      return;
    }
    S.generated = { params, payload };
    S.classified = null;
    renderCrop(payload);
    animateTrack(payload.trajectory);
    $("track-meta").textContent =
      `${payload.trajectory.length} points @ ${fmt(payload.fps, 0)} fps, ` +
      `${fmt(payload.track_um_per_px, 3)} um/px` +
      (payload.flow_correction_applied
        ? `, bulk flow (${payload.flow_px_s[0]}, ${payload.flow_px_s[1]}) px/s removed before grading`
        : "");
    renderOverrideWarning(payload);
    renderCasa(payload);
    syncKnobSliders(payload);
    renderState(payload);
    renderComparison();
    // Classify straight away so the comparison table is never empty. The
    // button remains, because pressing it repeatedly on an unchanged image is
    // the fastest way to see for yourself that the model is untrained: the
    // answers change every time.
    await runClassify();
  }

  const scheduleGenerate = debounce(() => {
    runGenerate();
  }, 160);

  // ------------------------------------------------------------ comparison

  /**
   * Collapse a four-way motility class onto the generator's three-way label.
   *
   * Which classes count as progressive comes from `/aspects.progressive_classes`
   * -- the server's own `MotilityClass.is_progressive` -- so this function
   * groups, it does not decide.
   */
  function collapseMotility(className) {
    if (S.meta.progressive_classes.indexOf(className) >= 0) return "progressive";
    if (S.meta.motility_label_names.indexOf(className) >= 0) return className;
    return "undetermined";
  }

  function comparisonRow(item, trueCell, predCell, probCell, agreement) {
    const row = el("tr");
    if (agreement === "disagree") row.className = "disagree";
    else if (agreement === "unknown") row.className = "unknown";
    const head = el("th", null, item);
    head.scope = "row";
    row.appendChild(head);
    const trueTd = el("td");
    trueTd.appendChild(trueCell);
    row.appendChild(trueTd);
    const predTd = el("td");
    predTd.appendChild(predCell);
    row.appendChild(predTd);
    row.appendChild(el("td", "mono-cell", probCell));
    const agreeTd = el("td");
    agreeTd.appendChild(pill(agreement));
    row.appendChild(agreeTd);
    return row;
  }

  function renderComparison() {
    const body = $("compare-table").querySelector("tbody");
    body.innerHTML = "";
    if (!S.generated) return;
    const truth = S.generated.payload;
    const pred = S.classified;

    for (const aspect of S.meta.aspects) {
      const trueLabel = truth.true_aspects[aspect];
      let predCell = el("span", "pill pill-warn", "not classified yet");
      let prob = "–";
      let agreement = "unknown";
      if (pred) {
        const predLabel = pred.pred_aspects[aspect];
        predCell = labelPill(predLabel);
        const detail = pred.aspect_detail[aspect];
        prob = `${fmt(detail.p_normal, 3)}  (threshold ${fmt(detail.threshold, 2)})`;
        agreement = predLabel === trueLabel ? "agree" : "disagree";
      }
      body.appendChild(comparisonRow(aspect, labelPill(trueLabel), predCell, prob, agreement));
    }

    // Motility.
    const trueMotility = el("span", "pill pill-info");
    trueMotility.appendChild(el("span", "pill-mark", "→"));
    trueMotility.appendChild(
      el("span", null, `${truth.true_motility_label_name}  (${truth.true_motility})`)
    );
    let predMotility = el("span", "pill pill-warn", "not classified yet");
    let motilityAgreement = "unknown";
    let motilityNote = "–";
    if (pred) {
      const collapsed = collapseMotility(pred.pred_motility);
      const known = collapsed !== "undetermined";
      predMotility = el("span", "pill " + (known ? "pill-info" : "pill-warn"));
      predMotility.appendChild(el("span", "pill-mark", known ? "→" : "?"));
      predMotility.appendChild(el("span", null, `${collapsed}  (${pred.pred_motility})`));
      motilityAgreement = !known
        ? "unknown"
        : collapsed === truth.true_motility_label_name
        ? "agree"
        : "disagree";
      motilityNote = pred.motility_source === "casa_rule" ? "WHO rule on CASA" : "no track";
    }
    body.appendChild(
      comparisonRow("motility", trueMotility, predMotility, motilityNote, motilityAgreement)
    );

    // Overall.
    const trueOverall = el("span", "pill " + (truth.true_label === 0 ? "pill-ok" : "pill-bad"));
    trueOverall.appendChild(el("span", "pill-mark", truth.true_label === 0 ? "○" : "●"));
    trueOverall.appendChild(el("span", null, `${truth.true_label_name} (${truth.true_label})`));
    let predOverall = el("span", "pill pill-warn", "not classified yet");
    let overallAgreement = "unknown";
    if (pred) {
      predOverall = el("span", "pill " + (pred.pred_label === 0 ? "pill-ok" : "pill-bad"));
      predOverall.appendChild(el("span", "pill-mark", pred.pred_label === 0 ? "○" : "●"));
      predOverall.appendChild(el("span", null, `${pred.pred_label_name} (${pred.pred_label})`));
      overallAgreement = pred.pred_label === truth.true_label ? "agree" : "disagree";
    }
    const overallRow = comparisonRow(
      "overall (AI-eligible)",
      trueOverall,
      predOverall,
      pred ? (pred.all_four_normal ? "all four normal" : "not all normal") : "–",
      overallAgreement
    );
    overallRow.classList.add("row-overall");
    body.appendChild(overallRow);

    $("eligibility-note").textContent = pred
      ? `Rule: ${S.meta.health_rule} ` +
        `Predicted eligibility: ${pred.ai_eligible ? "eligible" : "not eligible"} ` +
        `(reason: ${pred.ineligibility_reason}). ` +
        `Motility grade from ${pred.motility_rule}: ${pred.pred_motility_reason}`
      : `Rule: ${S.meta.health_rule}`;
  }

  async function runClassify() {
    if (!S.generated) return;
    try {
      S.classified = await request("/classify", {
        seed: S.generated.params.seed,
        params: S.generated.params,
      });
    } catch (error) {
      showError(String(error.message || error));
      return;
    }
    applyModelProvenance(S.classified.model);
    renderComparison();
  }

  function applyModelProvenance(model) {
    const banner = $("untrained-banner");
    banner.classList.toggle("is-trained", !model.untrained);
    $("untrained-tag").textContent = model.headline;
    $("untrained-warning").textContent = model.untrained
      ? model.untrained_warning
      : `weights provenance: ${model.provenance}`;
    $("untrained-detail").textContent = model.detail;
    $("comparison-caveat").textContent = model.untrained
      ? ` The morphology columns come from ${model.engine_class}, which is ${model.untrained_warning}.`
      : "";
    $("classify-note").textContent = model.untrained
      ? `${model.untrained_warning}. ` +
        (model.reads_the_image ? "" : "The engine never reads the image. ") +
        "Press the button twice on the same crop: the answers change, because they are noise."
      : `weights: ${model.provenance}`;
    $("footer-provenance").textContent =
      `morphology engine: ${model.engine_class} (${model.provenance}); ` +
      `polarity: ${model.label_polarity}`;
  }

  // -------------------------------------------------------------- decision

  function syncDecisionInputs() {
    const trackable = Number($("trackable").value);
    let eligible = Number($("eligible").value);
    // The numerator must be a subset of the denominator. Clamping the *input*
    // is a UI affordance; the server still rejects an impossible pair with a
    // 422, and the demo relies on that rather than on this clamp.
    if (eligible > trackable) {
      eligible = trackable;
      $("eligible").value = String(eligible);
    }
    $("eligible").max = String(Math.max(trackable, 0));
    $("trackable_out").textContent = String(trackable);
    $("eligible_out").textContent = String(eligible);
    $("decide-inputs-note").textContent =
      `${eligible} of ${trackable} sperm in this shot satisfied both progressive ` +
      "motility and all four normal morphology aspects.";
  }

  function renderVerdict(decision) {
    const box = $("verdict");
    box.classList.remove("is-accept", "is-reject", "is-indeterminate");
    box.classList.add("is-" + decision.status);
    $("verdict-ratio").textContent =
      `${decision.ai_eligible_count} / ${decision.trackable_count}` +
      (decision.percent === null ? "" : ` = ${fmt(decision.percent, 2)}%`) +
      (decision.exactly_at_threshold ? "   ← exactly at the threshold" : "");
    $("verdict-status").textContent = decision.status_upper;
    $("verdict-field").textContent = decision.field_command;
    $("verdict-meaning").textContent = decision.field_command_meaning;
    $("verdict-rationale").textContent = decision.rationale;
  }

  async function runDecision(rebuildLadder) {
    const trackable = Number($("trackable").value);
    const eligible = Number($("eligible").value);
    let decision;
    try {
      decision = await request("/decide", {
        ai_eligible_count: eligible,
        trackable_count: trackable,
      });
    } catch (error) {
      showError(String(error.message || error));
      return;
    }
    renderVerdict(decision);
    $("decide-config-note").textContent =
      `threshold ${fmt(decision.threshold, 2)}, minimum trackable ${decision.minimum_trackable}. ` +
      decision.boundary_rule +
      " " +
      decision.minimum_rule;
    if (rebuildLadder || S.ladderFor !== trackable) {
      S.ladderFor = trackable;
      await buildLadder(trackable);
    }
  }

  /**
   * A window of eligible counts around the boundary, each decided by the server.
   *
   * Choosing *which* counts to show is presentation. What each one means is
   * fetched: every row below is a separate `/decide` response, so the flip point
   * in the table is wherever the engine actually puts it, not wherever this file
   * thinks it should be.
   */
  async function buildLadder(trackable) {
    const body = $("ladder-table").querySelector("tbody");
    body.innerHTML = "";
    const low = Math.max(0, Math.floor(trackable * 0.45));
    const high = Math.min(trackable, Math.ceil(trackable * 0.78));
    const counts = [];
    for (let k = low; k <= high && counts.length < 12; k++) counts.push(k);
    if (!counts.length) counts.push(0);

    let decisions;
    try {
      decisions = await Promise.all(
        counts.map((k) =>
          request("/decide", { ai_eligible_count: k, trackable_count: trackable })
        )
      );
    } catch (error) {
      showError(String(error.message || error));
      return;
    }

    let seenAccept = false;
    for (const decision of decisions) {
      const row = el("tr");
      row.appendChild(el("td", "mono-cell", `${decision.ai_eligible_count} / ${decision.trackable_count}`));
      row.appendChild(
        el("td", "mono-cell", decision.percent === null ? "n/a" : `${fmt(decision.percent, 2)}%`)
      );
      const statusTd = el("td");
      const statusPill = el(
        "span",
        "pill " +
          (decision.status === "accept"
            ? "pill-ok"
            : decision.status === "reject"
            ? "pill-bad"
            : "pill-warn")
      );
      statusPill.appendChild(
        el("span", "pill-mark", decision.status === "accept" ? "✓" : decision.status === "reject" ? "✗" : "?")
      );
      statusPill.appendChild(el("span", null, decision.status_upper));
      statusTd.appendChild(statusPill);
      row.appendChild(statusTd);

      const fieldTd = el("td");
      const fieldPill = el("span", "pill " + (decision.is_rejection ? "pill-bad" : "pill-ok"));
      fieldPill.appendChild(el("span", "pill-mark", decision.is_rejection ? "⚡" : "○"));
      fieldPill.appendChild(el("span", null, decision.field_command));
      fieldTd.appendChild(fieldPill);
      row.appendChild(fieldTd);

      const notes = [];
      if (decision.exactly_at_threshold) notes.push("exactly at the threshold — REJECTED");
      if (decision.status === "accept" && !seenAccept) notes.push("first accepting count");
      if (decision.status === "indeterminate") notes.push("below the minimum trackable count");
      if (decision.is_rejection) notes.push("field energised — diverted to waste");
      if (decision.status === "accept") seenAccept = true;
      row.appendChild(el("td", null, notes.join("; ")));

      if (decision.exactly_at_threshold) row.className = "disagree";
      body.appendChild(row);
    }
    $("ladder-note").textContent =
      `Every row is a separate POST /decide against the real engine at ` +
      `threshold ${fmt(decisions[0].threshold, 2)}. Watch the status flip one ` +
      `sperm past the boundary, and watch the field command flip with it.`;
  }

  // ---------------------------------------------------------------- optics

  function statCard(label, value, note, warn) {
    const card = el("div", "stat" + (warn ? " is-warn" : ""));
    card.appendChild(el("div", "stat-label", label));
    card.appendChild(el("div", "stat-value", value));
    if (note) card.appendChild(el("div", "stat-note", note));
    return card;
  }

  function renderOptics(config) {
    const f = config.feasibility;
    const host = $("optics-stats");
    host.innerHTML = "";

    host.appendChild(
      statCard(
        "sampling",
        `${fmt(f.um_per_px, 4)} um/px`,
        f.um_per_px_is_measured ? "from a real calibration" : "NOMINAL optics, never calibrated",
        !f.um_per_px_is_measured
      )
    );
    host.appendChild(
      statCard(
        "field of view",
        `${fmt(f.field_width_um, 1)} x ${fmt(f.field_height_um, 1)} um`,
        `${fmt(f.field_width_px, 0)} x ${fmt(f.field_height_px, 0)} px`,
        false
      )
    );
    host.appendChild(
      statCard(
        "sperm head",
        `${fmt(f.head_span_px, 0)} px`,
        `a ${fmt(f.head_length_um, 1)} um head spans ${fmt(f.head_span_px, 0)} x ` +
          `${fmt(f.head_width_span_px, 0)} px — ample for centroid tracking`,
        false
      )
    );
    host.appendChild(
      statCard(
        "whole spermatozoon",
        f.whole_sperm_fits_across_flow ? "fits" : "DOES NOT FIT",
        `a ${fmt(f.sperm_length_um, 0)} um cell needs ${fmt(f.sperm_span_px, 0)} px; ` +
          `the short field axis holds only ${fmt(100 * f.fraction_of_sperm_across_field, 0)}% of it, ` +
          "so full-flagellum imaging is orientation-dependent and often truncated",
        !f.whole_sperm_fits_across_flow
      )
    );
    host.appendChild(
      statCard(
        "residence time",
        f.residence_time_s === null ? "n/a" : `${fmt(1000 * f.residence_time_s, 1)} ms`,
        f.frames_per_transit === null
          ? "no bulk flow configured"
          : `${fmt(f.frames_per_transit, 1)} frames observed, ${f.min_frames_required} required`,
        f.frames_per_transit !== null && f.frames_per_transit < f.min_frames_required
      )
    );
    host.appendChild(
      statCard(
        "implied concentration",
        f.required_concentration_per_ml === null
          ? "n/a"
          : `${fmt(f.required_concentration_per_ml / 1e6, 1)} M/mL`,
        `to hold ${fmt(f.required_visible_sperm, 2)} sperm in view at once ` +
          `(${fmt(f.chamber_depth_um, 0)} um assumed chamber depth)`,
        false
      )
    );

    const warnings = $("optics-warnings");
    warnings.innerHTML = "";
    if (!f.warnings.length) {
      const ok = el("div", "stat");
      ok.appendChild(el("div", "stat-label", "warnings"));
      ok.appendChild(el("div", "stat-value", "none"));
      warnings.appendChild(ok);
    }
    for (const text of f.warnings) {
      const item = el("div", "warn-item");
      item.appendChild(el("span", "warn-tag", "⚠ WARNING"));
      item.appendChild(el("span", null, text));
      warnings.appendChild(item);
    }
    $("optics-report").textContent = f.formatted;

    const table = $("config-table").querySelector("tbody");
    table.innerHTML = "";
    for (const [key, value] of Object.entries(config.summary)) {
      const row = el("tr");
      row.appendChild(el("th", null, key));
      row.appendChild(el("td", null, String(value)));
      table.appendChild(row);
    }
    for (const [key, value] of Object.entries(config.decision)) {
      if (typeof value === "object") continue;
      const row = el("tr");
      row.appendChild(el("th", null, "decision." + key));
      row.appendChild(el("td", null, String(value)));
      table.appendChild(row);
    }
  }

  // ------------------------------------------------------------------ boot

  function wireControls() {
    $("reroll").addEventListener("click", () => {
      // 2^32 - 1 is the server's documented ceiling for `seed`.
      $("seed").value = String(Math.floor(Math.random() * 4294967295));
      runGenerate();
    });
    $("replay").addEventListener("click", () => {
      if (S.generated) animateTrack(S.generated.payload.trajectory);
    });
    $("classify").addEventListener("click", runClassify);
    $("force_aspects").addEventListener("change", () => {
      $("aspect-forcing").classList.toggle("hidden", !$("force_aspects").checked);
      scheduleGenerate();
    });

    const simple = [
      ["blur_px", (v) => `${fmt(v, 1)} px`],
      ["noise_sigma", (v) => fmt(v, 1)],
      ["n_points", (v) => `${v} pts`],
      ["fps", (v) => `${v} fps`],
      ["track_um_per_px", (v) => `${fmt(v, 3)} um/px`],
      ["flow_vx_px_s", (v) => `${v} px/s`],
      ["flow_vy_px_s", (v) => `${v} px/s`],
      ["progressive_rate", (v) => fmt(v, 2)],
    ];
    for (const [id, formatter] of simple) {
      const input = $(id);
      const out = $(id + "_out");
      const update = () => {
        out.textContent = formatter(Number(input.value));
      };
      update();
      input.addEventListener("input", () => {
        update();
        scheduleGenerate();
      });
    }
    for (const id of ["seed", "motility", "image_size"]) {
      $(id).addEventListener("change", scheduleGenerate);
    }
    for (const aspect of S.meta.aspects) {
      $("prev_" + aspect).addEventListener("input", scheduleGenerate);
    }

    for (const id of ["trackable", "eligible"]) {
      $(id).addEventListener("input", () => {
        syncDecisionInputs();
        scheduleDecision();
      });
    }

    window.addEventListener(
      "resize",
      debounce(() => {
        if (S.generated && S.generated.decodedImage) drawCrop(S.generated.decodedImage);
        if (S.anim.points) drawTrack(Math.floor(S.anim.index));
      }, 120)
    );
  }

  const scheduleDecision = debounce(() => {
    runDecision(false);
  }, 120);

  async function boot() {
    try {
      const [meta, config] = await Promise.all([request("/aspects"), request("/config")]);
      S.meta = meta;
      S.config = config;
    } catch (error) {
      showError("could not reach the API: " + String(error.message || error));
      return;
    }
    applyModelProvenance(S.config.morphology);
    buildMotilitySelect();
    buildPrevalenceSliders();
    buildAspectToggles();
    buildKnobSliders();
    buildPresetCases();
    renderOptics(S.config);
    wireControls();

    // Start on the boundary case, because it is the one people get wrong.
    const boundary = S.meta.mandated_decision_cases[0];
    $("trackable").value = String(boundary.trackable_count);
    $("eligible").value = String(boundary.ai_eligible_count);
    syncDecisionInputs();

    await runGenerate();
    await runDecision(true);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
