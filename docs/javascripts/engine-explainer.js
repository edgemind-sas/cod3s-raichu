/*
 * Engine explainer player for the "How the engine works" pages.
 *
 * Replays a trace recorded by docs/figures/engine_traces.py, step by step.
 * Every value shown comes from the trace JSON block or from the page text
 * block (captions, going further panels, labels, thresholds): the script
 * holds no model logic, it only replays what the trace records.
 *
 * Markup consumed:
 *   <div class="engine-player" data-trace="<id>" data-page="<page>"></div>
 *   <script type="application/json" id="trace-<id>">...</script>
 *   <script type="application/json" id="engine-player-text-<page>">...</script>
 *
 * Vanilla ES2017, no dependency, no network access.
 */
(function () {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var FONT = 13; // rendered px: the SVGs are drawn at scale 1
  var FONT_WINNER = 13.5; // the race winner's label, emphasised (see .ee-winner-text)
  var CHAR = 7; // estimated average glyph width at FONT, kept generous
  var FALLBACK_WIDTH = 720;

  // Presentation tone of an automaton state. Each tone also has an icon
  // and the state name is always written out, so colour is never alone.
  var STATE_TONE = {
    working: "ok",
    running: "ok",
    normal: "ok",
    failed: "fault",
    alarm: "fault",
    standby: "neutral",
    full: "caution"
  };

  /* ---------- small helpers ---------- */

  function readJson(id) {
    var node = document.getElementById(id);
    if (!node) {
      return { error: "missing" };
    }
    try {
      return { value: JSON.parse(node.textContent) };
    } catch (err) {
      return { error: "unparsable" };
    }
  }

  function round2(x) {
    return Number(Number(x).toFixed(2)).toString();
  }

  function fmtTime(x) {
    if (x === null || x === undefined || isNaN(Number(x))) {
      return "none";
    }
    return round2(x) + " h";
  }

  function fmtValue(v) {
    if (typeof v === "boolean") {
      return v ? "true" : "false";
    }
    if (typeof v === "number") {
      return v.toFixed(2);
    }
    if (v === null || v === undefined) {
      return "none";
    }
    return String(v);
  }

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) {
      node.className = cls;
    }
    if (text !== undefined) {
      node.textContent = text;
    }
    return node;
  }

  function svg(tag, attrs, text) {
    var node = document.createElementNS(SVGNS, tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (key) {
        node.setAttribute(key, attrs[key]);
      });
    }
    if (text !== undefined) {
      node.textContent = text;
    }
    return node;
  }

  function componentOf(name) {
    var cut = String(name).indexOf(".");
    return cut < 0 ? String(name) : String(name).slice(0, cut);
  }

  function ownerOf(transition) {
    // "P1.mode.fail" -> automaton "P1.mode"
    var cut = String(transition).lastIndexOf(".");
    return cut < 0 ? String(transition) : String(transition).slice(0, cut);
  }

  function shortName(name) {
    var cut = String(name).indexOf(".");
    return cut < 0 ? String(name) : String(name).slice(cut + 1);
  }

  function missingBlock(id) {
    return "Engine player: the block \"" + id + "\" is missing or unreadable.";
  }

  function stepTime(step) {
    if (step.time !== undefined) {
      return step.time;
    }
    return step.to;
  }

  function textWidth(text, size) {
    return String(text).length * CHAR * ((size || FONT) / FONT);
  }

  function wrap(text, maxWidth, size) {
    var maxChars = Math.max(4, Math.floor(maxWidth / (CHAR * ((size || FONT) / FONT))));
    var words = String(text).split(" ");
    var lines = [];
    var line = "";
    words.forEach(function (word) {
      while (word.length > maxChars) {
        if (line) {
          lines.push(line);
          line = "";
        }
        lines.push(word.slice(0, maxChars));
        word = word.slice(maxChars);
      }
      var candidate = line ? line + " " + word : word;
      if (candidate.length > maxChars && line) {
        lines.push(line);
        line = word;
      } else {
        line = candidate;
      }
    });
    if (line) {
      lines.push(line);
    }
    return lines.length ? lines : [""];
  }

  function niceStep(span, count) {
    var raw = span / Math.max(1, count);
    var mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var steps = [1, 2, 5, 10];
    for (var i = 0; i < steps.length; i += 1) {
      if (steps[i] * mag >= raw) {
        return steps[i] * mag;
      }
    }
    return 10 * mag;
  }

  function fill(template, values) {
    return String(template).replace(/\{(\w+)\}/g, function (match, key) {
      return Object.prototype.hasOwnProperty.call(values, key) ? values[key] : match;
    });
  }

  function safeUrl(url) {
    return /^https?:\/\//.test(String(url)) ? String(url) : null;
  }

  /* ---------- icons (each tone has its own shape) ---------- */

  function stateIcon(tone, cx, cy) {
    var g = svg("g", { "class": "ee-icon ee-tone-" + tone, "aria-hidden": "true" });
    if (tone === "caution") {
      g.appendChild(svg("path", {
        d: "M" + cx + " " + (cy - 7) + " L" + (cx + 7) + " " + (cy + 6) +
          " L" + (cx - 7) + " " + (cy + 6) + " Z",
        "class": "ee-icon-shape"
      }));
      g.appendChild(svg("path", {
        d: "M" + cx + " " + (cy - 2) + " V" + (cy + 2) + " M" + cx + " " + (cy + 3.5) + " V" + (cy + 4),
        "class": "ee-icon-mark"
      }));
      return g;
    }
    g.appendChild(svg("circle", { cx: cx, cy: cy, r: 7, "class": "ee-icon-shape" }));
    var d;
    if (tone === "ok") {
      d = "M" + (cx - 3.5) + " " + cy + " L" + (cx - 1) + " " + (cy + 3) + " L" + (cx + 4) + " " + (cy - 3);
    } else if (tone === "fault") {
      d = "M" + (cx - 3) + " " + (cy - 3) + " L" + (cx + 3) + " " + (cy + 3) +
        " M" + (cx + 3) + " " + (cy - 3) + " L" + (cx - 3) + " " + (cy + 3);
    } else {
      d = "M" + (cx - 2) + " " + (cy - 3) + " V" + (cy + 3) + " M" + (cx + 2) + " " + (cy - 3) + " V" + (cy + 3);
    }
    g.appendChild(svg("path", { d: d, "class": "ee-icon-mark" }));
    return g;
  }

  function statusMarker(status, x, y, winner) {
    var r = winner ? 7 : 5;
    var cls = "ee-mark ee-mark-" + status + (winner ? " ee-mark-winner" : "");
    var g = svg("g", { "class": cls, "aria-hidden": "true" });
    if (winner) {
      g.appendChild(svg("circle", { cx: x, cy: y, r: r + 4, "class": "ee-mark-ring" }));
    }
    if (status === "new") {
      g.appendChild(svg("path", {
        d: "M" + x + " " + (y - r - 1) + " L" + (x + r + 1) + " " + y + " L" + x + " " + (y + r + 1) +
          " L" + (x - r - 1) + " " + y + " Z"
      }));
    } else if (status === "rescheduled") {
      g.appendChild(svg("rect", { x: x - r, y: y - r, width: 2 * r, height: 2 * r }));
    } else if (status === "dropped") {
      g.appendChild(svg("path", {
        d: "M" + (x - r) + " " + (y - r) + " L" + (x + r) + " " + (y + r) +
          " M" + (x + r) + " " + (y - r) + " L" + (x - r) + " " + (y + r),
        "class": "ee-mark-cross"
      }));
    } else {
      g.appendChild(svg("circle", { cx: x, cy: y, r: r }));
    }
    return g;
  }

  /* ---------- the player ---------- */

  function Player(root) {
    this.root = root;
    this.uid = "ee" + Math.random().toString(36).slice(2, 8);
    this.initial = root.getAttribute("data-trace") || "";
    this.page = root.getAttribute("data-page") || "";
    this.traces = {};
    this.index = 0;
    this.lastWidth = 0;
  }

  Player.prototype.loadTrace = function (id) {
    if (this.traces[id]) {
      return this.traces[id];
    }
    var read = readJson("trace-" + id);
    var trace = read.value;
    if (!trace || !Array.isArray(trace.steps) || !trace.steps.length) {
      return null;
    }
    this.traces[id] = trace;
    return trace;
  };

  Player.prototype.start = function () {
    var root = this.root;
    root.textContent = "";
    root.classList.add("ee-ready");
    var trace = this.loadTrace(this.initial);
    if (!trace) {
      var msg = el("p", "ee-error", missingBlock("trace-" + this.initial));
      msg.setAttribute("role", "alert");
      root.appendChild(msg);
      return;
    }
    this.scenario = this.initial;
    this.trace = trace;

    var textRead = readJson("engine-player-text-" + this.page);
    this.text = textRead.value && typeof textRead.value === "object" ? textRead.value : {};
    this.captions = this.text.captions || {};
    this.further = this.text.going_further || {};
    this.labels = this.text.labels || {};
    this.thresholds = Array.isArray(this.text.thresholds) ? this.text.thresholds : [];
    this.continuous = trace.steps.some(function (s) { return s.kind === "integrate"; });

    root.setAttribute("role", "region");
    root.setAttribute("aria-label", "Engine replay");
    root.setAttribute("tabindex", "0");

    this.errorLine = el("p", "ee-error");
    this.errorLine.setAttribute("role", "alert");
    this.errorLine.hidden = true;
    if (textRead.error) {
      this.errorLine.textContent = missingBlock("engine-player-text-" + this.page);
      this.errorLine.hidden = false;
    }

    this.stage = el("div", "ee-stage");
    this.diagramBox = el("div", "ee-figure ee-diagram");
    this.chartBox = el("div", "ee-figure ee-chart");
    this.timelineBox = el("div", "ee-figure ee-timeline");
    this.stage.appendChild(this.diagramBox);
    if (this.continuous) {
      this.stage.appendChild(this.chartBox);
    }
    this.stage.appendChild(this.timelineBox);

    this.caption = el("p", "ee-caption");
    this.caption.setAttribute("aria-live", "polite");

    this.branchBox = el("div", "ee-branch");
    this.branchBox.setAttribute("role", "group");
    this.branchBox.hidden = true;

    this.drill = el("details", "ee-drill");
    var drillSummary = el("summary", null, "Details of this step");
    this.drillBody = el("div", "ee-drill-body");
    this.drill.appendChild(drillSummary);
    this.drill.appendChild(this.drillBody);

    this.going = el("details", "ee-further");
    var goingSummary = el("summary", null, "Going further");
    this.goingBody = el("div", "ee-further-body");
    this.going.appendChild(goingSummary);
    this.going.appendChild(this.goingBody);

    var controls = el("div", "ee-controls");
    this.btnPrev = el("button", "ee-btn", "Previous");
    this.btnNext = el("button", "ee-btn ee-btn-primary", "Next");
    this.btnEvent = el("button", "ee-btn", "Next event");
    this.btnRestart = el("button", "ee-btn", "Restart");
    [this.btnPrev, this.btnNext, this.btnEvent, this.btnRestart].forEach(function (b) {
      b.type = "button";
      controls.appendChild(b);
    });
    this.counter = el("span", "ee-counter");
    this.endLabel = el("span", "ee-end", "End of scenario");
    var status = el("div", "ee-status");
    status.appendChild(this.counter);
    status.appendChild(this.endLabel);
    controls.appendChild(status);

    [this.errorLine, this.stage, this.caption, this.branchBox, this.drill, this.going, controls]
      .forEach(function (node) { root.appendChild(node); });

    var self = this;
    this.btnPrev.addEventListener("click", function () { self.go(self.index - 1); });
    this.btnNext.addEventListener("click", function () { self.go(self.index + 1); });
    this.btnEvent.addEventListener("click", function () { self.go(self.nextEventIndex()); });
    this.btnRestart.addEventListener("click", function () { self.restart(); });
    root.addEventListener("keydown", function (event) {
      var tag = event.target && event.target.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
        return;
      }
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        self.go(self.index - 1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        self.go(self.index + 1);
      }
    });

    if (typeof ResizeObserver === "function") {
      var pending = false;
      new ResizeObserver(function () {
        if (pending) {
          return;
        }
        pending = true;
        window.requestAnimationFrame(function () {
          pending = false;
          if (self.measure() !== self.lastWidth) {
            self.renderFigures();
          }
        });
      }).observe(this.stage);
    } else {
      window.addEventListener("resize", function () { self.renderFigures(); });
    }

    this.render(true);
  };

  Player.prototype.steps = function () {
    return this.trace.steps;
  };

  Player.prototype.label = function (name) {
    return Object.prototype.hasOwnProperty.call(this.labels, name) ? this.labels[name] : String(name);
  };

  Player.prototype.go = function (target) {
    var last = this.steps().length - 1;
    var next = Math.max(0, Math.min(last, target));
    if (next === this.index) {
      return;
    }
    this.index = next;
    this.render(true);
  };

  Player.prototype.restart = function () {
    var trace = this.loadTrace(this.initial);
    if (trace) {
      this.scenario = this.initial;
      this.trace = trace;
    }
    this.index = 0;
    this.render(true);
  };

  Player.prototype.nextEventIndex = function () {
    var steps = this.steps();
    for (var k = this.index + 1; k < steps.length; k += 1) {
      if (steps[k].kind === "branch") {
        return k; // the reader chooses before the replay goes on
      }
      if (steps[k].kind === "stable") {
        return Math.min(k + 1, steps.length - 1);
      }
    }
    return steps.length - 1;
  };

  Player.prototype.choose = function (option, branchIndex) {
    var trace = this.loadTrace(option.scenario);
    if (!trace) {
      this.errorLine.textContent = missingBlock("trace-" + option.scenario);
      this.errorLine.hidden = false;
      return;
    }
    this.scenario = option.scenario;
    this.trace = trace;
    this.index = Math.min(branchIndex + 1, trace.steps.length - 1);
    this.render(true);
  };

  // State replayed from the last stable step up to the current one.
  Player.prototype.stateAt = function (index) {
    var steps = this.steps();
    var start = 0;
    for (var k = index; k >= 0; k -= 1) {
      if (steps[k].kind === "stable") {
        start = k;
        break;
      }
    }
    var state = Object.assign({}, steps[start].state || {});
    function apply(writes) {
      (writes || []).forEach(function (w) { state[w.attribute] = w["new"]; });
    }
    for (var j = start + 1; j <= index; j += 1) {
      var s = steps[j];
      if (s.kind === "fired") {
        state[ownerOf(s.transition)] = s.to_state;
        apply(s.edge_writes);
      } else if (s.kind === "iteration") {
        apply(s.writes);
      }
    }
    return state;
  };

  Player.prototype.scheduleStep = function (index) {
    var steps = this.steps();
    for (var k = index; k >= 0; k -= 1) {
      if (steps[k].kind === "race" || steps[k].kind === "stable") {
        return steps[k];
      }
    }
    return null;
  };

  Player.prototype.measure = function () {
    var w = this.stage.clientWidth;
    return w > 0 ? Math.floor(w) : FALLBACK_WIDTH;
  };

  /* ---------- rendering ---------- */

  Player.prototype.render = function (stepChanged) {
    var steps = this.steps();
    var step = steps[this.index];
    var last = steps.length - 1;

    var hadFocus = this.root.contains(document.activeElement);
    if (stepChanged) {
      this.drill.open = false;
    }
    this.renderFigures();
    this.renderCaption(step);
    this.renderBranch(step);
    this.renderDrill(step);
    this.renderFurther(step);

    this.btnPrev.disabled = this.index === 0;
    this.btnNext.disabled = this.index === last;
    this.btnEvent.disabled = this.index === last;
    // Keep the keyboard on the player when the focused control was
    // disabled or removed (a branch choice is rebuilt on every render).
    var focused = document.activeElement;
    if (hadFocus && (!this.root.contains(focused) || focused.disabled)) {
      this.root.focus();
    }
    this.counter.textContent = "step " + (this.index + 1) + " / " + steps.length;
    this.endLabel.hidden = this.index !== last;
    this.root.classList.toggle("ee-at-end", this.index === last);
  };

  Player.prototype.renderFigures = function () {
    var width = this.measure();
    this.lastWidth = width;
    var step = this.steps()[this.index];
    var axis = this.axis(width);
    this.diagramBox.textContent = "";
    this.diagramBox.appendChild(this.drawDiagram(step, width));
    if (this.continuous) {
      this.chartBox.textContent = "";
      this.chartBox.appendChild(this.drawChart(step, axis));
    }
    this.timelineBox.textContent = "";
    this.timelineBox.appendChild(this.drawTimeline(step, axis));
  };

  Player.prototype.axis = function (width) {
    var left = 48;
    var right = 18;
    var horizon = Number(this.trace.horizon) || 1;
    var plot = Math.max(60, width - left - right);
    return {
      width: width,
      left: left,
      right: left + plot,
      horizon: horizon,
      x: function (t) {
        var c = Math.max(0, Math.min(horizon, Number(t)));
        return left + (c / horizon) * plot;
      },
      beyond: function (t) { return Number(t) > horizon; }
    };
  };

  Player.prototype.drawDiagram = function (step, width) {
    var self = this;
    var trace = this.trace;
    var state = this.stateAt(this.index);
    // Components in the order of first appearance in the trace's display
    // lists: attributes first, then automata.
    var order = [];
    (trace.attributes || []).concat(trace.automata || []).forEach(function (name) {
      var c = componentOf(name);
      if (order.indexOf(c) < 0) {
        order.push(c);
      }
    });

    var evaluating = step.kind === "iteration" ? componentOf(step["function"]) : null;
    var firing = step.kind === "fired" ? componentOf(step.transition) : null;
    var firedAutomaton = step.kind === "fired" ? ownerOf(step.transition) : null;
    var written = {};
    if (step.kind === "iteration") {
      (step.writes || []).forEach(function (w) { written[w.attribute] = true; });
    }

    var gutter = width < 520 ? 44 : 80;
    var boxW = Math.min(250, Math.floor((width - gutter - 4) / (order.length > 1 ? 2 : 1)));
    var pad = 10;
    var lineH = 18;
    var inner = boxW - 2 * pad;

    function layoutBox(comp) {
      var rows = [];
      var title = wrap(self.label(comp), inner - (comp === evaluating || comp === firing ? textWidth("evaluating", 12) + 8 : 0));
      rows.push({ type: "title", lines: title });
      (trace.automata || []).forEach(function (name) {
        if (componentOf(name) !== comp) {
          return;
        }
        var value = state[name];
        var text;
        var tone;
        if (name === firedAutomaton) {
          text = self.label(shortNameIfUnlabelled(name)) + ": " + step.from_state + " -> " + step.to_state;
          tone = STATE_TONE[step.to_state] || "neutral";
        } else {
          text = self.label(shortNameIfUnlabelled(name)) + ": " + fmtValue(value);
          tone = STATE_TONE[value] || "neutral";
        }
        rows.push({ type: "state", name: name, tone: tone, lines: wrap(text, inner - 22),
          fired: name === firedAutomaton });
      });
      (trace.attributes || []).forEach(function (name) {
        if (componentOf(name) !== comp) {
          return;
        }
        var value = state[name];
        var text = self.label(shortNameIfUnlabelled(name)) + ": " + fmtValue(value);
        rows.push({ type: "attr", name: name, value: value, lines: wrap(text, inner - 18),
          highlight: !!written[name] });
      });
      var h = pad;
      rows.forEach(function (r) {
        r.y = h;
        r.h = r.lines.length * lineH + (r.type === "title" ? 6 : 4);
        h += r.h;
      });
      return { comp: comp, rows: rows, h: h + pad - 4 };
    }

    function shortNameIfUnlabelled(name) {
      return Object.prototype.hasOwnProperty.call(self.labels, name) ? name : shortName(name);
    }

    var boxes = order.map(layoutBox);
    var leftBoxes = boxes.length > 1 ? boxes.slice(0, -1) : boxes;
    var rightBox = boxes.length > 1 ? boxes[boxes.length - 1] : null;
    var vGap = 44;
    var y = 4;
    leftBoxes.forEach(function (b) {
      b.x = 2;
      b.y = y;
      y += b.h + vGap;
    });
    var leftH = y - vGap;
    var height = leftH + 4;
    if (rightBox) {
      rightBox.x = width - boxW - 2;
      rightBox.y = Math.max(4, Math.round((leftH - rightBox.h) / 2) + 2);
      height = Math.max(height, rightBox.y + rightBox.h + 4);
    }

    var root = svg("svg", {
      viewBox: "0 0 " + width + " " + height,
      width: width,
      height: height,
      "class": "ee-svg",
      role: "img",
      "aria-label": "Components and their current state"
    });
    var defs = svg("defs");
    var marker = svg("marker", {
      id: this.uid + "-arrow", viewBox: "0 0 10 10", refX: 9, refY: 5,
      markerWidth: 8, markerHeight: 8, orient: "auto-start-reverse"
    });
    marker.appendChild(svg("path", { d: "M0 0 L10 5 L0 10 Z", "class": "ee-arrowhead" }));
    defs.appendChild(marker);
    root.appendChild(defs);

    // Flows from every left component to the right one.
    if (rightBox) {
      leftBoxes.forEach(function (b, i) {
        var y1 = b.y + b.h / 2;
        var y2 = rightBox.y + rightBox.h * (i + 1) / (leftBoxes.length + 1);
        var x1 = b.x + boxW;
        var x2 = rightBox.x - 2;
        var mid = (x1 + x2) / 2;
        root.appendChild(svg("path", {
          d: "M" + x1 + " " + y1 + " C" + mid + " " + y1 + " " + mid + " " + y2 + " " + x2 + " " + y2,
          "class": "ee-flow",
          "marker-end": "url(#" + this.uid + "-arrow)"
        }));
      }, this);
    }
    // Dashed signal between the first two left components.
    if (leftBoxes.length >= 2) {
      var a = leftBoxes[0];
      var b2 = leftBoxes[1];
      var sx = a.x + 28;
      root.appendChild(svg("path", {
        d: "M" + sx + " " + (a.y + a.h) + " V" + (b2.y - 2),
        "class": "ee-signal",
        "marker-end": "url(#" + this.uid + "-arrow)"
      }));
      root.appendChild(svg("text", {
        x: sx + 10, y: (a.y + a.h + b2.y) / 2 + 4, "class": "ee-text ee-signal-label"
      }, "solicits"));
    }

    boxes.forEach(function (b) {
      var g = svg("g", { "class": "ee-box" });
      var cls = "ee-box-frame";
      if (b.comp === evaluating) {
        cls += " ee-evaluating";
      }
      if (b.comp === firing) {
        cls += " ee-firing";
      }
      g.appendChild(svg("rect", { x: b.x, y: b.y, width: boxW, height: b.h, rx: 6, "class": cls }));
      b.rows.forEach(function (r) {
        var ty = b.y + r.y + 13;
        if (r.type === "title") {
          r.lines.forEach(function (line, i) {
            g.appendChild(svg("text", { x: b.x + pad, y: ty + i * lineH, "class": "ee-text ee-box-title" }, line));
          });
          if (b.comp === evaluating || b.comp === firing) {
            g.appendChild(svg("text", {
              x: b.x + boxW - pad, y: ty, "text-anchor": "end",
              "class": "ee-text ee-tag"
            }, b.comp === evaluating ? "evaluating" : "fired"));
          }
          return;
        }
        if (r.highlight || r.fired) {
          g.appendChild(svg("rect", {
            x: b.x + 4, y: b.y + r.y - 2, width: boxW - 8, height: r.h, rx: 4,
            "class": "ee-row-highlight"
          }));
        }
        if (r.type === "state") {
          g.appendChild(stateIcon(r.tone, b.x + pad + 7, ty - 4));
        } else if (typeof r.value === "boolean") {
          g.appendChild(svg("circle", {
            cx: b.x + pad + 5, cy: ty - 4, r: 5,
            "class": r.value ? "ee-bool ee-bool-true" : "ee-bool ee-bool-false"
          }));
        } else {
          g.appendChild(svg("rect", {
            x: b.x + pad + 1, y: ty - 8, width: 8, height: 8, "class": "ee-num-mark"
          }));
        }
        var textCls = "ee-text" + (r.highlight || r.fired ? " ee-strong" : "") +
          (r.type === "state" ? " ee-state-text ee-tone-text-" + r.tone : "");
        var tx = b.x + pad + (r.type === "state" ? 22 : 18);
        r.lines.forEach(function (line, i) {
          g.appendChild(svg("text", { x: tx, y: ty + i * lineH, "class": textCls }, line));
        });
      });
      root.appendChild(g);
    });
    return root;
  };

  Player.prototype.drawAxisGrid = function (group, axis, top, bottom, labels) {
    var stepT = niceStep(axis.horizon, Math.max(2, Math.floor((axis.right - axis.left) / 70)));
    for (var t = 0; t <= axis.horizon + 1e-9; t += stepT) {
      var x = axis.x(t);
      group.appendChild(svg("line", { x1: x, x2: x, y1: top, y2: bottom, "class": "ee-grid" }));
      if (labels) {
        group.appendChild(svg("text", {
          x: x, y: bottom + 16, "text-anchor": "middle", "class": "ee-text ee-tick"
        }, round2(t) + (t + stepT > axis.horizon + 1e-9 ? " h" : "")));
      }
    }
  };

  Player.prototype.cursor = function (group, axis, time, top, bottom, withLabel) {
    var x = axis.x(time);
    group.appendChild(svg("line", { x1: x, x2: x, y1: top, y2: bottom, "class": "ee-cursor" }));
    if (withLabel) {
      var text = "t = " + fmtTime(time);
      var anchor = x + textWidth(text) + 6 > axis.width ? "end" : "start";
      group.appendChild(svg("text", {
        x: anchor === "end" ? x - 4 : x + 4, y: top + 12, "text-anchor": anchor,
        "class": "ee-text ee-cursor-label"
      }, text));
    }
  };

  Player.prototype.drawTimeline = function (step, axis) {
    var self = this;
    var sched = this.scheduleStep(this.index);
    var entries = [];
    var winner = null;
    if (sched && sched.kind === "race") {
      winner = sched.winner;
      (sched.armed || []).forEach(function (e) {
        entries.push({ transition: e.transition, date: e.date, status: "armed" });
      });
    } else if (sched) {
      (sched.schedule || []).forEach(function (e) {
        entries.push({ transition: e.transition, date: e.date, status: e.status || "kept" });
      });
    }

    var rowH = 22;
    var top = 22;
    var rows = [];
    entries.forEach(function (e) {
      var isWinner = winner !== null && e.transition === winner;
      var parts = [self.label(e.transition)];
      if (e.date !== null && e.date !== undefined) {
        parts.push(fmtTime(e.date) + (axis.beyond(e.date) ? " (after horizon)" : ""));
      }
      parts.push(isWinner ? "winner" : String(e.status));
      var text = parts[0] + "  " + parts.slice(1).join(", ");
      var hasDate = e.date !== null && e.date !== undefined;
      var mx = hasDate ? axis.x(e.date) : axis.x(stepTime(step));
      var w = textWidth(text, isWinner ? FONT_WINNER : FONT);
      var place;
      if (mx + 12 + w <= axis.width - 2) {
        place = { x: mx + 12, anchor: "start", below: false };
      } else if (mx - 12 - w >= 2) {
        place = { x: mx - 12, anchor: "end", below: false };
      } else {
        place = { x: 4, anchor: "start", below: true };
      }
      rows.push({ e: e, winner: isWinner, text: text, mx: mx, hasDate: hasDate, place: place,
        lines: place.below ? wrap(text, axis.width - 8) : [text] });
    });
    var y = top + 14;
    rows.forEach(function (r) {
      r.y = y;
      y += rowH + (r.place.below ? r.lines.length * 17 : 0);
    });
    if (winner !== null && !rows.some(function (r) { return r.winner; })) {
      // The winner is not among the armed dates (a threshold crossing).
      var wText = self.label(winner) + "  winner";
      rows.push({ special: true, text: wText, y: y });
      y += rowH;
    }
    var axisY = y + 4;
    var height = axisY + 26;

    var root = svg("svg", {
      viewBox: "0 0 " + axis.width + " " + height,
      width: axis.width,
      height: height,
      "class": "ee-svg",
      role: "img",
      "aria-label": "Timeline of the scheduled transitions"
    });
    var g = svg("g");
    this.drawAxisGrid(g, axis, top, axisY, true);
    g.appendChild(svg("line", { x1: axis.left, x2: axis.right, y1: axisY, y2: axisY, "class": "ee-axis" }));
    // The cursor goes under the rows so that their labels stay readable.
    this.cursor(g, axis, stepTime(step), 0, axisY, !this.continuous);
    rows.forEach(function (r) {
      if (r.special) {
        g.appendChild(svg("text", { x: axis.left, y: r.y + 4, "class": "ee-text ee-winner-text" }, r.text));
        return;
      }
      g.appendChild(svg("line", {
        x1: axis.left, x2: axis.right, y1: r.y, y2: r.y, "class": "ee-row-line"
      }));
      g.appendChild(statusMarker(r.e.status, r.mx, r.y, r.winner));
      if (r.hasDate && axis.beyond(r.e.date)) {
        g.appendChild(svg("path", {
          d: "M" + (r.mx + 8) + " " + (r.y - 4) + " L" + (r.mx + 13) + " " + r.y + " L" + (r.mx + 8) + " " + (r.y + 4),
          "class": "ee-beyond"
        }));
      }
      var cls = "ee-text" + (r.winner ? " ee-winner-text" : "") +
        (r.e.status === "dropped" ? " ee-dropped-text" : "");
      r.lines.forEach(function (line, i) {
        g.appendChild(svg("text", {
          x: r.place.x, y: r.place.below ? r.y + 20 + i * 17 : r.y + 4,
          "text-anchor": r.place.anchor, "class": cls
        }, line));
      });
    });
    root.appendChild(g);
    return root;
  };

  Player.prototype.drawChart = function (step, axis) {
    var steps = this.steps();
    var all = [];
    var shown = [];
    var self = this;
    steps.forEach(function (s, k) {
      if (s.kind !== "integrate") {
        return;
      }
      (s.samples || []).forEach(function (p) {
        all.push(p);
        if (k <= self.index) {
          shown.push(p);
        }
      });
    });
    var values = all.map(function (p) { return Number(p[1]); });
    this.thresholds.forEach(function (th) { values.push(Number(th.value)); });
    var lo = values.length ? Math.min.apply(null, values) : 0;
    var hi = values.length ? Math.max.apply(null, values) : 1;
    if (hi === lo) {
      hi = lo + 1;
    }
    var span = hi - lo;
    var stepV = niceStep(span, 4);
    lo = Math.floor((lo - span * 0.05) / stepV) * stepV;
    hi = Math.ceil((hi + span * 0.05) / stepV) * stepV;

    var top = 20;
    var plotH = 170;
    var bottom = top + plotH;
    var height = bottom + 8;
    function yOf(v) {
      return bottom - ((Number(v) - lo) / (hi - lo)) * plotH;
    }

    var root = svg("svg", {
      viewBox: "0 0 " + axis.width + " " + height,
      width: axis.width,
      height: height,
      "class": "ee-svg",
      role: "img",
      "aria-label": "Level over time"
    });
    var g = svg("g");
    this.drawAxisGrid(g, axis, top, bottom, false);
    for (var v = lo; v <= hi + 1e-9; v += stepV) {
      var yy = yOf(v);
      g.appendChild(svg("line", { x1: axis.left, x2: axis.right, y1: yy, y2: yy, "class": "ee-grid" }));
      g.appendChild(svg("text", {
        x: axis.left - 6, y: yy + 4, "text-anchor": "end", "class": "ee-text ee-tick"
      }, round2(v)));
    }
    g.appendChild(svg("line", { x1: axis.left, x2: axis.left, y1: top, y2: bottom, "class": "ee-axis" }));
    g.appendChild(svg("line", { x1: axis.left, x2: axis.right, y1: bottom, y2: bottom, "class": "ee-axis" }));

    this.thresholds.forEach(function (th) {
      var ty = yOf(th.value);
      g.appendChild(svg("line", { x1: axis.left, x2: axis.right, y1: ty, y2: ty, "class": "ee-threshold" }));
      g.appendChild(svg("text", {
        x: axis.right - 2, y: ty - 4, "text-anchor": "end", "class": "ee-text ee-threshold-label"
      }, String(th.label !== undefined ? th.label : "") + " " + round2(th.value)));
    });

    if (shown.length) {
      var d = shown.map(function (p, i) {
        return (i ? "L" : "M") + axis.x(p[0]).toFixed(1) + " " + yOf(p[1]).toFixed(1);
      }).join(" ");
      g.appendChild(svg("path", { d: d, "class": "ee-level" }));
      var lastP = shown[shown.length - 1];
      g.appendChild(svg("circle", { cx: axis.x(lastP[0]), cy: yOf(lastP[1]), r: 4, "class": "ee-level-dot" }));
    }

    function vmark(t, label, cls, labelY) {
      var x = axis.x(t);
      g.appendChild(svg("line", { x1: x, x2: x, y1: top, y2: bottom, "class": cls }));
      var anchor = x > (axis.left + axis.right) / 2 ? "end" : "start";
      g.appendChild(svg("text", {
        x: anchor === "end" ? x - 4 : x + 4, y: labelY, "text-anchor": anchor,
        "class": "ee-text ee-vmark-label " + cls + "-label"
      }, label));
    }

    if (step.kind === "integrate") {
      if (step.next_date !== null && step.next_date !== undefined) {
        vmark(step.next_date, "next clock date " + fmtTime(step.next_date) +
          (axis.beyond(step.next_date) ? " (after horizon)" : ""), "ee-next-date", top - 6);
      }
      if (step.stopped_at_threshold) {
        vmark(step.to, "threshold crossed", "ee-crossed", bottom - 8);
      }
    }
    this.cursor(g, axis, stepTime(step), top, bottom, false);
    root.appendChild(g);
    return root;
  };

  Player.prototype.renderCaption = function (step) {
    var values = { time: fmtTime(stepTime(step)) };
    var key = step.kind;
    if (step.kind === "race") {
      values.winner = this.label(step.winner);
      values.runners = String((step.armed || []).length);
      (step.armed || []).forEach(function (e) {
        if (e.transition === step.winner) {
          values.date = fmtTime(e.date);
        }
      });
    } else if (step.kind === "fired") {
      values.transition = this.label(step.transition);
      values.from = String(step.from_state);
      values.to = String(step.to_state);
    } else if (step.kind === "iteration") {
      values.index = String(step.index);
      values["function"] = this.label(step["function"]);
      values.count = String((step.writes || []).length);
    } else if (step.kind === "stable") {
      values.armed = String((step.schedule || []).filter(function (e) {
        return e.status !== "dropped";
      }).length);
    } else if (step.kind === "integrate") {
      values.from = fmtTime(step.from);
      values.to = fmtTime(step.to);
      values.next_date = fmtTime(step.next_date);
      if (step.stopped_at_threshold) {
        key = "integrate_threshold";
      }
    } else if (step.kind === "branch") {
      values.transition = this.label(step.transition);
    }
    var template = this.captions[key];
    this.caption.textContent = template !== undefined ? fill(template, values) : "";
  };

  Player.prototype.renderBranch = function (step) {
    var self = this;
    this.branchBox.textContent = "";
    if (step.kind !== "branch" || !Array.isArray(step.options) || !step.options.length) {
      this.branchBox.hidden = true;
      return;
    }
    this.branchBox.hidden = false;
    this.branchBox.setAttribute("aria-label", "Outcome of " + this.label(step.transition));
    var branchIndex = this.index;
    step.options.forEach(function (option) {
      var b = el("button", "ee-btn ee-choice", String(option.label));
      b.type = "button";
      var chosen = option.scenario === self.scenario;
      b.setAttribute("aria-pressed", chosen ? "true" : "false");
      b.addEventListener("click", function () { self.choose(option, branchIndex); });
      self.branchBox.appendChild(b);
    });
  };

  Player.prototype.renderDrill = function (step) {
    var self = this;
    var writes = null;
    var lead = null;
    if (step.kind === "iteration") {
      writes = step.writes || [];
      lead = "Function: " + this.label(step["function"]) + " (iteration of the fixpoint " + step.index + ")";
    } else if (step.kind === "fired" && (step.edge_writes || []).length) {
      writes = step.edge_writes;
      lead = "Written when " + this.label(step.transition) + " fired";
    }
    this.drillBody.textContent = "";
    if (writes === null) {
      this.drill.hidden = true;
      return;
    }
    this.drill.hidden = false;
    this.drillBody.appendChild(el("p", "ee-drill-lead", lead));
    if (!writes.length) {
      this.drillBody.appendChild(el("p", null, "No attribute changed."));
      return;
    }
    var list = el("ul", "ee-writes");
    writes.forEach(function (w) {
      var item = el("li");
      item.appendChild(el("code", null, self.label(w.attribute)));
      item.appendChild(document.createTextNode(": " + fmtValue(w.old) + " -> " + fmtValue(w["new"])));
      list.appendChild(item);
    });
    this.drillBody.appendChild(list);
  };

  Player.prototype.renderFurther = function (step) {
    var key = step.kind === "integrate" && step.stopped_at_threshold &&
      this.further.integrate_threshold ? "integrate_threshold" : step.kind;
    var entry = this.further[key];
    if (this.goingKey !== key) {
      this.going.open = false;
      this.goingKey = key;
    }
    this.goingBody.textContent = "";
    if (!entry) {
      this.going.hidden = true;
      return;
    }
    this.going.hidden = false;
    if (entry.rule) {
      var rule = el("p", "ee-rule");
      rule.appendChild(document.createTextNode("Rule: "));
      rule.appendChild(el("code", null, String(entry.rule)));
      this.goingBody.appendChild(rule);
    }
    if (entry.text) {
      this.goingBody.appendChild(el("p", null, String(entry.text)));
    }
    var url = safeUrl(entry.code);
    if (url) {
      var p = el("p");
      var a = el("a", null, "Read the code");
      a.href = url;
      a.rel = "noopener";
      p.appendChild(a);
      this.goingBody.appendChild(p);
    }
  };

  function boot() {
    var nodes = document.querySelectorAll(".engine-player");
    Array.prototype.forEach.call(nodes, function (node) {
      if (node.getAttribute("data-ee-started")) {
        return;
      }
      node.setAttribute("data-ee-started", "1");
      try {
        new Player(node).start();
      } catch (err) {
        node.textContent = "";
        var msg = el("p", "ee-error", "Engine player: the replay could not start.");
        msg.setAttribute("role", "alert");
        node.appendChild(msg);
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
