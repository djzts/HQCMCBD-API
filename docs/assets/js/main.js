(function () {
  "use strict";

  /* ---------- theme toggle ---------- */
  var root = document.documentElement;
  var toggle = document.getElementById("themeToggle");
  function currentTheme() {
    var t = root.getAttribute("data-theme");
    if (t) return t;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("hqc-theme", next); } catch (e) {}
    });
  }

  /* ---------- math ---------- */
  function renderMath(scope) {
    if (!window.katex) return;
    (scope || document).querySelectorAll("[data-tex]").forEach(function (el) {
      katex.render(el.getAttribute("data-tex"), el, {
        displayMode: el.classList.contains("math-block"),
        throwOnError: false,
        strict: false
      });
    });
    (scope || document).querySelectorAll(".m").forEach(function (el) {
      katex.render(el.textContent, el, { displayMode: false, throwOnError: false, strict: false });
      el.classList.remove("m");
    });
  }
  renderMath();

  /* ---------- code highlighting ---------- */
  function highlight(el) {
    if (!window.hljs) return;
    delete el.dataset.highlighted;
    el.classList.remove("hljs");
    hljs.highlightElement(el);
  }
  if (window.hljs) {
    hljs.configure({ ignoreUnescapedHTML: true });
    document.querySelectorAll("pre code[class*='language-']").forEach(function (el) {
      if (el.id !== "cfgJson") highlight(el);
    });
  }

  /* ---------- copy ---------- */
  function copyText(text, btn) {
    function done(ok) {
      if (!btn) return;
      var old = btn.textContent;
      btn.textContent = ok ? "Copied" : "Press Ctrl+C";
      setTimeout(function () { btn.textContent = old; }, 1400);
    }
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(function () { done(true); }, function () { done(fallback()); });
    } else {
      done(fallback());
    }
    function fallback() {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      var ok = false;
      try { ok = document.execCommand("copy"); } catch (e) {}
      document.body.removeChild(ta);
      return ok;
    }
  }
  document.querySelectorAll(".codecard .copy").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var code = btn.parentElement.querySelector("pre code");
      if (code) copyText(code.textContent, btn);
    });
  });

  /* ---------- tabs ---------- */
  document.querySelectorAll("[data-tabs]").forEach(function (box) {
    var tabs = Array.prototype.slice.call(box.querySelectorAll('[role="tab"]'));
    function select(tab) {
      tabs.forEach(function (t) {
        var on = t === tab;
        t.setAttribute("aria-selected", on ? "true" : "false");
        t.tabIndex = on ? 0 : -1;
        document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
      });
    }
    tabs.forEach(function (tab, i) {
      tab.addEventListener("click", function () { select(tab); });
      tab.addEventListener("keydown", function (e) {
        var j = null;
        if (e.key === "ArrowRight") j = (i + 1) % tabs.length;
        if (e.key === "ArrowLeft") j = (i - 1 + tabs.length) % tabs.length;
        if (j !== null) { e.preventDefault(); tabs[j].focus(); select(tabs[j]); }
      });
    });
  });

  /* ---------- active nav link ---------- */
  var navLinks = document.querySelectorAll(".topnav__links a");
  if ("IntersectionObserver" in window && navLinks.length) {
    var byId = {};
    navLinks.forEach(function (a) { byId[a.getAttribute("href").slice(1)] = a; });
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting && byId[en.target.id]) {
          navLinks.forEach(function (a) { a.classList.remove("is-active"); });
          byId[en.target.id].classList.add("is-active");
        }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    Object.keys(byId).forEach(function (id) {
      var s = document.getElementById(id);
      if (s) io.observe(s);
    });
  }

  /* ---------- config builder ---------- */
  var form = document.getElementById("cfgForm");
  if (!form) return;

  // Values of config_default.json in HQCMCBD-API.
  var DEFAULTS = {
    nonneg: 15, decimal: 4, negative: 15,
    submethod: "l_shape", debug: false, hybrid: true,
    dmode: "cqm", token: "YOUR_DWAVE_TOKEN", mcut: true, cutnums: 2, reads: 100,
    ttype: "absolute", gap: 0.5, max_steps: 20
  };
  var PLACEHOLDER_TOKENS = ["", "example-token", "YOUR_DWAVE_TOKEN"];
  var LOCAL_MODES = ["openjij", "bifurcation"];

  var out = document.getElementById("cfgJson");
  var info = document.getElementById("lambdaInfo");
  var note = document.getElementById("cfgNote");
  var dwaveSet = document.getElementById("dwaveSet");
  var el = form.elements;

  function intVal(name, fallback) {
    var v = parseInt(el[name].value, 10);
    var min = parseInt(el[name].min, 10);
    var max = parseInt(el[name].max, 10);
    if (isNaN(v)) return fallback;
    if (!isNaN(min) && v < min) v = min;
    if (!isNaN(max) && v > max) v = max;
    return v;
  }
  function numVal(name, fallback) {
    var v = parseFloat(el[name].value);
    return isNaN(v) || v <= 0 ? fallback : v;
  }
  function fmt(n) {
    return n.toLocaleString("en-US", { maximumFractionDigits: 12 });
  }

  function read() {
    return {
      lambda_var: {
        nonneg_bits_length: intVal("nonneg", DEFAULTS.nonneg),
        decimal_bits_length: intVal("decimal", DEFAULTS.decimal),
        negative_bits_length: intVal("negative", DEFAULTS.negative)
      },
      submethod: el.submethod.value,
      debug_mode: el.debug.checked,
      Hybrid_mode: el.hybrid.checked,
      dwave: {
        mode: el.dmode.value,
        DWave_token: el.token.value.trim(),
        Mcut_flag: el.mcut.checked,
        Cutnums: intVal("cutnums", DEFAULTS.cutnums),
        num_of_read: intVal("reads", DEFAULTS.reads)
      },
      threshold: {
        type: el.ttype.value,
        gap: numVal("gap", DEFAULTS.gap)
      },
      max_steps: intVal("max_steps", DEFAULTS.max_steps)
    };
  }

  function update() {
    var cfg = read();
    out.textContent = JSON.stringify(cfg, null, 4);
    highlight(out);

    // Lambda coefficients as built in HQCMCBD.py:
    //   negative part: -2^1 ... -2^M
    //   non-negative part: 2^-D ... 2^(N-1)
    var N = cfg.lambda_var.nonneg_bits_length;
    var D = cfg.lambda_var.decimal_bits_length;
    var M = cfg.lambda_var.negative_bits_length;
    var hi = Math.pow(2, N) - Math.pow(2, -D);
    var lo = -(Math.pow(2, M + 1) - 2);
    var step = Math.pow(2, -D);
    info.innerHTML =
      "λ ranges over <b>[" + fmt(lo) + ", " + fmt(hi) + "]</b> in steps of <b>" + fmt(step) +
      "</b>, using <b>" + (N + D + M) + "</b> binary variables in the master problem.";

    dwaveSet.disabled = !cfg.Hybrid_mode;
    el.cutnums.disabled = !cfg.dwave.Mcut_flag;

    var msg = "";
    if (!cfg.Hybrid_mode) {
      msg = "Hybrid_mode is off, so Gurobi solves the master problem classically and the quantum backend settings are ignored.";
    } else if (LOCAL_MODES.indexOf(cfg.dwave.mode) !== -1) {
      msg = "“" + cfg.dwave.mode + "” runs locally as a quantum-inspired solver. No D-Wave token is needed.";
    } else if (PLACEHOLDER_TOKENS.indexOf(cfg.dwave.DWave_token) !== -1) {
      msg = "With a placeholder token, HQC-Bend reads your D-Wave Leap token from the DWAVE_API_TOKEN environment variable.";
    } else {
      msg = '<span class="warn">Don’t commit a real token.</span> Keep this file out of version control, or use the DWAVE_API_TOKEN environment variable instead.';
    }
    note.innerHTML = msg;
  }

  function load(d) {
    el.nonneg.value = d.nonneg;
    el.decimal.value = d.decimal;
    el.negative.value = d.negative;
    el.submethod.value = d.submethod;
    el.debug.checked = d.debug;
    el.hybrid.checked = d.hybrid;
    el.dmode.value = d.dmode;
    el.token.value = d.token;
    el.mcut.checked = d.mcut;
    el.cutnums.value = d.cutnums;
    el.reads.value = d.reads;
    el.ttype.value = d.ttype;
    el.gap.value = d.gap;
    el.max_steps.value = d.max_steps;
    update();
  }

  form.addEventListener("input", update);
  form.addEventListener("change", update);
  form.addEventListener("submit", function (e) { e.preventDefault(); });

  document.getElementById("cfgDefault").addEventListener("click", function () { load(DEFAULTS); });
  document.getElementById("cfgCopy").addEventListener("click", function (e) {
    copyText(out.textContent + "\n", e.currentTarget);
  });
  document.getElementById("cfgDownload").addEventListener("click", function () {
    var blob = new Blob([out.textContent + "\n"], { type: "application/json" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = "config.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  });

  load(DEFAULTS);
})();
