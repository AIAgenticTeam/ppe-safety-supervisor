/* Page actions. Every write goes to the JSON API; nothing here decides anything.
 *
 * The two that matter are identity and approval. Both are refused server-side (a name
 * not on the roster, an escalation already signed), and this script only reports what
 * the API said -- it never treats a click as the decision.
 */
(function () {
  "use strict";
  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };

  // Build DOM with textContent only. Event bodies and model output reach these pages,
  // so nothing from the API is ever assigned to innerHTML.
  function h(tag, attrs) {
    var el = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === "class") el.className = attrs[k];
      else if (k === "text") el.textContent = attrs[k];
      else el.setAttribute(k, attrs[k]);
    });
    Array.prototype.slice.call(arguments, 2).forEach(function (c) {
      if (c != null) el.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return el;
  }

  function call(method, url, body) {
    var opts = { method: method, headers: {} };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    return fetch(url, opts).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (data) {
        return { ok: r.ok, status: r.status, data: data };
      });
    }).catch(function () {
      return { ok: false, status: 0, data: { detail: "The service did not answer. Is it running?" } };
    });
  }

  function say(el, text, kind) {
    if (!el) return;
    el.textContent = text;
    el.className = "msg msg--" + kind;
  }

  // ---- evidence: a missing image is normal (fixtures ship without them) ----------
  $$("[data-evidence] img").forEach(function (img) {
    function missing() { img.closest("[data-evidence]").classList.add("is-missing"); }
    if (img.complete && img.naturalWidth === 0) missing();
    img.addEventListener("error", missing);
  });

  // ---- buttons stay disabled until a name is typed -------------------------------
  // The typed name is the signature. An empty one must not be submittable.
  $$("[data-needs-name]").forEach(function (form) {
    var input = $("input[type=text]", form);
    var btn = $("button[data-submit]", form);
    function sync() { btn.disabled = !input.value.trim(); }
    input.addEventListener("input", sync);
    sync();
  });

  // ---- identify -------------------------------------------------------------------
  $$("[data-identify]").forEach(function (card) {
    var id = card.getAttribute("data-identify");
    var btn = $("button[data-submit]", card);
    btn.addEventListener("click", function () {
      var pick = $("select", card);
      var by = $("input[type=text]", card).value.trim();
      var msg = $(".msg", card);
      btn.disabled = true;
      call("POST", "/events/" + encodeURIComponent(id) + "/identity",
           { worker_id: pick.value, bound_by: by }).then(function (r) {
        if (r.ok) {
          say(msg, "Attributed to " + pick.options[pick.selectedIndex].text + ".", "ok");
          setTimeout(function () { location.reload(); }, 600);
        } else {
          say(msg, r.data.detail || "Refused.", "err");
          btn.disabled = false;
        }
      });
    });
  });

  // ---- approvals ------------------------------------------------------------------
  $$("[data-approve]").forEach(function (card) {
    var id = card.getAttribute("data-approve");
    var btn = $("button[data-submit]", card);
    btn.addEventListener("click", function () {
      var by = $("input[type=text]", card).value.trim();
      var msg = $(".msg", card);
      btn.disabled = true;
      call("POST", "/decisions/" + encodeURIComponent(id) + "/approve",
           { approved_by: by }).then(function (r) {
        if (r.ok) {
          say(msg, "Approved.", "ok");
          setTimeout(function () { location.reload(); }, 600);
        } else {
          say(msg, r.data.detail || "Refused.", "err");
          btn.disabled = false;
        }
      });
    });
    // "Later" leaves the escalation exactly where it is: unsigned, still in the queue.
    var later = $("[data-later]", card);
    if (later) later.addEventListener("click", function () {
      card.classList.toggle("is-deferred");
      later.textContent = card.classList.contains("is-deferred") ? "Reopen" : "Later";
    });
  });

  // ---- roster ---------------------------------------------------------------------
  var roster = $("[data-roster-form]");
  if (roster) roster.addEventListener("submit", function (e) {
    e.preventDefault();
    var msg = $(".msg", roster);
    var body = {
      worker_id: roster.worker_id.value.trim(),
      name: roster.name.value.trim(),
      email: roster.email.value.trim(),
      role: roster.role.value.trim()
    };
    if (!body.worker_id || !body.name) return say(msg, "A worker id and a name are both needed.", "err");
    call("POST", "/roster", body).then(function (r) {
      if (r.ok) location.reload();
      else say(msg, r.data.detail || "Refused.", "err");
    });
  });

  // ---- replay ---------------------------------------------------------------------
  var rp = $("[data-replay]");
  if (rp) {
    var go = $("button[data-go]", rp);
    var judge = $("input[name=judge]", rp);
    var source = $("select[name=source]", rp);
    var bar = $(".progress__bar", rp);
    var rmsg = $(".msg", rp);
    function label() { $(".lbl", go).textContent = judge.checked ? "Replay and judge" : "Replay without judging"; }
    // The label says what the button will do, because a tickbox called "run the agents"
    // reads as the action and gets clicked instead of the button.
    judge.addEventListener("change", label);
    label();

    go.addEventListener("click", function () {
      var src = source.value;
      go.disabled = true;
      call("GET", "/replay/" + encodeURIComponent(src)).then(function (r) {
        var files = (r.data && r.data.files) || [];
        if (!files.length) { go.disabled = false; return say(rmsg, "Nothing to replay from that source.", "err"); }
        var done = 0, i = 0;
        rp.classList.add("is-running");
        (function next() {
          if (i >= files.length) {
            say(rmsg, done + " of " + files.length + " accepted.", done ? "ok" : "err");
            return setTimeout(function () { location.reload(); }, 900);
          }
          say(rmsg, "posting " + (i + 1) + "/" + files.length + "…", "info");
          bar.style.width = (100 * i / files.length) + "%";
          call("POST", "/replay/" + encodeURIComponent(src) + "?name=" + encodeURIComponent(files[i]) +
               "&judge=" + judge.checked).then(function (res) {
            if (res.ok) done += 1;
            i += 1;
            next();
          });
        })();
      });
    });
  }

  // ---- monitoring (drift) ---------------------------------------------------------
  var drift = $("[data-drift]");
  if (drift) {
    var days = drift.getAttribute("data-drift");
    var out = $("#drift-out");
    function tile(label, value) {
      return h("div", { class: "tile" }, h("div", { class: "tile__value", text: String(value) }),
               h("div", { class: "tile__label", text: label }));
    }
    function render(d) {
      out.textContent = "";
      if (!d.ran) {
        out.appendChild(h("div", { class: "note-box" },
          h("p", { text: d.reason || "No drift check could be run yet." }),
          h("p", { class: "note", text: "This is a refusal, not a pass. Too few findings cannot tell you the detector is healthy — only that nobody has looked." })));
        return;
      }
      out.appendChild(h("div", { class: "tiles" },
        tile("Reference findings", d.reference_rows), tile("Current findings", d.current_rows),
        tile("Features drifted", d.drifted_columns.length + " / " + d.checked_columns)));
      if (d.drifted_columns.length) {
        var chips = h("div", { class: "chips" });
        d.drifted_columns.forEach(function (c) { chips.appendChild(h("span", { class: "chip", style: "background:#E35205", text: c })); });
        out.appendChild(chips);
        if (d.drifted_columns.indexOf("person_confidence") >= 0 || d.drifted_columns.indexOf("weakest_recall") >= 0)
          out.appendChild(h("div", { class: "note-box note-box--warn" }, h("p", { text:
            "Detection confidence has moved. A camera that was repositioned, different light, or unfamiliar clothing all look like this — and all of them mean the detector is working outside what it was measured on." })));
      } else {
        out.appendChild(h("div", { class: "note-box note-box--ok" }, h("p", { text: "Nothing moved beyond its threshold." })));
      }
      out.appendChild(h("h4", { class: "label", text: "every feature checked" }));
      var tbody = h("tbody");
      (d.columns || []).forEach(function (c) {
        tbody.appendChild(h("tr", {}, h("td", { text: c.column }), h("td", { text: c.drifted ? "yes" : "no" }),
          h("td", { text: Number(c.score).toPrecision(3) }), h("td", { text: String(c.threshold) }), h("td", { text: c.method })));
      });
      out.appendChild(h("div", { class: "table-wrap" }, h("table", { class: "data-table" },
        h("thead", {}, h("tr", {}, h("th", { text: "feature" }), h("th", { text: "drifted" }), h("th", { text: "p-value" }),
          h("th", { text: "threshold" }), h("th", { text: "test" }))), tbody)));
      out.appendChild(h("p", { class: "note", text:
        "A small p-value means the two windows differ more than chance would explain. Features with nothing recorded in them are left out entirely rather than counted as unchanged." }));
    }
    function load() {
      out.textContent = "checking for drift…";
      call("GET", "/drift?days=" + encodeURIComponent(days)).then(function (r) {
        if (r.ok) render(r.data);
        else out.textContent = "The drift check failed (" + r.status + "). " + ((r.data && r.data.detail) || "");
      });
    }
    var again = $("[data-recheck]");
    if (again) again.addEventListener("click", load);
    load();
  }
})();
