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

  // A number over a label, used by the drift check and the roster import preview.
  function tile(label, value) {
    return h("div", { class: "tile" }, h("div", { class: "tile__value", text: String(value) }),
             h("div", { class: "tile__label", text: label }));
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

  // ---- roster: search -------------------------------------------------------------
  // A big import makes a long roster, so the table opens on the first 100. Searching always
  // looks across everyone; "show all" is for scrolling, not for finding.
  var table0 = $("[data-roster-table]");
  if (table0) {
    var PAGE = 100, expanded = false;
    var filter = $("[data-roster-filter]"), more = $("[data-roster-more]");
    var moreWrap = $("[data-roster-more-wrap]"), none = $("[data-roster-none]");
    var trs = $$("tbody tr", table0);
    function refresh() {
      var q = filter.value.trim().toLowerCase(), matched = 0;
      trs.forEach(function (tr) {
        var hit = !q || tr.getAttribute("data-search").indexOf(q) >= 0;
        if (hit) matched += 1;
        tr.hidden = !(hit && (q || expanded || matched <= PAGE));
      });
      none.hidden = matched > 0;
      moreWrap.hidden = !!q || expanded || matched <= PAGE;
      more.textContent = "Show all " + matched + " people";
      syncPicks(true);
    }
    filter.addEventListener("input", refresh);
    more.addEventListener("click", function () { expanded = true; refresh(); });

    // ---- roster: removal ------------------------------------------------------------
    // One dialog for one person or many, and it says what will happen to each: someone with
    // findings is hidden, not erased, because their history belongs to the record.
    var dlg = $("[data-remove-dialog]"), bar = $("[data-bulk]"), pickAll = $("[data-pick-all]");
    var pending = [];
    function pickOf(tr) { return $("[data-pick]", tr); }
    function chosen() { return trs.filter(function (tr) { return pickOf(tr).checked; }); }
    function syncPicks(dropHidden) {
      if (dropHidden) trs.forEach(function (tr) { if (tr.hidden) pickOf(tr).checked = false; });
      var n = chosen().length;
      bar.hidden = n === 0;
      $("[data-bulk-count]", bar).textContent = n + " selected";
      var shown = trs.filter(function (tr) { return !tr.hidden; });
      pickAll.checked = shown.length > 0 && shown.every(function (tr) { return pickOf(tr).checked; });
    }
    function findings(tr) { return Number(tr.getAttribute("data-findings")); }
    function name(tr) { return tr.getAttribute("data-name"); }

    function ask(rows) {
      if (!rows.length) return;
      pending = rows;
      var kept = rows.filter(function (tr) { return findings(tr) > 0; });
      var gone = rows.length - kept.length;
      var body = $("[data-rm-body]", dlg);
      body.textContent = "";
      $("[data-rm-title]", dlg).textContent = "Remove " + (rows.length === 1 ? name(rows[0]) : rows.length + " people") + " from the roster?";
      if (rows.length > 1) {
        var names = rows.slice(0, 5).map(name).join(", ");
        body.appendChild(h("p", { text: names + (rows.length > 5 ? " and " + (rows.length - 5) + " more" : "") + "." }));
      }
      // who: "They" for one person, otherwise a count ("3 people have", "1 person has")
      function who(n) { return rows.length === 1 ? "They have" : n + (n === 1 ? " person has" : " people have"); }
      if (gone) body.appendChild(h("p", { text: who(gone) + " no findings, so they will be deleted permanently." }));
      if (kept.length) {
        var total = kept.reduce(function (s, tr) { return s + findings(tr); }, 0);
        body.appendChild(h("p", { class: "dialog__note", text:
          who(kept.length) + " " + total + " finding" + (total === 1 ? "" : "s") + " on record, so they will be " +
          "hidden from the roster instead of deleted. Their findings stay attributed to them and reports keep " +
          "their name. Importing their id again brings them back." }));
      }
      $(".msg", dlg).textContent = "";
      var go = $("[data-rm-confirm]", dlg);
      go.disabled = false;
      go.textContent = rows.length === 1 ? "Remove" : "Remove " + rows.length + " people";
      dlg.showModal();
    }

    pickAll.addEventListener("change", function () {
      trs.forEach(function (tr) { if (!tr.hidden) pickOf(tr).checked = pickAll.checked; });
      syncPicks(false);
    });
    table0.addEventListener("change", function (e) { if (e.target.matches("[data-pick]")) syncPicks(false); });
    table0.addEventListener("click", function (e) {
      var btn = e.target.closest("[data-remove]");
      if (btn) ask([btn.closest("tr")]);
    });
    $("[data-bulk-remove]", bar).addEventListener("click", function () { ask(chosen()); });
    $("[data-bulk-clear]", bar).addEventListener("click", function () {
      trs.forEach(function (tr) { pickOf(tr).checked = false; });
      syncPicks(false);
    });
    $("[data-rm-cancel]", dlg).addEventListener("click", function () { dlg.close(); });
    $("[data-rm-confirm]", dlg).addEventListener("click", function () {
      var go = this, msg = $(".msg", dlg);
      go.disabled = true;
      call("POST", "/roster/remove", { worker_ids: pending.map(function (tr) { return tr.getAttribute("data-id"); }) })
        .then(function (r) {
          if (r.ok) {
            var d = r.data.deleted.length, k = r.data.deactivated.length, parts = [];
            if (d) parts.push(d + " deleted");
            if (k) parts.push(k + " hidden (they have findings)");
            say(msg, "Removed: " + parts.join(", ") + ".", "ok");
            setTimeout(function () { location.reload(); }, 800);
          } else {
            say(msg, r.data.detail || "Refused.", "err");
            go.disabled = false;
          }
        });
    });

    refresh();
  }

  // ---- roster: CSV import ---------------------------------------------------------
  // Two steps on purpose. Choosing a file only asks the server for a preview (nothing is
  // written); the second request, made by the confirm button, is the one that saves.
  var imp = $("[data-import]");
  if (imp) {
    var LIMIT = 1000000;
    var pick = $("input[type=file]", imp), zone = $("[data-dropzone]", imp);
    var result = $("[data-import-result]", imp);
    var bytes = null, fileName = "";

    function send(dry, skip) {
      var url = "/roster/import?dry_run=" + dry + (skip ? "&skip_invalid=true" : "");
      return fetch(url, { method: "POST", headers: { "Content-Type": "text/csv" }, body: bytes })
        .then(function (r) {
          return r.json().catch(function () { return {}; }).then(function (d) {
            return { ok: r.ok, status: r.status, data: d };
          });
        })
        .catch(function () {
          return { ok: false, status: 0, data: { detail: "The service did not answer. Is it running?" } };
        });
    }
    function clear() { bytes = null; pick.value = ""; result.textContent = ""; }
    function warn(text) {
      result.textContent = "";
      result.appendChild(h("div", { class: "note-box note-box--warn" }, h("p", { text: text })));
    }
    function statusChip(s) { return h("span", { class: "chip chip--" + s, text: s }); }
    function people(n) { return n + (n === 1 ? " person" : " people"); }
    function rows(n) { return n + (n === 1 ? " row" : " rows"); }

    function table(headers, body) {
      var tbody = h("tbody");
      body.forEach(function (cells) {
        var tr = h("tr");
        cells.forEach(function (c) { tr.appendChild(h("td", {}, c)); });
        tbody.appendChild(tr);
      });
      var head = h("tr");
      headers.forEach(function (x) { head.appendChild(h("th", { text: x })); });
      return h("div", { class: "table-wrap" }, h("table", { class: "data-table" }, h("thead", {}, head), tbody));
    }

    function preview(d) {
      var change = d.new + d.updated, problems = d.issue_count;
      result.textContent = "";
      result.appendChild(h("p", { class: "note", text: fileName + " · " + rows(d.rows) + " · " +
        d.delimiter + "-separated, " + d.encoding }));
      result.appendChild(h("div", { class: "tiles tiles--compact" }, tile("New", d.new), tile("Updated", d.updated),
        tile("Unchanged", d.unchanged), tile("Problems", problems)));

      if (d.ignored_columns.length) result.appendChild(h("div", { class: "note-box" },
        h("p", { text: "Ignored, not imported: " + d.ignored_columns.join(", ") + "." })));

      if (problems) {
        result.appendChild(h("h4", { class: "label", text: "rows with problems" }));
        result.appendChild(table(["row", "problem"], d.issues.slice(0, 20).map(function (i) {
          return [String(i.row), i.problem];
        })));
        if (problems > 20) result.appendChild(h("p", { class: "note", text: "…and " + (problems - 20) + " more." }));
      }

      if (d.preview.length) {
        result.appendChild(h("h4", { class: "label", text: "first rows" }));
        result.appendChild(table(["id", "name", "role", "email", ""], d.preview.map(function (r) {
          return [r.worker_id, r.name, r.role || "—", r.email || "—", statusChip(r.status)];
        })));
      }

      var actions = h("div", { class: "import-actions" });
      var msg = h("output", { class: "msg" });
      if (change > 0) {
        var label = problems ? "Import " + rows(change).replace("row", "valid row") + ", skip " + problems
                             : "Import " + people(change);
        var go = h("button", { type: "button", class: "thm-btn", text: label });
        go.addEventListener("click", function () {
          go.disabled = true;
          send(false, problems > 0).then(function (r) {
            if (r.ok) {
              say(msg, "Imported " + people(r.data.applied) + ".", "ok");
              setTimeout(function () { location.reload(); }, 900);
            } else {
              say(msg, r.data.detail || "Refused.", "err");
              go.disabled = false;
            }
          });
        });
        actions.appendChild(go);
      } else if (d.valid > 0) {
        result.appendChild(h("div", { class: "note-box note-box--ok" }, h("p", { text:
          "Nothing to change: everyone in this file is already on the roster exactly as listed." })));
      } else {
        result.appendChild(h("div", { class: "note-box note-box--warn" }, h("p", { text:
          "There are no valid rows to import. Fix the problems above and choose the file again." })));
      }
      var cancel = h("button", { type: "button", class: "thm-btn thm-btn--ghost",
                                 text: change > 0 ? "Cancel" : "Choose another file" });
      cancel.addEventListener("click", clear);
      actions.appendChild(cancel);
      result.appendChild(actions);
      result.appendChild(msg);
    }

    function load(file) {
      if (!file) return;
      if (file.size > LIMIT) return warn("That file is " + (file.size / 1e6).toFixed(1) + " MB; the limit is 1 MB.");
      fileName = file.name;
      file.arrayBuffer().then(function (buf) {
        bytes = buf;
        return send(true, false);
      }).then(function (r) {
        if (r.ok) preview(r.data);
        else warn(r.data.detail || "That file could not be read.");
      });
    }

    pick.addEventListener("change", function () { load(pick.files[0]); });
    ["dragenter", "dragover"].forEach(function (ev) {
      zone.addEventListener(ev, function (e) { e.preventDefault(); zone.classList.add("is-over"); });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      zone.addEventListener(ev, function (e) { e.preventDefault(); zone.classList.remove("is-over"); });
    });
    zone.addEventListener("drop", function (e) { load(e.dataTransfer.files[0]); });
  }

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
