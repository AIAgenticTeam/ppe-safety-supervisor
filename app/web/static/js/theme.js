/* Menu, sticky header and scroll-to-top.
 *
 * The template does this with jQuery and ~30 plugins. It needs none of them, so this is
 * the same behaviour in plain JS: the nav is cloned into the mobile drawer and the
 * sticky bar, and the sticky bar appears once the page has scrolled past the header.
 */
(function () {
  "use strict";
  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };

  // Clone the nav into the mobile drawer and the sticky bar, as the template does.
  var list = $(".main-menu .main-menu__list");
  var drawer = $(".mobile-nav__container");
  var sticky = $(".sticky-header__content");
  var nav = $("nav.main-menu");
  if (list && drawer) drawer.innerHTML = list.outerHTML;
  if (sticky && nav) sticky.innerHTML = nav.innerHTML;

  // Dropdowns in the drawer open with a button, since there is no hover on a phone.
  $$(".mobile-nav__container .main-menu__list .dropdown > a").forEach(function (a) {
    var btn = document.createElement("button");
    btn.setAttribute("aria-label", "Toggle submenu");
    btn.innerHTML = '<svg class="ico" viewBox="0 0 24 24" aria-hidden="true"><use href="#i-chevron"/></svg>';
    a.appendChild(btn);
    btn.addEventListener("click", function (e) {
      e.preventDefault();
      var open = a.parentElement.classList.toggle("expanded");
      btn.classList.toggle("expanded", open);
      var sub = a.parentElement.querySelector(":scope > ul");
      if (sub) sub.style.display = open ? "block" : "none";
    });
  });

  // Delegated, because the sticky bar's toggler is a clone made after page load.
  document.addEventListener("click", function (e) {
    if (e.target.closest(".mobile-nav__toggler")) {
      e.preventDefault();
      var wrapper = $(".mobile-nav__wrapper");
      wrapper.classList.toggle("expanded");
      document.body.classList.toggle("locked", wrapper.classList.contains("expanded"));
    }
    if (e.target.closest(".scroll-to-top")) {
      e.preventDefault();
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  });

  var stuck = $(".stricked-menu");
  var toTop = $(".scroll-to-top");
  var bar = $(".scroll-to-top__inner");
  function onScroll() {
    var y = window.scrollY || document.documentElement.scrollTop;
    if (stuck) stuck.classList.toggle("stricky-fixed", y > 130);
    if (toTop) toTop.classList.toggle("show", y > 500);
    if (bar) {
      var seen = (window.innerHeight + y) / document.body.scrollHeight * 100;
      bar.style.width = Math.min(seen, 100) + "%";
    }
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
})();
