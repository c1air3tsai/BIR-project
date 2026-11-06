/* =============================================================================
   weighting/static/weighting/section_nav.js
   Project 3 / Project 4 頁面共用：段落導覽列（.tw-steps）固定在頁首下方

   1. 量出網站頁首（.site-header，本身是 sticky）的實際高度，存成 CSS 變數 --header-h，
      導覽列就用 top: var(--header-h) 貼在頁首正下方（手機與桌面的頁首高度不同，所以不寫死）
   2. 點導覽列跳到某一段時，扣掉頁首 + 導覽列的高度，標題才不會被蓋住
   3. 捲動時自動把「目前看到的段落」標成 active
   ============================================================================= */
'use strict';
(() => {
  const nav = document.querySelector('.tw-steps');
  if (!nav) return;
  const header = document.querySelector('.site-header');
  const links = [...nav.querySelectorAll('a[href^="#"]')];
  const sections = links.map(link => document.querySelector(link.getAttribute('href'))).filter(Boolean);

  function measure() {
    const headerHeight = header ? header.offsetHeight : 0;
    document.documentElement.style.setProperty('--header-h', `${headerHeight}px`);
    document.documentElement.style.scrollPaddingTop = `${headerHeight + nav.offsetHeight + 16}px`;
  }

  function markActive() {
    const line = (header ? header.offsetHeight : 0) + nav.offsetHeight + 80;
    let current = sections[0];
    for (const section of sections) {
      if (section.getBoundingClientRect().top <= line) current = section;
    }
    links.forEach(link => {
      const active = link.getAttribute('href') === `#${current.id}`;
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current');
    });
  }

  measure();
  markActive();
  window.addEventListener('resize', () => { measure(); markActive(); });
  window.addEventListener('scroll', markActive, {passive: true});
})();
