/* =============================================================================
   weighting/static/weighting/section_nav.js
   Project 3 / Project 4 頁面共用：段落導覽列（.tw-steps）固定在頁首下方

   1. 量出網站頁首（.site-header，本身是 sticky）的實際高度，存成 CSS 變數 --header-h，
      導覽列就用 top: var(--header-h) 貼在頁首正下方（手機與桌面的頁首高度不同，所以不寫死）
   2. 點導覽列跳到某一段時，扣掉頁首 + 導覽列的高度，標題才不會被蓋住
   3. 捲動時自動把「目前看到的段落」標成 active
   4. 在某一段送出表單（搜尋、查鄰近詞）或點「同一頁、只改參數」的連結時，頁面會重新載入；
      載入後自動捲回原本那一段，而且讓那一段停在跟送出前一樣的螢幕位置，不會跑回頁面最上面。
      圖片、互動圖晚一點載入會把版面往下推，所以在使用者自己捲動之前，載入完成時再對齊一次
   ============================================================================= */
'use strict';
(() => {
  const nav = document.querySelector('.tw-steps');
  if (!nav) return;
  const header = document.querySelector('.site-header');
  const links = [...nav.querySelectorAll('a[href^="#"]')];
  const sections = links.map(link => document.querySelector(link.getAttribute('href'))).filter(Boolean);
  const RETURN_KEY = 'section-nav-return';

  const stickyHeight = () => (header ? header.offsetHeight : 0) + nav.offsetHeight;

  function measure() {
    const headerHeight = header ? header.offsetHeight : 0;
    document.documentElement.style.setProperty('--header-h', `${headerHeight}px`);
    document.documentElement.style.scrollPaddingTop = `${headerHeight + nav.offsetHeight + 16}px`;
  }

  function markActive() {
    const line = stickyHeight() + 80;
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

  // 記下「要回到哪一段、那一段離螢幕頂端多遠」；觸發的表單 / 連結不在那一段裡（例如評估表的查詢連結跳到搜尋段），
  // 就只記段落，載入後把那一段的標題放在導覽列下方
  function remember(hash, trigger) {
    const target = hash && document.getElementById(decodeURIComponent(hash.slice(1)));
    if (!target) return;
    const offset = target.contains(trigger) ? Math.round(target.getBoundingClientRect().top) : null;
    try { sessionStorage.setItem(RETURN_KEY, JSON.stringify({path: location.pathname, hash, offset})); } catch (_) { /* 無痕模式等情況 */ }
  }

  // 用 getAttribute 讀 method / action：表單裡有 name="method" 的欄位時，form.method 會變成那個欄位
  document.addEventListener('submit', event => {
    const form = event.target;
    const action = new URL(form.getAttribute('action') || '', location.href);
    if ((form.getAttribute('method') || 'get').toLowerCase() === 'get' && action.pathname === location.pathname && action.hash) {
      remember(action.hash, form);
    }
  });
  document.addEventListener('click', event => {
    const link = event.target.closest('a[href]');
    if (!link || event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || link.target) return;
    const url = new URL(link.href, location.href);
    if (url.pathname === location.pathname && url.search !== location.search && url.hash) remember(url.hash, link);
  });

  function restore() {
    let saved = null;
    try {
      saved = JSON.parse(sessionStorage.getItem(RETURN_KEY) || 'null');
      sessionStorage.removeItem(RETURN_KEY);
    } catch (_) { /* 讀不到就只依網址的 #段落 對齊 */ }
    const navigation = performance.getEntriesByType?.('navigation')[0];
    // 重新整理、上一頁 / 下一頁時交給瀏覽器還原原本的捲動位置
    if (navigation && navigation.type !== 'navigate') return;
    const target = location.hash && document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if (!target) return;
    const sameSpot = saved && saved.path === location.pathname && saved.hash === location.hash && saved.offset !== null;
    const offset = () => (sameSpot ? saved.offset : stickyHeight() + 16);
    const align = () => window.scrollTo({top: window.scrollY + target.getBoundingClientRect().top - offset(), behavior: 'instant'});
    align();
    let userMoved = false;
    for (const type of ['wheel', 'touchstart', 'keydown', 'mousedown']) {
      window.addEventListener(type, () => { userMoved = true; }, {once: true, passive: true});
    }
    const again = () => { if (!userMoved) { align(); markActive(); } };
    if (document.readyState === 'complete') requestAnimationFrame(again); else window.addEventListener('load', again, {once: true});
    setTimeout(again, 700);
  }

  measure();
  restore();
  markActive();
  window.addEventListener('resize', () => { measure(); markActive(); });
  window.addEventListener('scroll', markActive, {passive: true});
})();
