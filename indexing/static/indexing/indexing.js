// =============================================================================
// indexing/static/indexing/indexing.js
// Project 5 Indexing 頁面
//   1. 點任何 docID（data-doc）→ 彈出視窗顯示那篇文件的全文，查詢詞（data-terms）用螢光筆標出來
//      內容向 /indexing/doc/<docID>/?source=...&terms=... 取得（views.document，標題與內文已在伺服器端 escape
//      並加上 <mark>）；source 是目前頁面的資料來源（dialog 的 data-source）
//      Esc、點視窗外面、按 × 或「關閉」都會關掉
//   2. 作業 6 個問題的「全部展開 / 全部收起」
// =============================================================================
(() => {
  const dialog = document.getElementById('ix-doc-dialog');
  if (dialog) {
    const field = name => dialog.querySelector(`[data-field="${name}"]`);
    let request = 0;                      // 連點時只顯示最後一次點的那篇

    const showTerms = counts => {
      const box = field('terms');
      box.replaceChildren();
      counts.forEach(({ term, count }) => {
        const item = document.createElement('span');
        const mark = document.createElement('mark');
        mark.textContent = term;
        item.append(mark, ` 出現 ${count} 次`);
        if (!count) item.classList.add('is-missing');
        box.append(item);
      });
    };

    const open = async (docId, terms) => {
      const current = ++request;
      field('meta').textContent = `Doc ${docId}`;
      field('title').textContent = '載入中…';
      field('terms').replaceChildren();
      field('body').replaceChildren();
      field('link').hidden = true;
      if (!dialog.open) dialog.showModal();
      const params = new URLSearchParams({ source: dialog.dataset.source || 'reuters', terms });
      const url = dialog.dataset.base.replace(/0\/$/, `${encodeURIComponent(docId)}/`) + '?' + params;
      try {
        const response = await fetch(url, { headers: { Accept: 'application/json' } });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const doc = await response.json();
        if (current !== request) return;
        field('meta').textContent = [`Doc ${doc.id}`, ...doc.meta].join(' · ');
        field('title').innerHTML = doc.title;
        showTerms(doc.counts);
        field('body').innerHTML = doc.paragraphs.length
          ? doc.paragraphs.map(text => `<p>${text}</p>`).join('')
          : '<p class="ix-dialog-empty">這篇新聞只有標題，沒有內文。</p>';
        field('body').scrollTop = 0;
        field('link').href = doc.url;
        field('link').textContent = `${doc.link} ↗`;
        field('link').hidden = false;
        dialog.querySelector('.ix-dialog-body mark')?.scrollIntoView({ block: 'center' });
      } catch (error) {
        if (current !== request) return;
        field('title').textContent = `讀不到 Doc ${docId}（${error.message}）`;
      }
    };

    document.addEventListener('click', event => {
      const button = event.target.closest('[data-doc]');
      if (!button) return;
      event.preventDefault();
      open(button.dataset.doc, button.dataset.terms || '');
    });
    dialog.addEventListener('click', event => {
      if (event.target === dialog || event.target.closest('[data-close]')) dialog.close();
    });
  }

  const toggle = document.querySelector('[data-toggle-answers]');
  if (toggle) {
    const items = [...toggle.closest('.ix-answers').querySelectorAll('details')];
    const label = () => { toggle.textContent = items.every(item => item.open) ? '全部收起' : '全部展開'; };
    toggle.addEventListener('click', () => {
      const openAll = !items.every(item => item.open);
      items.forEach(item => { item.open = openAll; });
      label();
    });
    items.forEach(item => item.addEventListener('toggle', label));
  }
})();
