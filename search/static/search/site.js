'use strict';
const topicForm = document.querySelector('.topic-toolbar-form');
if (topicForm) {
  // Search, year/sort and term forms use the same visible topic selection.
  for (const form of document.querySelectorAll('.search-panel, [data-auto-filter], .term-controls, .embedding-controls, .embedding-query, .matching-controls')) {
    form.addEventListener('submit', () => {
      form.querySelectorAll('input[name=topic]').forEach(input => input.remove());
      for (const box of topicForm.querySelectorAll('input[name=topic]:checked')) {
        const input = document.createElement('input');
        input.type = 'hidden'; input.name = 'topic'; input.value = box.value;
        form.appendChild(input);
      }
    });
  }
  topicForm.addEventListener('submit', () => {
    // Keep edits in the page's search or year/sort controls when applying topics.
    const pageForm = document.querySelector('.search-panel, [data-auto-filter], .term-controls, .embedding-controls, .embedding-query, .matching-controls');
    for (const name of ['q', 'year', 'sort', 'analysis_term', 'condition', 'tq', 'limit', 'tsort', 'series', 'fit', 'architecture', 'dimensions', 'window', 'min_count', 'epochs', 'word', 'report_term', 'left', 'right', 'normalization', 'case', 'suggest']) {
      const field = pageForm?.elements.namedItem(name);
      const value = field ? field.value : new URL(location.href).searchParams.get(name);
      if (value === null || value === undefined) continue;
      let hidden = topicForm.querySelector(`input[name="${name}"]`);
      if (!hidden) {
        hidden = document.createElement('input'); hidden.type = 'hidden'; hidden.name = name;
        topicForm.appendChild(hidden);
      }
      hidden.value = value;
    }
    document.getElementById('scope-loading')?.removeAttribute('hidden');
  });
  let refresh;
  const applySelection = () => {
    clearTimeout(refresh);
    // A short debounce lets users select several chips before one full update.
    refresh = setTimeout(() => topicForm.requestSubmit(), 450);
  };
  topicForm.querySelectorAll('input[name=topic]').forEach(box => box.addEventListener('change', applySelection));
  topicForm.querySelector('[data-clear-topics]')?.addEventListener('click', applySelection);
  topicForm.addEventListener('submit', () => clearTimeout(refresh));
}
for (const form of document.querySelectorAll('[data-auto-filter], .term-controls')) {
  form.querySelectorAll('select').forEach(select => select.addEventListener('change', () => form.requestSubmit()));
}
for (const filter of document.querySelectorAll('[data-topic-filter]')) {
  const boxes = Array.from(filter.querySelectorAll('input[type=checkbox]'));
  const all = filter.querySelector('[data-clear-topics]');
  function update() { all.classList.toggle('is-selected', !boxes.some(box => box.checked)); }
  all.addEventListener('click', () => { boxes.forEach(box => { box.checked = false; }); update(); });
  boxes.forEach(box => box.addEventListener('change', update));
}
for (const button of document.querySelectorAll('[data-confirm]')) {
  button.addEventListener('click', event => { if (!window.confirm(button.dataset.confirm)) event.preventDefault(); });
}
for (const button of document.querySelectorAll('[data-quantity]')) {
  button.addEventListener('click', () => { document.getElementById('id_count').value = button.dataset.quantity; });
}
for (const form of document.querySelectorAll('[data-busy-form]')) {
  form.addEventListener('submit', () => {
    const button = form.querySelector('button[type=submit]');
    if (button) { button.disabled = true; button.textContent = 'Preparing…'; }
  });
}
window.addEventListener('pageshow', event => { if (event.persisted) window.location.reload(); });
for (const panel of document.querySelectorAll('[data-import-job]')) {
  let active = true;
  async function poll() {
    if (!active) return;
    try {
      const response = await fetch(panel.dataset.statusUrl, {cache:'no-store'});
      if (!response.ok) throw new Error('Status unavailable');
      const data = await response.json();
      for (const key of ['added','imported','linked','duplicates','skipped']) {
        panel.querySelector(`[data-job-${key}]`).textContent = data[key].toLocaleString();
      }
      panel.querySelector('[data-job-percent]').textContent = `${data.progress}%`;
      panel.querySelector('[data-job-bar]').style.width = `${data.progress}%`;
      panel.querySelector('[role=progressbar]').setAttribute('aria-valuenow', data.progress);
      panel.querySelector('[data-job-message]').textContent = data.message;
      const status = panel.querySelector('[data-job-status]');
      status.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
      status.className = `status-pill ${data.status}`;
      if (!data.active) {
        active = false;
        panel.querySelector('[data-stop-form]').hidden = true;
        panel.querySelector('[data-job-finished]').hidden = false;
        // Reload explicitly via the link so completion never interrupts what is being read.
        document.querySelector('.import-form-area button[type=submit]').disabled = false;
        document.querySelector('.import-form-area button[type=submit]').textContent = 'Find & import articles';
      }
    } catch (_) { panel.querySelector('[data-job-message]').textContent = 'Reconnecting to the import… Collected articles are saved.'; }
    if (active) setTimeout(poll, 1800);
  }
  poll();
}
const dialog = document.getElementById('delete-dialog');
if (dialog) {
  const opener = document.getElementById('open-delete-dialog');
  opener?.addEventListener('click', () => { document.getElementById('cancel-delete')?.focus(); });
  document.addEventListener('keydown', event => { if (event.key === 'Escape' && !dialog.hidden) { dialog.hidden=true; opener?.focus(); } });
  dialog.addEventListener('keydown', event => {
    if (event.key !== 'Tab') return;
    const items = Array.from(dialog.querySelectorAll('button,a[href],input:not([type=hidden])'));
    const first = items[0], last = items[items.length-1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  });
}

const queryInput = document.querySelector('.main-search input[name=q]');
queryInput?.addEventListener('input', () => {
  const exact = queryInput.form.querySelector('input[name=analysis_term]');
  if (exact) exact.value='';
});
