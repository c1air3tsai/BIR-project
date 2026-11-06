/* =============================================================================
   reuters/static/reuters/projection.js
   Project 4「詞層級分析」的互動散布圖：種子詞與它們的 Top-5 鄰近詞（t-SNE / PCA 投影）

   操作：滾輪或 ＋／− 縮放、拖曳平移、雙擊放大、「重設」回到原始大小；
         滑鼠移到點上 → 顯示詞、所屬種子詞與 cosine，並把同一組的詞標亮；
         點圖例可隱藏 / 顯示某個主題；可切換 CBOW / Skip-Gram、t-SNE / PCA。

   「語意縮放」：放大時只拉開點與點的距離，字的大小不變，所以疊在一起的字會分開；
   每次重畫都會把放不下（互相重疊）的鄰近詞標籤藏起來，放大後空間夠了就會出現。
   資料：reuters/views.py 以 json_script 放在頁面上（來源 docs/project4_results/projection.json）
   ============================================================================= */
'use strict';
(() => {
  const root = document.querySelector('[data-projection]');
  const source = document.getElementById('rt-projection-data');
  if (!root || !source) return;
  const data = JSON.parse(source.textContent);
  const NS = 'http://www.w3.org/2000/svg';
  const COLORS = {earn: '#E69F00', acq: '#CC79A7', crude: '#D55E00', grain: '#009E73', trade: '#0072B2'};
  const svg = root.querySelector('svg');
  const tip = root.querySelector('.rt-proj-tip');
  const state = {method: 'tsne', arch: 'cbow', k: 1, tx: 0, ty: 0, hidden: new Set()};

  const el = (name, attrs = {}, text) => {
    const node = document.createElementNS(NS, name);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    return node;
  };

  // 資料座標 → 螢幕座標：先等比例縮放到畫布內（留邊），再套用使用者的縮放 k 與平移 tx / ty
  function projector(points, width, height) {
    const pad = 46;
    const xs = points.map(p => p.x), ys = points.map(p => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const scale = Math.min((width - 2 * pad) / (maxX - minX || 1), (height - 2 * pad) / (maxY - minY || 1));
    const offX = (width - (maxX - minX) * scale) / 2, offY = (height - (maxY - minY) * scale) / 2;
    return p => [
      (offX + (p.x - minX) * scale) * state.k + state.tx,
      (height - offY - (p.y - minY) * scale) * state.k + state.ty,   // y 軸向上
    ];
  }

  function render() {
    const width = svg.clientWidth || 800, height = svg.clientHeight || 560;
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.replaceChildren();
    const points = data[state.method][state.arch].filter(p => !state.hidden.has(p.topic));
    if (!points.length) return;
    const project = projector(data[state.method][state.arch], width, height);
    const placed = [];
    const overlaps = box => placed.some(b => box.x < b.x + b.w && box.x + box.w > b.x && box.y < b.y + b.h && box.y + box.h > b.y);
    const layer = el('g'), labels = el('g');
    // 種子詞先放標籤（一定顯示），鄰近詞依 cosine 由高到低，和已放的標籤重疊就先藏起來
    const order = [...points].sort((a, b) => (b.is_seed - a.is_seed) || (b.cosine - a.cosine));
    for (const p of order) {
      const [x, y] = project(p);
      p._x = x; p._y = y;
      const box = {x: x + 7, y: y - 10, w: p.word.length * (p.is_seed ? 8 : 6.6) + 4, h: 15};
      const fits = p.is_seed || !overlaps(box);
      if (fits) placed.push(box);
      const label = el('text', {x: x + 7, y: y + 4, class: p.is_seed ? 'rt-proj-seed' : 'rt-proj-word',
                                'data-seed': p.seed, 'data-fits': fits ? '1' : '0'}, p.word);
      if (!fits) label.setAttribute('display', 'none');
      labels.append(label);
    }
    for (const p of order.slice().reverse()) {
      const dot = el('circle', {cx: p._x, cy: p._y, r: p.is_seed ? 6.5 : 4.5, fill: COLORS[p.topic],
                                stroke: '#fff', 'stroke-width': 1.5, 'data-seed': p.seed});
      const hit = el('circle', {cx: p._x, cy: p._y, r: 11, fill: 'transparent', class: 'rt-proj-hit'});
      hit.addEventListener('pointerenter', event => showTip(event, p));
      hit.addEventListener('pointerleave', hideTip);
      layer.append(dot, hit);
    }
    svg.append(layer, labels);
  }

  // 標亮同一個種子詞的那一組（其他變淡），並把這組被藏起來的標籤暫時顯示
  function setFocus(seed) {
    svg.querySelectorAll('[data-seed]').forEach(node => {
      const mine = !seed || node.dataset.seed === seed;
      node.setAttribute('opacity', mine ? 1 : 0.15);
      if (node.tagName === 'text') {
        const show = node.dataset.fits === '1' || (seed && node.dataset.seed === seed);
        show ? node.removeAttribute('display') : node.setAttribute('display', 'none');
      }
    });
  }

  function showTip(event, p) {
    setFocus(p.seed);
    tip.innerHTML = '';
    const title = document.createElement('strong');
    title.textContent = p.word;
    const detail = document.createElement('span');
    detail.textContent = p.is_seed ? `種子詞 · 主題 ${p.topic}` : `「${p.seed}」的鄰近詞 · cosine ${p.cosine.toFixed(3)} · 主題 ${p.topic}`;
    tip.append(title, detail);
    const box = root.querySelector('.rt-proj-canvas').getBoundingClientRect();
    tip.style.left = `${Math.min(event.clientX - box.left + 14, box.width - 230)}px`;
    tip.style.top = `${event.clientY - box.top + 14}px`;
    tip.hidden = false;
  }

  function hideTip() {
    tip.hidden = true;
    setFocus(null);
  }

  // 以畫布上的 (cx, cy) 為中心縮放
  function zoom(factor, cx, cy) {
    const k = Math.min(12, Math.max(0.6, state.k * factor));
    const real = k / state.k;
    state.tx = cx - (cx - state.tx) * real;
    state.ty = cy - (cy - state.ty) * real;
    state.k = k;
    render();
    root.querySelector('[data-zoom-level]').textContent = `${state.k.toFixed(1)}×`;
  }
  const center = () => [svg.clientWidth / 2, svg.clientHeight / 2];

  svg.addEventListener('wheel', event => {
    event.preventDefault();
    const box = svg.getBoundingClientRect();
    zoom(event.deltaY < 0 ? 1.2 : 1 / 1.2, event.clientX - box.left, event.clientY - box.top);
  }, {passive: false});
  svg.addEventListener('dblclick', event => {
    const box = svg.getBoundingClientRect();
    zoom(1.6, event.clientX - box.left, event.clientY - box.top);
  });

  // 拖曳平移
  let drag = null;
  svg.addEventListener('pointerdown', event => {
    if (event.target.classList.contains('rt-proj-hit')) return;
    drag = {x: event.clientX, y: event.clientY, tx: state.tx, ty: state.ty};
    svg.setPointerCapture(event.pointerId);
    svg.classList.add('is-dragging');
  });
  svg.addEventListener('pointermove', event => {
    if (!drag) return;
    state.tx = drag.tx + event.clientX - drag.x;
    state.ty = drag.ty + event.clientY - drag.y;
    render();
  });
  const endDrag = () => { drag = null; svg.classList.remove('is-dragging'); };
  svg.addEventListener('pointerup', endDrag);
  svg.addEventListener('pointercancel', endDrag);

  root.querySelector('[data-zoom="in"]').addEventListener('click', () => zoom(1.4, ...center()));
  root.querySelector('[data-zoom="out"]').addEventListener('click', () => zoom(1 / 1.4, ...center()));
  root.querySelector('[data-zoom="reset"]').addEventListener('click', () => {
    Object.assign(state, {k: 1, tx: 0, ty: 0});
    zoom(1, 0, 0);
  });

  // 切換架構 / 投影方法（切換時回到原始大小）
  root.querySelectorAll('[data-set]').forEach(button => button.addEventListener('click', () => {
    const [key, value] = button.dataset.set.split(':');
    state[key] = value;
    Object.assign(state, {k: 1, tx: 0, ty: 0});
    root.querySelectorAll(`[data-set^="${key}:"]`).forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    root.querySelector('[data-proj-note]').textContent = state.method === 'tsne'
      ? 't-SNE 保留「誰跟誰很近」的局部結構；群與群之間的距離沒有意義。'
      : 'PCA 保留整體變異最大的兩個方向；群與群之間的相對位置比較可信，但局部容易擠在一起。';
    zoom(1, 0, 0);
  }));

  // 圖例：點一下隱藏 / 顯示某個主題
  root.querySelectorAll('[data-topic]').forEach(button => button.addEventListener('click', () => {
    const topic = button.dataset.topic;
    state.hidden.has(topic) ? state.hidden.delete(topic) : state.hidden.add(topic);
    button.setAttribute('aria-pressed', String(!state.hidden.has(topic)));
    render();
  }));

  window.addEventListener('resize', render);
  render();
})();
