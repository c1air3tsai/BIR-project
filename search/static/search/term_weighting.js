'use strict';
(() => {
  const root = document.querySelector('.tw-page');
  if (!root) return;
  const $ = selector => root.querySelector(selector);
  const scope = () => $('#tw-corpus').value;
  const t = key => window.biomedI18n?.t('tw.' + key) || key;
  const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const fmt = (number, digits = 4) => number == null ? '—' : Number(number).toFixed(digits);
  const methodName = {tf:'Raw TF', log_tf:'Log TF', idf:'IDF', tfidf:'TF-IDF'};
  const methods = ['tf','log_tf','idf','tfidf'];
  const metricNames = {p5:'P@5',p10:'P@10',r10:'R@10',mrr:'MRR',ndcg10:'nDCG@10'};
  const state = {query:null, evaluation:null, activeQuery:null};
  const errorKeys = {'Index not ready. Build it first.':'error.index','Enter a query':'error.query',
    'Unknown weighting method':'error.method','Select a valid corpus':'error.corpus',
    'Query already exists':'error.duplicate','Enter query text':'error.query',
    'Document is outside the current judgment pool':'error.pool',
    'Choose relevant, not relevant, or unjudged':'error.judgment'};
  const errorText = message => errorKeys[message] ? t(errorKeys[message]) : message;
  async function get(url, parameters = {}) {
    const target = new URL(url, location.origin);
    target.searchParams.set('corpus', scope());
    for (const [key, value] of Object.entries(parameters)) target.searchParams.set(key, value);
    const response = await fetch(target, {cache:'no-store'});
    const data = await response.json();
    if (!response.ok) throw new Error(errorText(data.error || t('error')));
    return data;
  }
  async function post(url, values) {
    const body = new URLSearchParams({corpus:scope(), ...values});
    const response = await fetch(url, {method:'POST',body,headers:{'X-CSRFToken':root.dataset.csrf}});
    const data = await response.json();
    if (!response.ok) throw new Error(errorText(data.error || t('error')));
    return data;
  }
  function setStatus(selector, text, isError = false) {
    const node = $(selector); node.textContent = text; node.classList.toggle('tw-error', isError);
  }
  function panelError(selector, statusSelector, error) {
    $(selector).innerHTML = `<p class="tw-error">${esc(error.message)}</p>`;
    setStatus(statusSelector, t('error'), true);
  }
  async function refreshStatus() {
    const data = await get(root.dataset.statusUrl);
    $('#tw-available').textContent = data.available;
    $('#tw-indexed').textContent = data.indexed;
    $('#tw-vocabulary').textContent = data.vocabulary;
    $('#tw-index-state').textContent = data.ready ? t('ready') : t('missing');
    $('#tw-build').textContent = data.ready ? t('rebuild') : t('build');
    $('#tw-build').dataset.ready = data.ready ? '1' : '0';
    $('#tw-export').href = root.dataset.exportUrl + '?corpus=' + encodeURIComponent(scope());
    return data;
  }
  $('#tw-corpus').value = root.dataset.scope;
  $('#tw-corpus').addEventListener('change', async () => {
    state.evaluation = null; state.activeQuery = null;
    $('#tw-search-results').innerHTML = `<p class="tw-muted">${esc(t('search.prompt'))}</p>`;
    $('#tw-compare-results').innerHTML = `<p class="tw-muted">${esc(t('compare.prompt'))}</p>`;
    $('#tw-evaluation-results').innerHTML = '';
    $('#tw-query-detail').hidden = true;
    try { await refreshStatus(); } catch (error) { setStatus('#tw-search-status', error.message, true); }
  });
  $('#tw-build').addEventListener('click', async () => {
    const button = $('#tw-build'); button.disabled = true;
    $('#tw-index-state').textContent = t('building');
    try {
      const data = await post(root.dataset.buildUrl, {force:button.dataset.ready});
      await refreshStatus();
      setStatus('#tw-search-status', `${data.cache_hit ? t('cachehit') : t('built')} · ${fmt(data.seconds,3)}s`);
    } catch (error) { setStatus('#tw-search-status', error.message, true); $('#tw-index-state').textContent = t('missing'); }
    finally { button.disabled = false; }
  });
  root.querySelectorAll('[data-tab]').forEach(button => button.addEventListener('click', () => {
    root.querySelectorAll('[data-tab]').forEach(tab => {const active=tab===button;tab.classList.toggle('active',active);tab.setAttribute('aria-selected',String(active));});
    root.querySelectorAll('.tw-tab-panel').forEach(panel => {panel.hidden = panel.id !== 'tw-' + button.dataset.tab + '-panel';});
    if (button.dataset.tab === 'evaluate' && !state.evaluation) loadEvaluation();
  }));
  function highlighted(text, terms) {
    let source = String(text || '');
    const variants = [...new Set(terms.flatMap(term => [term,term.replace(/-/g,'‐'),term.replace(/-/g,'–')]))].filter(Boolean);
    if (!variants.length) return esc(source);
    const pattern = new RegExp('(' + variants.map(term => term.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|') + ')','gi');
    return source.split(pattern).map(piece => variants.some(term => piece.toLocaleLowerCase()===term.toLocaleLowerCase()) ? `<mark>${esc(piece)}</mark>` : esc(piece)).join('');
  }
  function resultsTable(rows, terms, explain = false) {
    if (!rows.length) return `<p class="tw-muted">${esc(t('noresults'))}</p>`;
    const hasJudgments = rows.some(row => Object.hasOwn(row, 'relevance'));
    const header = `<tr><th>${esc(t('rank'))}</th><th>${esc(t('article_title'))}</th><th class="num">${esc(t('score'))}</th><th>${esc(t('matched'))}</th><th>${esc(t('yearsource'))}</th>${hasJudgments ? `<th>${esc(t('relevant'))}</th>` : ''}${explain ? `<th>${esc(t('why'))}</th>` : ''}</tr>`;
    const body = rows.map(row => `<tr><td class="tw-rank">${row.rank}</td><td><a class="article-copy tw-result-title" href="/document/${row.id}/">${esc(row.title)}</a><small class="article-copy tw-snippet">${highlighted(row.snippet,terms)}</small></td><td class="num">${fmt(row.score)}</td><td>${row.matched.map(esc).join(' · ')}</td><td>${esc([row.year,row.source].filter(Boolean).join(' · '))}</td>${hasJudgments ? `<td>${row.relevance===null?'—':row.relevance>0?'✓':'✕'}</td>` : ''}${explain ? `<td><button type="button" class="tw-link" data-explain="${row.id}">${esc(t('why'))}</button></td>` : ''}</tr>${explain ? `<tr id="tw-explain-${row.id}" class="tw-explain-row" hidden><td colspan="${hasJudgments?7:6}"></td></tr>` : ''}`).join('');
    return `<div class="tw-table-wrap"><table class="tw-table"><thead>${header}</thead><tbody>${body}</tbody></table></div>`;
  }
  $('#tw-search-form').addEventListener('submit', async event => {
    event.preventDefault();
    const query = $('#tw-query').value.trim(), method = $('input[name=method]:checked').value;
    state.query = query; $('#tw-compare-query').value = query;
    setStatus('#tw-search-status', t('searching'));
    try {
      const data = await get(root.dataset.searchUrl,{q:query,method});
      $('#tw-search-results').innerHTML = `<h3>${esc(t('top10'))} <small>· ${data.total_matches} ${esc(t('matches'))}</small></h3>` + resultsTable(data.results,data.tokens,true);
      const oov = $('#tw-oov'); oov.hidden = !data.oov.length; oov.textContent = data.oov.length ? `${t('oov')}: ${data.oov.join(' · ')}` : '';
      setStatus('#tw-search-status', t('ready'));
    } catch (error) { panelError('#tw-search-results','#tw-search-status',error); }
  });
  $('#tw-search-results').addEventListener('click', async event => {
    const button = event.target.closest('[data-explain]'); if (!button) return;
    const row = $('#tw-explain-' + button.dataset.explain), cell = row.firstElementChild;
    if (!row.hidden) {row.hidden=true;return;}
    row.hidden=false;cell.textContent=t('loading');
    try {
      const data = await get(root.dataset.explainUrl,{q:state.query,method:$('input[name=method]:checked').value,document_id:button.dataset.explain});
      cell.innerHTML = `<div class="tw-table-wrap"><table class="tw-table tw-detail-table"><thead><tr>${['term','queryweight','documenttf','df','idf','documentweight','contribution'].map(key=>`<th class="${key==='term'?'':'num'}">${esc(t(key))}</th>`).join('')}</tr></thead><tbody>${data.terms.map(item=>`<tr><td>${esc(item.term)}</td><td class="num">${fmt(item.query_weight)}</td><td class="num">${item.tf}</td><td class="num">${item.df}</td><td class="num">${fmt(item.idf)}</td><td class="num">${fmt(item.document_weight)}</td><td class="num">${fmt(item.contribution)}</td></tr>`).join('')}</tbody></table></div><p class="tw-score-line">${esc(t('cosinescore'))} = ${fmt(data.numerator)} / (${fmt(data.query_norm)} × ${fmt(data.document_norm)}) = <strong>${fmt(data.score)}</strong></p>`;
    } catch (error) {cell.textContent=error.message;}
  });
  root.querySelectorAll('[data-query]').forEach(button => button.addEventListener('click', () => {$('#tw-query').value=button.dataset.query;$('#tw-search-form').requestSubmit();}));
  $('#tw-terms-form').addEventListener('submit', async event => {
    event.preventDefault();
    try {
      const data = await get(root.dataset.termsUrl,{terms:$('#tw-terms').value});
      $('#tw-terms-results').innerHTML = `<div class="tw-table-wrap"><table class="tw-table"><thead><tr><th>${esc(t('term'))}</th><th class="num">CF</th><th class="num">DF</th><th class="num">IDF</th></tr></thead><tbody>${data.terms.map(row=>`<tr><td>${esc(row.term)}</td><td class="num">${row.cf}</td><td class="num">${row.df}</td><td class="num">${fmt(row.idf)}</td></tr>`).join('')}</tbody></table></div>`;
    } catch (error) {$('#tw-terms-results').textContent=error.message;}
  });
  $('#tw-compare-form').addEventListener('submit', async event => {
    event.preventDefault(); setStatus('#tw-compare-status',t('comparing'));
    try {
      const data = await get(root.dataset.compareUrl,{q:$('#tw-compare-query').value.trim()});
      const summary=data.summary, l2=data.l2, evidence=data.term_evidence;
      const termEvidence = evidence ? `<div class="tw-term-evidence"><strong>${esc(t('tf.evidence'))}: <span class="article-copy">${esc(evidence.title)}</span></strong><div class="tw-table-wrap"><table class="tw-table"><thead><tr><th>${esc(t('term'))}</th><th class="num">TF</th><th class="num">${esc(t('rawweight'))}</th><th class="num">${esc(t('logweight'))}</th></tr></thead><tbody>${evidence.terms.map(row=>`<tr><td>${esc(row.term)}</td><td class="num">${row.tf}</td><td class="num">${fmt(row.raw_weight)}</td><td class="num">${fmt(row.log_weight)}</td></tr>`).join('')}</tbody></table></div></div>` : '';
      $('#tw-compare-results').innerHTML = `<div class="tw-evidence"><div><span>${esc(t('overlap'))} · Raw TF / Log TF</span><strong>${summary.overlap}/10</strong></div><div><span>Jaccard</span><strong>${fmt(summary.jaccard,2)}</strong></div><div><span>${esc(t('largestchange'))}</span><strong>${summary.largest_shared_rank_change}</strong></div><div><span>${esc(t('sharedfirst'))}</span><strong>${summary.shared_first?esc(t('yes')):esc(t('no'))}</strong></div></div>${termEvidence}<div class="tw-table-wrap"><table class="tw-table"><thead><tr><th>${esc(t('document'))}</th>${methods.map(method=>`<th class="num">${methodName[method]}</th>`).join('')}</tr></thead><tbody>${data.rows.map(row=>`<tr><td class="article-copy">${esc(row.title)}</td>${methods.map(method=>`<td class="num">${row.ranks[method]??'—'}</td>`).join('')}</tr>`).join('')}</tbody></table></div><details class="tw-l2"><summary>${esc(t('l2experiment'))}</summary><div class="tw-evidence"><div><span>${esc(t('overlap'))}</span><strong>${l2.overlap}/10</strong></div><div><span>${esc(t('maxdifference'))}</span><strong>${Number(l2.max_score_difference).toExponential(2)}</strong></div><div><span>${esc(t('sameorder'))}</span><strong>${l2.same_order?esc(t('yes')):esc(t('no'))}</strong></div></div><p class="tw-muted">${esc(t('l2equivalent'))}</p></details>`;
      setStatus('#tw-compare-status',t('ready'));
    } catch (error) {panelError('#tw-compare-results','#tw-compare-status',error);}
  });
  async function loadEvaluation() {
    setStatus('#tw-evaluate-status',t('calculating'));
    try {
      const data = await get(root.dataset.evaluationUrl);
      state.evaluation=data;
      const cells=methods.map(method=>`<tr><th>${methodName[method]}</th>${Object.keys(metricNames).map(name=>`<td class="num ${data.winners[name].includes(method)?'tw-best':''}">${fmt(data.aggregate[method][name],3)}</td>`).join('')}</tr>`).join('');
      const winners = data.ready ? `<p class="tw-winners">${['p10','mrr','ndcg10'].map(name=>`${metricNames[name]}: <strong>${data.winners[name].map(method=>methodName[method]).join(', ')}</strong>`).join(' · ')}</p>` : '';
      $('#tw-evaluation-results').innerHTML = `<div class="tw-eval-count"><strong>${data.ready}/${data.total}</strong> ${esc(data.ready===data.total&&data.total>=10?t('readyqueries'):t('incomplete'))}</div><div class="tw-table-wrap"><table class="tw-table"><thead><tr><th>${esc(t('method'))}</th>${Object.values(metricNames).map(name=>`<th class="num">${name}</th>`).join('')}</tr></thead><tbody>${cells}</tbody></table></div><p class="tw-muted">${esc(t('meanonly'))} · ${esc(t('evaluated'))}: ${data.ready}</p>${winners}<h3>${esc(t('queries'))}</h3><div class="tw-table-wrap"><table class="tw-table"><thead><tr><th>${esc(t('query'))}</th><th class="num">${esc(t('judged'))}</th><th class="num">${esc(t('relevant'))}</th><th>${esc(t('status'))}</th><th></th></tr></thead><tbody>${data.queries.map(row=>`<tr><td>${esc(row.text)}</td><td class="num">${row.judged}/${row.pool}</td><td class="num">${row.relevant}</td><td>${esc(row.ready?t('ready'):t('needsjudgment'))}</td><td><button type="button" class="tw-link" data-query-id="${row.id}">${esc(t('judge'))}</button> <button type="button" class="tw-link" data-edit-id="${row.id}">${esc(t('edit'))}</button> <button type="button" class="tw-link" data-delete-id="${row.id}">${esc(t('delete'))}</button></td></tr>`).join('')}</tbody></table></div>`;
      setStatus('#tw-evaluate-status',t('ready'));
    } catch (error) {panelError('#tw-evaluation-results','#tw-evaluate-status',error);}
  }
  $('#tw-refresh-evaluation').addEventListener('click',loadEvaluation);
  $('#tw-evaluation-results').addEventListener('click',async event=>{
    const button=event.target.closest('[data-query-id],[data-edit-id],[data-delete-id]'); if(!button)return;
    const id=Number(button.dataset.queryId||button.dataset.editId||button.dataset.deleteId);
    const query=state.evaluation.queries.find(item=>item.id===id);
    if(button.dataset.queryId){await showQuery(id);return;}
    if(button.dataset.editId){
      const text=prompt(t('editquery'),query.text);if(text===null||!text.trim())return;
      try{await post(root.dataset.queriesUrl,{action:'edit',query_id:id,text});await loadEvaluation();}catch(error){setStatus('#tw-evaluate-status',error.message,true);}return;
    }
    if(!confirm(`${t('deletequery')} “${query.text}”?`))return;
    try{await post(root.dataset.queriesUrl,{action:'delete',query_id:id});$('#tw-query-detail').hidden=true;await loadEvaluation();}catch(error){setStatus('#tw-evaluate-status',error.message,true);}
  });
  $('#tw-add-query').addEventListener('submit',async event=>{
    event.preventDefault();try{await post(root.dataset.queriesUrl,{action:'add',text:$('#tw-new-query').value});$('#tw-new-query').value='';await loadEvaluation();}catch(error){setStatus('#tw-evaluate-status',error.message,true);}
  });
  async function showQuery(id) {
    const detail=$('#tw-query-detail');detail.hidden=false;detail.textContent=t('loading');state.activeQuery=id;
    try{
      const data=await get(root.dataset.poolUrl+id+'/');
      const query=state.evaluation.queries.find(item=>item.id===id);
      detail.innerHTML=`<div class="section-heading-row"><h3>${esc(data.query.text)}</h3><span>${data.documents.filter(row=>row.relevance!==null).length}/${data.documents.length} ${esc(t('judged'))}</span></div><p class="tw-muted">${esc(t('pooled'))}</p><div class="tw-pool">${data.documents.map((row,i)=>`<div class="tw-pool-row"><div><strong>${i+1}. <span class="article-copy">${esc(row.title)}</span></strong><small class="article-copy">${esc(row.snippet)}</small><small>${esc(t('matched'))}: ${row.matched.map(esc).join(' · ')}</small></div><div class="tw-judgment" data-document="${row.id}"><button type="button" data-value="1" class="${row.relevance===1?'selected':''}">${esc(t('relevant'))}</button><button type="button" data-value="0" class="${row.relevance===0?'selected':''}">${esc(t('notrelevant'))}</button><button type="button" data-value="unjudged" class="${row.relevance===null?'selected':''}">${esc(t('unjudged'))}</button></div></div>`).join('')}</div>${query?.ready?`<h4>${esc(t('perquery'))}</h4><div class="tw-table-wrap"><table class="tw-table"><thead><tr><th>${esc(t('method'))}</th>${Object.values(metricNames).map(name=>`<th class="num">${name}</th>`).join('')}</tr></thead><tbody>${methods.map(method=>`<tr><th>${methodName[method]}</th>${Object.keys(metricNames).map(name=>`<td class="num">${fmt(query.metrics[method][name],3)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>${methods.map(method=>`<details><summary>${methodName[method]} ${esc(t('top10'))}</summary>${resultsTable(query.rankings[method],[],false)}</details>`).join('')}`:''}`;
    }catch(error){detail.textContent=error.message;}
  }
  $('#tw-query-detail').addEventListener('click',async event=>{
    const button=event.target.closest('[data-value]');if(!button)return;
    const group=button.closest('[data-document]');
    try{await post(root.dataset.judgmentUrl,{query_id:state.activeQuery,document_id:group.dataset.document,relevance:button.dataset.value});
      group.querySelectorAll('button').forEach(item=>item.classList.toggle('selected',item===button));
      const groups = [...$('#tw-query-detail').querySelectorAll('.tw-judgment')];
      const judged = groups.filter(item => !item.querySelector('[data-value="unjudged"].selected')).length;
      $('#tw-query-detail .section-heading-row span').textContent = `${judged}/${groups.length} ${t('judged')}`;
      await loadEvaluation();
    }catch(error){setStatus('#tw-evaluate-status',error.message,true);}
  });
  refreshStatus().catch(error=>setStatus('#tw-search-status',error.message,true));
})();
