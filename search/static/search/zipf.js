'use strict';
(() => {
  const panel = document.querySelector('[data-zipf-chart]');
  const source = document.getElementById('zipf-data');
  if (!panel || !source) return;
  const data = JSON.parse(source.textContent);
  const keys = panel.dataset.condition === 'all' ? ['A','B','C','D'] : [panel.dataset.condition];
  const sets = keys.map(key => data[key]).filter(Boolean);
  const svg = document.getElementById('zipf-chart');
  const tip = panel.querySelector('.chart-tooltip');
  const ns = 'http://www.w3.org/2000/svg';
  // Okabe-Ito-inspired, color-blind-friendly hues. Domain A/B and condition
  // comparisons must remain distinguishable without relying on blue shades.
  const colors = {A:'#0072B2',B:'#D55E00',C:'#009E73',D:'#CC79A7'};
  let L=74, T=34, W=946, H=294, canvasW=1050, canvasH=390;
  const parameters = new URL(location.href).searchParams;
  let mode=['log','frequency','residual'].includes(parameters.get('chart')) ? parameters.get('chart') : 'log', coords=[], hoverDot=null;
  panel.querySelector('[data-show-fit]').checked = parameters.get('fit') !== '0';
  function rememberControls() {
    const url = new URL(location.href);
    url.searchParams.set('chart', mode);
    url.searchParams.set('fit', panel.querySelector('[data-show-fit]').checked ? '1' : '0');
    history.replaceState(null, '', url);
    document.querySelectorAll('.condition-bar a, .term-section .pagination a').forEach(link => {
      const target = new URL(link.href); target.searchParams.set('chart', mode); target.searchParams.set('fit', url.searchParams.get('fit')); link.href=target;
    });
    document.querySelectorAll('.term-controls').forEach(form => {
      for(const name of ['chart','fit']) {
        let input=form.querySelector(`input[name="${name}"]`);
        if(!input) {input=document.createElement('input');input.type='hidden';input.name=name;form.appendChild(input);}
        input.value=url.searchParams.get(name);
      }
    });
  }
  function add(tag, attrs={}, text=null, parent=svg) {
    const element = document.createElementNS(ns,tag);
    for (const [key,value] of Object.entries(attrs)) element.setAttribute(key,String(value));
    if (text !== null) element.textContent=text;
    parent.appendChild(element); return element;
  }
  function label(x,y,text,extra={}) {
    return add('text', {x,y,fill:'#687985','font-size':13,'font-family':'Segoe UI, Arial, sans-serif',...extra},text);
  }
  function predicted(d,rank) {
    return d.fit.slope === null ? 0 : d.fit.intercept+d.fit.slope*Math.log(rank);
  }
  function value(d,p) {
    if (mode === 'frequency') return p.cf;
    if (mode === 'residual') return Math.log(p.cf)-predicted(d,p.rank);
    return Math.log(p.cf);
  }
  function compact(n) {
    if (mode !== 'frequency') return n.toFixed(1);
    return n>=1000 ? `${Number((n/1000).toFixed(1))}k` : `${Math.round(n)}`;
  }
  function draw() {
    svg.replaceChildren(); coords=[]; tip.hidden=true;
    const narrow=window.innerWidth<=760;
    canvasW=narrow?Math.max(280,panel.querySelector('.chart-container').clientWidth):1050;
    canvasH=narrow?310:390; L=narrow?53:74; W=canvasW-L-20; H=canvasH-T-65;
    svg.setAttribute('viewBox',`0 0 ${canvasW} ${canvasH}`);
    const count = sets.reduce((n,d)=>Math.max(n,d.vocabulary),1);
    const xmax=mode==='frequency'?Math.max(count,1):Math.max(Math.log(count),1);
    const values = sets.flatMap(d=>d.points.map(p=>value(d,p)));
    let ymin=0,ymax=Math.max(1,...values);
    const fitChecked=panel.querySelector('[data-show-fit]').checked;
    if (mode==='log' && fitChecked) {
      const ends=sets.filter(d=>d.fit.slope!==null).flatMap(d=>[predicted(d,1),predicted(d,Math.max(d.vocabulary,1))]);
      ymin=Math.min(0,...ends); ymax=Math.max(ymax,...ends);
    }
    if (mode==='residual') { const absolute=Math.max(.5,...values.map(Math.abs)); ymin=-absolute; ymax=absolute; }
    const ypad=(ymax-ymin)*.06; ymax+=ypad; if (mode==='residual') ymin-=ypad;
    const x = r=> L+(mode==='frequency'?r:Math.log(r))/xmax*W;
    const y = v=> T+H-(v-ymin)/(ymax-ymin)*H;
    add('title',{},`Zipf analysis: ${keys.join(', ')}. ${mode}. ${sets[0].documents} distinct abstracts.`);
    add('desc',{},'Terms ranked by collection frequency. Regression uses all vocabulary entries; display points are sampled.');
    add('rect',{x:0,y:0,width:canvasW,height:canvasH,fill:'#ffffff'});
    const defs=add('defs'); const clip=add('clipPath',{id:'chart-clip'},null,defs);
    add('rect',{x:L,y:T,width:W,height:H},null,clip);
    const group=add('g',{'clip-path':'url(#chart-clip)'});
    if (sets.length===1 && mode!=='frequency' && sets[0].vocabulary>20) {
      const cuts=[1,Math.ceil(count*.05),Math.ceil(count*.5),count];
      for(let i=0;i<3;i++) {
        add('rect',{x:x(cuts[i]),y:T,width:Math.max(0,x(cuts[i+1])-x(cuts[i])),height:H,fill:i===1?'#f6f9fb':'#fbfdff'},null,group);
        if (x(cuts[i+1])-x(cuts[i])>95) label((x(cuts[i])+x(cuts[i+1]))/2,T+14,['HEAD','MIDDLE','TAIL'][i],{'text-anchor':'middle','font-size':11,fill:'#91a0aa'});
      }
    }
    const ticks=narrow?4:5;
    for(let i=0;i<=ticks;i++) {
      const v=ymin+i*(ymax-ymin)/ticks, yy=y(v);
      add('line',{x1:L,y1:yy,x2:L+W,y2:yy,stroke:'#edf1f4','stroke-width':1});
      label(L-13,yy+4,compact(v),{'text-anchor':'end'});
      const vx=i*xmax/ticks, xx=L+i*W/ticks;
      add('line',{x1:xx,y1:T+H,x2:xx,y2:T+H+5,stroke:'#c7d3dc'});
      label(xx,T+H+22,compact(vx),{'text-anchor':'middle'});
    }
    add('line',{x1:L,y1:T,x2:L,y2:T+H,stroke:'#c7d3dc'});
    add('line',{x1:L,y1:T+H,x2:L+W,y2:T+H,stroke:'#c7d3dc'});
    label(L+W/2,canvasH-7,mode==='frequency'?'Term rank':'ln(Term rank)',{'text-anchor':'middle',fill:'#496277','font-size':14});
    label(17,T+H/2,mode==='frequency'?'Collection frequency (CF)':mode==='residual'?'Residual in ln(CF)':'ln(Collection frequency)',{transform:`rotate(-90 17 ${T+H/2})`,'text-anchor':'middle',fill:'#496277','font-size':14});
    if (mode==='residual') add('line',{x1:L,y1:y(0),x2:L+W,y2:y(0),stroke:'#8ca8bd','stroke-dasharray':'5 4'});
    for(const d of sets) {
      const points=d.points.map(p=>({x:x(p.rank),y:y(value(d,p)),p,d})); coords.push(...points);
      add('path',{d:points.map((p,i)=>`${i?'L':'M'}${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(' '),fill:'none',stroke:colors[d.key],'stroke-width':2.5,'stroke-linejoin':'round'},null,group);
      if (mode==='log' && fitChecked && d.fit.slope!==null) {
        add('line',{x1:x(1),y1:y(predicted(d,1)),x2:x(d.vocabulary),y2:y(predicted(d,d.vocabulary)),stroke:colors[d.key],'stroke-opacity':.55,'stroke-width':1.6,'stroke-dasharray':'6 5'},null,group);
      }
    }
    label(L,18,narrow?`Condition ${keys.join(' / ')}`:sets.map(d=>`${d.key} · ${d.label}`).join('    /    '),{'font-size':10,fill:'#496277'});
    hoverDot=add('circle',{r:5,fill:'#fff',stroke:'#2a6aa4','stroke-width':2,visibility:'hidden'});
    panel.querySelector('[data-chart-caption]').textContent = mode==='frequency'?'Linear axes · full rank range':mode==='residual'?'Observed ln(CF) − fitted ln(CF) · zero means on the regression line':'Natural log of rank and CF · dashed lines show the fitted regression';
    panel.querySelector('.fit-toggle').hidden=mode!=='log';
    panel.querySelectorAll('[data-chart-mode]').forEach(b=>{const selected=b.dataset.chartMode===mode;b.classList.toggle('active',selected);b.setAttribute('aria-pressed',String(selected));});
  }
  panel.querySelectorAll('[data-chart-mode]').forEach(button=>button.addEventListener('click',()=>{mode=button.dataset.chartMode;rememberControls();draw();}));
  panel.querySelector('[data-show-fit]').addEventListener('change',()=>{rememberControls();draw();});
  svg.addEventListener('pointermove',event=>{
    const bounds=svg.getBoundingClientRect();
    const point=svg.createSVGPoint();point.x=event.clientX;point.y=event.clientY;
    const local=point.matrixTransform(svg.getScreenCTM().inverse());
    if(local.x<L || local.x>L+W || local.y<T || local.y>T+H) {tip.hidden=true;hoverDot.setAttribute('visibility','hidden');return;}
    let nearest=null,distance=Infinity;
    for(const p of coords) { const dist=(p.x-local.x)**2+(p.y-local.y)**2;if(dist<distance){distance=dist;nearest=p;} }
    if(!nearest) return;
    tip.textContent=`${nearest.d.key} · ${nearest.p.term}\nRank ${nearest.p.rank.toLocaleString()}  /  CF ${nearest.p.cf.toLocaleString()}`;
    tip.hidden=false;
    const container=panel.querySelector('.chart-container').getBoundingClientRect();
    tip.style.left=`${Math.max(0,Math.min(event.clientX-container.left+12,container.width-225))}px`;
    tip.style.top=`${Math.max(0,event.clientY-container.top-68)}px`;
    hoverDot.setAttribute('cx',nearest.x);hoverDot.setAttribute('cy',nearest.y);hoverDot.setAttribute('stroke',colors[nearest.d.key]);hoverDot.setAttribute('visibility','visible');
  });
  svg.addEventListener('pointerleave',()=>{tip.hidden=true;hoverDot.setAttribute('visibility','hidden');});
  panel.querySelector('[data-save-chart]').addEventListener('click',()=>{
    const copy=svg.cloneNode(true);copy.setAttribute('xmlns',ns);copy.setAttribute('width',String(canvasW));copy.setAttribute('height',String(canvasH));
    copy.querySelector('circle')?.remove();
    const url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(copy)],{type:'image/svg+xml'}));
    const link=document.createElement('a');link.href=url;link.download=`zipf_${keys.join('')}_${mode}.svg`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  window.addEventListener('resize',draw);
  draw();
  rememberControls();
})();
