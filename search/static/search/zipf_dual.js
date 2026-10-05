'use strict';
(() => {
  const panel=document.querySelector('[data-zipf-chart]'),source=document.getElementById('zipf-data');
  if(!panel||!source)return;
  const data=JSON.parse(source.textContent),available=Object.keys(data).filter(key=>data[key]);
  const colors={A:'#0072B2',B:'#D55E00',C:'#009E73',D:'#CC79A7'};
  const ns='http://www.w3.org/2000/svg',parameters=new URL(location.href).searchParams;
  const requested=parameters.get('series');
  let active=requested===null?[...available]:requested==='none'?[]:requested.split(',').filter(key=>available.includes(key));
  const fitToggle=panel.querySelector('[data-show-fit]');
  fitToggle.checked=parameters.get('fit')!=='0';
  panel.querySelectorAll('[data-series]').forEach(box=>box.checked=active.includes(box.value));

  const charts=['frequency','log'].map(mode=>({
    mode,svg:panel.querySelector(`[data-chart-svg="${mode}"]`),container:panel.querySelector(`[data-chart-container="${mode}"]`),tip:panel.querySelector(`[data-chart-tooltip="${mode}"]`),
    coords:[],hover:null,L:62,T:30,W:0,H:0,canvasW:0,canvasH:350,
  }));
  function selectedSets(){return active.map(key=>data[key]).filter(Boolean);}
  function rememberControls(){
    const value=active.length?active.join(','):'none',url=new URL(location.href);
    url.searchParams.set('series',value);url.searchParams.set('fit',fitToggle.checked?'1':'0');url.searchParams.delete('chart');history.replaceState(null,'',url);
    document.querySelectorAll('.condition-bar a, .term-section .pagination a').forEach(link=>{const target=new URL(link.href);target.searchParams.set('series',value);target.searchParams.set('fit',url.searchParams.get('fit'));target.searchParams.delete('chart');link.href=target;});
    document.querySelectorAll('.term-controls').forEach(form=>{for(const name of ['series','fit']){let input=form.querySelector(`input[name="${name}"]`);if(!input){input=document.createElement('input');input.type='hidden';input.name=name;form.appendChild(input);}input.value=url.searchParams.get(name);}});
    panel.querySelectorAll('[data-series-legend]').forEach(item=>item.hidden=!active.includes(item.dataset.seriesLegend));
  }
  function draw(chart){
    const {mode,svg,container,tip}=chart,sets=selectedSets();svg.replaceChildren();chart.coords=[];tip.hidden=true;
    chart.canvasW=Math.max(420,container.clientWidth||520);chart.canvasH=350;chart.L=mode==='frequency'?66:62;chart.T=30;chart.W=chart.canvasW-chart.L-18;chart.H=chart.canvasH-chart.T-58;
    svg.setAttribute('viewBox',`0 0 ${chart.canvasW} ${chart.canvasH}`);
    const add=(tag,attrs={},text=null,parent=svg)=>{const element=document.createElementNS(ns,tag);for(const [key,value]of Object.entries(attrs))element.setAttribute(key,String(value));if(text!==null)element.textContent=text;parent.appendChild(element);return element;};
    const label=(x,y,text,extra={})=>add('text',{x,y,fill:'#687985','font-size':12,'font-family':'Segoe UI, Arial, sans-serif',...extra},text);
    add('title',{},`${mode==='frequency'?'Rank-frequency':'Log-log'} comparison: ${active.join(', ')||'no series'}.`);add('desc',{},mode==='frequency'?'Term rank and collection frequency on linear axes.':'Natural log of term rank and collection frequency; dashed lines are fitted regressions.');add('rect',{x:0,y:0,width:chart.canvasW,height:chart.canvasH,fill:'#fff'});
    if(!sets.length){label(chart.canvasW/2,chart.canvasH/2,'Select at least one series above.',{'text-anchor':'middle','font-size':14,fill:'#7a8993'});return;}
    const count=Math.max(1,...sets.map(set=>set.vocabulary)),xmax=mode==='frequency'?count:Math.max(Math.log(count),1),values=sets.flatMap(set=>set.points.map(point=>mode==='frequency'?point.cf:Math.log(point.cf)));
    let ymin=0,ymax=Math.max(1,...values);
    if(mode==='log'&&fitToggle.checked){const ends=sets.filter(set=>set.fit.slope!==null).flatMap(set=>[set.fit.intercept,set.fit.intercept+set.fit.slope*Math.log(Math.max(set.vocabulary,1))]);ymin=Math.min(0,...ends);ymax=Math.max(ymax,...ends);}
    ymax+=(ymax-ymin||1)*.06;
    const x=rank=>chart.L+(mode==='frequency'?rank:Math.log(rank))/xmax*chart.W,y=value=>chart.T+chart.H-(value-ymin)/(ymax-ymin||1)*chart.H,compact=value=>mode==='frequency'?(value>=1000?`${Number((value/1000).toFixed(1))}k`:`${Math.round(value)}`):value.toFixed(1);
    const defs=add('defs'),clipId=`chart-clip-${mode}`,clip=add('clipPath',{id:clipId},null,defs);add('rect',{x:chart.L,y:chart.T,width:chart.W,height:chart.H},null,clip);const group=add('g',{'clip-path':`url(#${clipId})`});
    if(mode==='log'&&sets.length===1&&sets[0].vocabulary>20){const cuts=[1,Math.ceil(count*.05),Math.ceil(count*.5),count];for(let i=0;i<3;i++){add('rect',{x:x(cuts[i]),y:chart.T,width:Math.max(0,x(cuts[i+1])-x(cuts[i])),height:chart.H,fill:i===1?'#f6f9fb':'#fbfdff'},null,group);if(x(cuts[i+1])-x(cuts[i])>78)label((x(cuts[i])+x(cuts[i+1]))/2,chart.T+14,['HEAD','MIDDLE','TAIL'][i],{'text-anchor':'middle','font-size':10,fill:'#91a0aa'});}}
    for(let i=0;i<=4;i++){const value=ymin+i*(ymax-ymin)/4,yy=y(value),axisValue=i*xmax/4,xx=chart.L+i*chart.W/4;add('line',{x1:chart.L,y1:yy,x2:chart.L+chart.W,y2:yy,stroke:'#edf1f4','stroke-width':1});label(chart.L-9,yy+4,compact(value),{'text-anchor':'end'});add('line',{x1:xx,y1:chart.T+chart.H,x2:xx,y2:chart.T+chart.H+5,stroke:'#c7d3dc'});label(xx,chart.T+chart.H+20,compact(axisValue),{'text-anchor':'middle'});}
    add('line',{x1:chart.L,y1:chart.T,x2:chart.L,y2:chart.T+chart.H,stroke:'#c7d3dc'});add('line',{x1:chart.L,y1:chart.T+chart.H,x2:chart.L+chart.W,y2:chart.T+chart.H,stroke:'#c7d3dc'});label(chart.L+chart.W/2,chart.canvasH-7,mode==='frequency'?'Term rank':'ln(Term rank)',{'text-anchor':'middle',fill:'#496277','font-size':13});label(15,chart.T+chart.H/2,mode==='frequency'?'Collection frequency (CF)':'ln(Collection frequency)',{transform:`rotate(-90 15 ${chart.T+chart.H/2})`,'text-anchor':'middle',fill:'#496277','font-size':13});
    for(const set of sets){const points=set.points.map(point=>({x:x(point.rank),y:y(mode==='frequency'?point.cf:Math.log(point.cf)),point,set}));chart.coords.push(...points);add('path',{d:points.map((point,index)=>`${index?'L':'M'}${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(' '),fill:'none',stroke:colors[set.key],'stroke-width':2.7,'stroke-linejoin':'round'},null,group);if(mode==='log'&&fitToggle.checked&&set.fit.slope!==null)add('line',{x1:x(1),y1:y(set.fit.intercept),x2:x(set.vocabulary),y2:y(set.fit.intercept+set.fit.slope*Math.log(set.vocabulary)),stroke:colors[set.key],'stroke-opacity':.62,'stroke-width':1.7,'stroke-dasharray':'6 5'},null,group);}
    chart.hover=add('circle',{class:'chart-hover-dot',r:5,fill:'#fff',stroke:'#2a6aa4','stroke-width':2,visibility:'hidden'});
  }
  function redraw(){charts.forEach(draw);rememberControls();}
  for(const chart of charts){chart.svg.addEventListener('pointermove',event=>{if(!chart.coords.length)return;const point=chart.svg.createSVGPoint();point.x=event.clientX;point.y=event.clientY;const local=point.matrixTransform(chart.svg.getScreenCTM().inverse());if(local.x<chart.L||local.x>chart.L+chart.W||local.y<chart.T||local.y>chart.T+chart.H){chart.tip.hidden=true;chart.hover?.setAttribute('visibility','hidden');return;}let nearest=null,distance=Infinity;for(const candidate of chart.coords){const current=(candidate.x-local.x)**2+(candidate.y-local.y)**2;if(current<distance){distance=current;nearest=candidate;}}if(!nearest)return;chart.tip.textContent=`${nearest.set.key} · ${nearest.point.term}\nRank ${nearest.point.rank.toLocaleString()}  /  CF ${nearest.point.cf.toLocaleString()}`;chart.tip.hidden=false;const bounds=chart.container.getBoundingClientRect();chart.tip.style.left=`${Math.max(0,Math.min(event.clientX-bounds.left+12,bounds.width-210))}px`;chart.tip.style.top=`${Math.max(0,event.clientY-bounds.top-66)}px`;chart.hover.setAttribute('cx',nearest.x);chart.hover.setAttribute('cy',nearest.y);chart.hover.setAttribute('stroke',colors[nearest.set.key]);chart.hover.setAttribute('visibility','visible');});chart.svg.addEventListener('pointerleave',()=>{chart.tip.hidden=true;chart.hover?.setAttribute('visibility','hidden');});}
  panel.querySelectorAll('[data-series]').forEach(box=>box.addEventListener('change',()=>{active=Array.from(panel.querySelectorAll('[data-series]:checked')).map(input=>input.value);redraw();}));fitToggle.addEventListener('change',redraw);
  panel.querySelectorAll('[data-save-chart]').forEach(button=>button.addEventListener('click',()=>{const chart=charts.find(item=>item.mode===button.dataset.saveChart),copy=chart.svg.cloneNode(true);copy.setAttribute('xmlns',ns);copy.setAttribute('width',String(chart.canvasW));copy.setAttribute('height',String(chart.canvasH));copy.querySelector('.chart-hover-dot')?.remove();const url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(copy)],{type:'image/svg+xml'})),link=document.createElement('a');link.href=url;link.download=`zipf_${active.join('')||'none'}_${chart.mode}.svg`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}));
  let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(redraw,120);});redraw();
})();
