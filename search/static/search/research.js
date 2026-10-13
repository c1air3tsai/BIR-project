'use strict';
(() => {
  const markStart=name=>{try{performance.mark(`${name}-start`);}catch(_){}};
  const markEnd=name=>{try{performance.mark(`${name}-end`);performance.measure(name,`${name}-start`,`${name}-end`);}catch(_){}};
  const t=(key,fallback)=>window.biomedI18n?.t(key)||fallback;
  markStart('word2vec-page-init');

  const domains=document.querySelector('[data-domain-form]');
  if(domains){
    let timer;
    domains.addEventListener('change',()=>{clearTimeout(timer);timer=setTimeout(()=>domains.requestSubmit(),450);});
    domains.addEventListener('submit',()=>{
      clearTimeout(timer);
      const url=new URL(location.href);
      for(const name of ['series','fit']){
        if(!url.searchParams.has(name))continue;
        let field=domains.elements.namedItem(name);
        if(!field){field=document.createElement('input');field.type='hidden';field.name=name;domains.appendChild(field);}
        field.value=url.searchParams.get(name);
      }
    });
  }

  const training=document.querySelector('[data-training-form]');
  if(training){
    const resetSubmitState=()=>{training.querySelectorAll('button[type=submit]').forEach(button=>button.disabled=false);training.querySelectorAll('[data-training-status]').forEach(status=>status.hidden=true);};
    window.addEventListener('pageshow',resetSubmitState);
    training.addEventListener('submit',()=>{training.querySelector('button[type=submit]').disabled=true;training.querySelector('[data-training-status]').hidden=false;});
    const preset=training.querySelector('[data-corpus-preset-field]');
    const medical=training.elements.namedItem('medical_count'),cs=training.elements.namedItem('cs_count');
    training.querySelectorAll('[data-corpus-preset]').forEach(button=>{
      if(button.dataset.corpusPreset===preset.value)button.classList.add('active');
      button.addEventListener('click',()=>{
        preset.value=button.dataset.corpusPreset;
        if(button.dataset.medical!==undefined)medical.value=button.dataset.medical;
        if(button.dataset.cs!==undefined)cs.value=button.dataset.cs;
        training.querySelectorAll('[data-corpus-preset]').forEach(item=>item.classList.toggle('active',item===button));
        if(button.dataset.corpusPreset!=='custom'){
          const url=new URL(location.href);url.search='';
          url.searchParams.set('corpus_preset',button.dataset.corpusPreset);
          url.searchParams.set('medical_count',medical.value);url.searchParams.set('cs_count',cs.value);
          for(const [name,value] of Object.entries({condition:'C',dimensions:'100',window:'5',min_count:'2',epochs:'20'}))url.searchParams.set(name,value);
          url.hash='training-setup';
          location.assign(url.toString());
        }
      });
    });
    [medical,cs].forEach(input=>input.addEventListener('input',()=>{preset.value='custom';training.querySelectorAll('[data-corpus-preset]').forEach(item=>item.classList.toggle('active',item.dataset.corpusPreset==='custom'));}));
  }

  const experimentForm=document.querySelector('[data-experiment-form]');
  if(experimentForm){
    window.addEventListener('pageshow',()=>{experimentForm.querySelectorAll('button[type=submit]').forEach(button=>button.disabled=false);experimentForm.querySelectorAll('[data-training-status]').forEach(status=>status.hidden=true);});
    const validateComparisonPairs=()=>{
      let valid=true;
      experimentForm.querySelectorAll('.parameter-input-pair').forEach(pair=>{
        const inputs=[...pair.querySelectorAll('input[type=number]')],same=inputs.length===2&&inputs[0].value===inputs[1].value;
        inputs.forEach(input=>input.setCustomValidity(same?t('we.compare.values.differ','Choose two different values.') : ''));
        if(same)valid=false;
      });
      return valid;
    };
    experimentForm.querySelectorAll('input[type=number]').forEach(input=>input.addEventListener('input',validateComparisonPairs));
    experimentForm.addEventListener('submit',event=>{if(!validateComparisonPairs()){event.preventDefault();experimentForm.reportValidity();return;}experimentForm.querySelector('button[type=submit]').disabled=true;experimentForm.querySelector('[data-training-status]').hidden=false;});
  }

  const localForm=document.querySelector('[data-local-form]');
  if(localForm){
    const resetLocalState=()=>{localForm.querySelectorAll('button[type=submit]').forEach(button=>button.disabled=false);document.querySelectorAll('[data-local-status]').forEach(status=>status.hidden=true);};
    window.addEventListener('pageshow',resetLocalState);
    localForm.addEventListener('submit',()=>{localForm.querySelector('button[type=submit]').disabled=true;document.querySelector('[data-local-status]').hidden=false;});
  }

  const ns='http://www.w3.org/2000/svg';
  function renderFailure(svg,error){
    console.error(`Projection render failed for ${svg?.id||'unknown chart'}`,error);
    if(!svg)return;
    svg.replaceChildren();
    const text=document.createElementNS(ns,'text');text.setAttribute('x','20');text.setAttribute('y','42');text.setAttribute('class','projection-error-text');text.textContent=t('we.render.error','This chart could not be rendered. The rest of the page remains available.');svg.appendChild(text);
    const notice=document.createElement('p');notice.className='notice warning chart-render-error';notice.dataset.i18nKey='we.render.error';notice.textContent=text.textContent;svg.after(notice);
  }
  function safelyRender(name,svg,callback){
    markStart(name);
    try{callback();}
    catch(error){renderFailure(svg,error);}
    finally{markEnd(name);}
  }

  function renderLoss(){
    const source=document.getElementById('embedding-loss-data'),svg=document.getElementById('embedding-loss-chart');
    if(!source||!svg)return;
    const series=JSON.parse(source.textContent)||[],points=series.flatMap(item=>item.points||[]);
    if(!points.length)return;
    svg.replaceChildren();
    const width=900,height=380,left=72,right=28,top=35,bottom=58;
    const maxEpoch=Math.max(...points.map(point=>point.epoch),1),maxLoss=Math.max(...points.map(point=>point.loss),1);
    const x=epoch=>left+(epoch-1)/(maxEpoch-1||1)*(width-left-right),y=value=>top+(1-value/maxLoss)*(height-top-bottom);
    const add=(tag,attrs,text,parent=svg)=>{const element=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([key,value])=>element.setAttribute(key,value));if(text!==undefined)element.textContent=text;parent.appendChild(element);return element;};
    add('rect',{x:0,y:0,width,height,fill:'#f8fbfe',rx:8});
    for(let index=0;index<=4;index++){const yy=top+index*(height-top-bottom)/4;add('line',{x1:left,y1:yy,x2:width-right,y2:yy,stroke:'#dbe6ee'});add('text',{x:left-10,y:yy+4,'text-anchor':'end',fill:'#61798c','font-size':11},(maxLoss*(1-index/4)).toFixed(0));}
    add('line',{x1:left,y1:height-bottom,x2:width-right,y2:height-bottom,stroke:'#7890a2'});
    add('text',{x:width/2,y:height-16,'text-anchor':'middle',fill:'#496277','font-size':12},t('we.epoch','Epoch'));
    const colors=['#0072b2','#d55e00'];
    series.forEach((item,index)=>{
      const rows=item.points||[],path=rows.map((point,rowIndex)=>`${rowIndex?'L':'M'} ${x(point.epoch)} ${y(point.loss)}`).join(' ');
      add('path',{d:path,fill:'none',stroke:colors[index%colors.length],'stroke-width':2.5});
      rows.forEach(point=>{const dot=add('circle',{cx:x(point.epoch),cy:y(point.loss),r:4,fill:colors[index%colors.length]});const title=document.createElementNS(ns,'title');title.textContent=`${item.label} · ${t('we.epoch','Epoch')} ${point.epoch} · ${t('we.loss','Loss')} ${point.loss.toFixed(3)} · ${point.seconds}s`;dot.appendChild(title);});
      add('rect',{x:width-right-150,y:16+index*20,width:12,height:12,fill:colors[index%colors.length]});add('text',{x:width-right-132,y:27+index*20,fill:'#203e59','font-size':12},item.label);
    });
  }

  function renderInteractiveProjection(sourceId,svgId,axisLabel){
    const source=document.getElementById(sourceId),svg=document.getElementById(svgId);
    if(!source||!svg)return;
    const data=JSON.parse(source.textContent)||{};
    const points=(data.points||[]).filter(point=>Number.isFinite(Number(point.x))&&Number.isFinite(Number(point.y))&&point.term);
    if(!points.length)return;
    svg.replaceChildren();
    const width=700,height=500,pad=55;
    const xs=points.map(point=>Number(point.x)),ys=points.map(point=>Number(point.y));
    const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
    const px=value=>pad+(Number(value)-xmin)/(xmax-xmin||1)*(width-pad*2),py=value=>pad+(1-(Number(value)-ymin)/(ymax-ymin||1))*(height-pad*2);
    const add=(tag,attrs,text,parent=svg)=>{const element=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([key,value])=>element.setAttribute(key,value));if(text!==undefined)element.textContent=text;parent.appendChild(element);return element;};
    add('rect',{x:0,y:0,width,height,fill:'#f8fbfe',rx:8});add('text',{x:16,y:24,fill:'#496277','font-size':12},axisLabel);
    const pointLayer=add('g',{'class':'projection-points'}),labelLayer=add('g',{'class':'projection-labels','pointer-events':'none'});
    const tooltip=add('g',{'class':'projection-tooltip','visibility':'hidden','pointer-events':'none'}),tooltipRect=add('rect',{rx:6,ry:6},undefined,tooltip),tooltipText=add('text',{},'',tooltip);
    const counts=points.map(point=>point.count||1),cmin=Math.min(...counts),cmax=Math.max(...counts);
    const pointFragment=document.createDocumentFragment();
    const items=points.map((point,index)=>{
      const baseX=px(point.x),baseY=py(point.y),rawRadius=4+4*Math.sqrt(((point.count||cmin)-cmin)/(cmax-cmin||1));
      const selected=point.highlight==='target'||point.term===data.selected,neighbor=point.highlight==='neighbor';
      const radius=selected?Math.max(10,rawRadius*1.6):(neighbor?Math.max(7,rawRadius*1.25):rawRadius);
      const fill=selected?'#d55e00':(neighbor?'#2e9d68':(index<5?'#0072b2':'#83a4bd'));
      const pointClass=selected?'is-target':(neighbor?'is-neighbor':'');
      const stroke=selected?'#8f3b00':(neighbor?'#176b48':'#fff'),strokeWidth=selected?2.5:(neighbor?1.8:1);
      const circle=add('circle',{cx:baseX,cy:baseY,r:radius,fill,'class':pointClass,'fill-opacity':(selected||neighbor)?0.94:0.82,stroke,'stroke-width':strokeWidth,'vector-effect':'non-scaling-stroke',tabindex:0,role:'img','aria-label':`${point.term}, CF ${point.count||'n/a'}`},undefined,pointFragment);
      const title=document.createElementNS(ns,'title');title.textContent=`${point.term} · CF ${point.count||'n/a'}`;circle.appendChild(title);
      return {point,index,baseX,baseY,radius,selected,neighbor,circle};
    });
    pointLayer.appendChild(pointFragment);
    const labels=new Map();
    const labelFor=item=>{
      if(labels.has(item.index))return labels.get(item.index);
      const group=add('g',{'class':'projection-label',display:'none'},undefined,labelLayer);
      const value={group,background:add('rect',{rx:4,ry:4},undefined,group),text:add('text',{},'',group)};
      labels.set(item.index,value);return value;
    };
    let scale=1,tx=0,ty=0,drag=null,activeItem=null,animationFrame=null;
    const screenPosition=item=>({x:tx+scale*item.baseX,y:ty+scale*item.baseY});
    const labelBudget=()=>scale<=1.2?24:scale<=2?40:scale<=4?80:120;
    const placeLabels=()=>{
      markStart('label-layout');labels.forEach(label=>label.group.setAttribute('display','none'));
      const ordered=[...items].sort((a,b)=>(b.point.count||0)-(a.point.count||0)||a.index-b.index).slice(0,labelBudget());
      const grid=new Map(),cellSize=56;
      const gridKeys=box=>{const keys=[];for(let x=Math.floor(box.x/cellSize);x<=Math.floor((box.x+box.width)/cellSize);x++)for(let y=Math.floor(box.y/cellSize);y<=Math.floor((box.y+box.height)/cellSize);y++)keys.push(`${x}:${y}`);return keys;};
      const collision=box=>gridKeys(box).some(key=>(grid.get(key)||[]).some(other=>box.x<other.x+other.width&&box.x+box.width>other.x&&box.y<other.y+other.height&&box.y+box.height>other.y));
      const occupy=box=>gridKeys(box).forEach(key=>{if(!grid.has(key))grid.set(key,[]);grid.get(key).push(box);});
      ordered.forEach(item=>{
        const point=screenPosition(item);if(point.x<-20||point.x>width+20||point.y<-20||point.y>height+20)return;
        const fullTerm=String(item.point.term),displayTerm=fullTerm.length>28?`${fullTerm.slice(0,27)}…`:fullTerm;
        const labelWidth=Math.max(34,displayTerm.length*6.4+12),labelHeight=18,gap=Math.max(7,item.radius*scale*.45+4);
        const candidates=[{x:point.x+gap,y:point.y-labelHeight-3},{x:point.x+gap,y:point.y+4},{x:point.x-gap-labelWidth,y:point.y-labelHeight-3},{x:point.x-gap-labelWidth,y:point.y+4},{x:point.x-labelWidth/2,y:point.y-gap-labelHeight},{x:point.x-labelWidth/2,y:point.y+gap}];
        let box=candidates.map(candidate=>({...candidate,width:labelWidth,height:labelHeight})).find(candidate=>candidate.x>=4&&candidate.x+labelWidth<=width-4&&candidate.y>=31&&candidate.y+labelHeight<=height-4&&!collision(candidate));
        if(!box)return;occupy(box);
        const label=labelFor(item);label.group.setAttribute('display','');label.background.setAttribute('x',box.x);label.background.setAttribute('y',box.y);label.background.setAttribute('width',labelWidth);label.background.setAttribute('height',labelHeight);label.text.setAttribute('x',box.x+6);label.text.setAttribute('y',box.y+12.5);label.text.textContent=displayTerm;
      });markEnd('label-layout');
    };
    const showTooltip=item=>{
      if(activeItem&&activeItem!==item)activeItem.circle.classList.remove('is-hovered');activeItem=item;item.circle.classList.add('is-hovered');
      const point=screenPosition(item),content=`${item.point.term} · CF ${item.point.count||'n/a'}`,tooltipWidth=Math.min(width-16,Math.max(96,content.length*7.1+18)),tooltipHeight=28;
      const x=Math.max(8,Math.min(width-tooltipWidth-8,point.x+12));let y=point.y-tooltipHeight-12;if(y<34)y=Math.min(height-tooltipHeight-8,point.y+12);
      tooltipRect.setAttribute('x',x);tooltipRect.setAttribute('y',y);tooltipRect.setAttribute('width',tooltipWidth);tooltipRect.setAttribute('height',tooltipHeight);tooltipText.setAttribute('x',x+9);tooltipText.setAttribute('y',y+18);tooltipText.textContent=content;tooltip.setAttribute('visibility','visible');svg.appendChild(tooltip);
    };
    const hideTooltip=item=>{if(activeItem!==item)return;item.circle.classList.remove('is-hovered');activeItem=null;tooltip.setAttribute('visibility','hidden');};
    items.forEach(item=>{item.circle.addEventListener('mouseenter',()=>showTooltip(item));item.circle.addEventListener('mouseleave',()=>hideTooltip(item));item.circle.addEventListener('focus',()=>showTooltip(item));item.circle.addEventListener('blur',()=>hideTooltip(item));});
    const updateNow=()=>{animationFrame=null;pointLayer.setAttribute('transform',`translate(${tx} ${ty}) scale(${scale})`);placeLabels();if(activeItem)showTooltip(activeItem);};
    const update=()=>{if(animationFrame===null)animationFrame=requestAnimationFrame(updateNow);};
    svg.addEventListener('wheel',event=>{event.preventDefault();const next=Math.max(.7,Math.min(8,scale*(event.deltaY<0?1.18:.85))),rect=svg.getBoundingClientRect(),mx=(event.clientX-rect.left)*width/rect.width,my=(event.clientY-rect.top)*height/rect.height;tx=mx-(mx-tx)*next/scale;ty=my-(my-ty)*next/scale;scale=next;update();},{passive:false});
    svg.addEventListener('pointerdown',event=>{drag={x:event.clientX,y:event.clientY,tx,ty};svg.setPointerCapture(event.pointerId);});
    svg.addEventListener('pointermove',event=>{if(!drag)return;const rect=svg.getBoundingClientRect();tx=drag.tx+(event.clientX-drag.x)*width/rect.width;ty=drag.ty+(event.clientY-drag.y)*height/rect.height;update();});
    const stopDrag=()=>{drag=null;};svg.addEventListener('pointerup',stopDrag);svg.addEventListener('pointercancel',stopDrag);
    document.querySelector(`[data-reset-plot="${svgId}"]`)?.addEventListener('click',()=>{scale=1;tx=0;ty=0;update();});
    document.querySelectorAll(`[data-zoom-plot="${svgId}"]`).forEach(button=>button.addEventListener('click',()=>{scale=Math.max(.7,Math.min(8,scale*Number(button.dataset.zoomFactor||1)));update();}));
    updateNow();
  }

  const renderProjection=svg=>{
    if(svg.dataset.renderState)return;
    svg.dataset.renderState='rendering';
    const architecture=svg.id.includes('skipgram')?'SkipGram':'CBOW',method=svg.id.endsWith('pca')?'PCA':'TSNE';
    safelyRender(`${architecture}-${method}-render`,svg,()=>renderInteractiveProjection(svg.dataset.projectionSource,svg.id,method==='PCA'?'PC1 / PC2':'t-SNE'));
    svg.dataset.renderState='ready';
  };
  const lossSvg=document.getElementById('embedding-loss-chart');if(lossSvg)safelyRender('Training-Loss-render',lossSvg,renderLoss);
  const projectionSvgs=[...document.querySelectorAll('[data-projection-source]')];
  if('IntersectionObserver' in window){
    const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{if(entry.isIntersecting){observer.unobserve(entry.target);requestAnimationFrame(()=>renderProjection(entry.target));}}),{rootMargin:'600px 0px'});
    projectionSvgs.forEach(svg=>observer.observe(svg));
  }else{
    const schedule=window.requestIdleCallback?((callback,options)=>window.requestIdleCallback(callback,options)):((callback)=>setTimeout(callback,16));
    const renderNext=index=>{if(index>=projectionSvgs.length)return;schedule(()=>{renderProjection(projectionSvgs[index]);renderNext(index+1);},{timeout:500});};
    renderNext(0);
  }
  markEnd('word2vec-page-init');
})();
