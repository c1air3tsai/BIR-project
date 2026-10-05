'use strict';
(() => {
  const domains=document.querySelector('[data-domain-form]');
  if(domains) {
    let timer;
    domains.addEventListener('change',()=>{clearTimeout(timer);timer=setTimeout(()=>domains.requestSubmit(),450);});
    domains.addEventListener('submit',()=>{
      clearTimeout(timer);
      const url=new URL(location.href);
      for(const name of ['series','fit']) {
        if(!url.searchParams.has(name))continue;
        let field=domains.elements.namedItem(name);
        if(!field){field=document.createElement('input');field.type='hidden';field.name=name;domains.appendChild(field);}
        field.value=url.searchParams.get(name);
      }
    });
  }
  const training=document.querySelector('[data-training-form]');
  if(training) {
    training.querySelectorAll('select').forEach(select=>select.addEventListener('change',()=>{
      const url=new URL(location.href);
      training.querySelectorAll('select').forEach(field=>url.searchParams.set(field.name,field.value));
      url.searchParams.delete('topic');
      document.querySelectorAll('.topic-toolbar input[name=topic]:checked').forEach(box=>url.searchParams.append('topic',box.value));
      location.href=url.toString();
    }));
    training.addEventListener('submit',()=>{training.querySelector('button[type=submit]').disabled=true;training.querySelector('[data-training-status]').hidden=false;});
  }
  const ns='http://www.w3.org/2000/svg';
  function renderProjection(sourceId,svgId,overview=false) {
    const source=document.getElementById(sourceId),svg=document.getElementById(svgId);
    if(!source || !svg)return;
    const data=JSON.parse(source.textContent),points=data.points||[];
    if(!points.length)return;
    const box=svg.viewBox.baseVal,width=box.width||500,height=box.height||350;
    const pad=overview?72:55,right=overview?110:95,top=55,bottom=55;
    const xs=points.map(p=>p.x),ys=points.map(p=>p.y);
    const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
    const px=value=>pad+(value-xmin)/(xmax-xmin||1)*(width-pad-right);
    const py=value=>top+(1-(value-ymin)/(ymax-ymin||1))*(height-top-bottom);
    const add=(tag,attrs,text)=>{const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text)e.textContent=text;svg.appendChild(e);return e;};
    add('rect',{x:0,y:0,width,height,fill:'#f8fbfe',rx:8});
    const chinese=document.documentElement.lang==='zh-Hant';
    add('text',{x:16,y:24,fill:'#496277','font-size':12},overview?(chinese?'PC1 / PC2 · 模型高頻詞':'PC1 / PC2 · frequent model words'):(chinese?'PC1 / PC2 · 局部投影':'PC1 / PC2 · local projection'));
    const xZero=px(Math.max(xmin,Math.min(0,xmax))),yZero=py(Math.max(ymin,Math.min(0,ymax)));
    add('line',{x1:pad,y1:yZero,x2:width-right+15,y2:yZero,stroke:'#dbe6ee','stroke-width':1});
    add('line',{x1:xZero,y1:top-10,x2:xZero,y2:height-bottom,stroke:'#dbe6ee','stroke-width':1});
    add('text',{x:width-right+22,y:yZero+4,fill:'#7890a2','font-size':11},'PC1');
    add('text',{x:xZero+5,y:top-15,fill:'#7890a2','font-size':11},'PC2');
    const counts=points.map(p=>p.count||1),cmin=Math.min(...counts),cmax=Math.max(...counts),labels=[];
    const topFrequencyColors=['#0072b2','#d55e00','#009e73','#cc79a7','#e69f00'];
    points.forEach((p,i)=>{
      const x=px(p.x),y=py(p.y),radius=overview?4+5*Math.sqrt(((p.count||cmin)-cmin)/(cmax-cmin||1)):(i?5:8);
      const color=overview?(topFrequencyColors[i]||'#83a4bd'):(i?'#5986b4':'#2a6aa4');
      const point=add('circle',{cx:x,cy:y,r:radius,fill:color,'fill-opacity':overview ? .82 : 1,tabindex:0});
      const title=document.createElementNS(ns,'title');title.textContent=p.count?`${p.term} · ${p.count}`:p.term;point.appendChild(title);
      const label=p.term.length>22?p.term.slice(0,21)+'…':p.term,labelWidth=label.length*6.2;
      let chosen=null;
      for(let attempt=0;attempt<30;attempt++) {
        const dy=attempt===0?-9:(attempt%2?1:-1)*(Math.ceil(attempt/2)*15);
        const lx=Math.max(8,Math.min(x+radius+4,width-8-labelWidth)),ly=Math.max(35,Math.min(y+dy,height-18));
        const candidate={x:lx,y:ly-12,w:labelWidth,h:15};
        if(!labels.some(b=>candidate.x<b.x+b.w+3 && candidate.x+candidate.w+3>b.x && candidate.y<b.y+b.h+2 && candidate.y+candidate.h+2>b.y)){chosen=candidate;break;}
      }
      if(!chosen && overview && i>=25)return;
      chosen ||= {x:Math.max(8,Math.min(x+radius+4,width-8-labelWidth)),y:y-21,w:labelWidth,h:15};labels.push(chosen);
      if(Math.abs(chosen.y+12-y)>17)add('line',{x1:x,y1:y,x2:chosen.x,y2:chosen.y+8,stroke:'#c7d9e7','stroke-width':1});
      add('text',{x:chosen.x,y:chosen.y+12,fill:'#203e59','font-size':overview?11:12},label);
    });
  }
  renderProjection('vocabulary-pca-data','vocabulary-pca-chart',true);
  renderProjection('embedding-data','embedding-chart',false);
})();
