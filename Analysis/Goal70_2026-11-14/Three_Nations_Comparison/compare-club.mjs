import fs from 'node:fs';
import {fetch,api,save,pool,ro,csv} from '../../Automation/research_2026_11_13.mjs';
const D='Analysis/Goal70_2026-11-14/Three_Nations_Comparison',items=[];
await pool(['5868546','5181846'],async id=>{const r=await fetch(api('matchDetails?matchId='+id));if(!r.ok||String(r.data.general?.matchId)!==id)throw Error('Identity/status unavailable '+id);save(D+'/fresh_'+id+'.json',r);const d=r.data;items.push({id,at_ro:ro(),source:r.source,status:d.header.status,lineup_type:d.content?.lineup?.lineupType,weather:d.content?.weather,teams:['homeTeam','awayTeam'].map(k=>({name:d.content?.lineup?.[k]?.name,formation:d.content?.lineup?.[k]?.formation,coach:d.content?.lineup?.[k]?.coach?.name}))});});
save(D+'/club_comparison_status.json',items);
const m=csv('Analysis/Goal70_2026-11-14/all_models.csv').find(r=>r.id==='5868546');
const a=((20/13+4/12)+(19/12+15/13))/2,b=((13/6+3/7)+(9/7+8/6))/2,l=(a+b)/2;
if(Math.abs(Number(m.p_raw)-(1-Math.exp(-l)))>1e-10)throw Error('Saved model does not reconcile');
console.log(JSON.stringify({at_ro:ro(),matches:items,calculation:{season:a,venue:b,combined:l,p:1-Math.exp(-l),stress:1-Math.exp(-.8*l)},unstarted:items.every(i=>!i.status.started&&!i.status.cancelled&&Date.parse(i.status.utcTime)>Date.now())},null,2));
