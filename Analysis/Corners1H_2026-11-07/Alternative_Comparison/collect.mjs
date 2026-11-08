import fs from 'node:fs';
import assert from 'node:assert/strict';
import {read,csv,save,writeCsv,pool,fetch,api,match,observation,ro} from '../../Automation/research_2026_11_07.mjs';
const goal='Analysis/Goal70_2026-11-07',base='Analysis/Corners1H_2026-11-07',dir=base+'/Alternative_Comparison';
const models=csv(base+'/all_models.csv').filter(x=>['5781800','5887679','5831220','5803309'].includes(x.id));
const ids=models.map(x=>x.id),teamIds=[...new Set(models.flatMap(x=>[x.home_id,x.away_id]))];
const clean=x=>Array.isArray(x)?x.map(clean):x&&typeof x==='object'?Object.fromEntries(Object.entries(x).filter(([k])=>!/odds|betting|bookmaker/i.test(k)).map(([k,v])=>[k,clean(v)])):x;
await pool([...ids.map(id=>['match',id]),...teamIds.map(id=>['team',id])],async([type,id])=>{
 const r=await fetch(api(type==='match'?'matchDetails?matchId='+id:'teams?id='+id));
 if(r.ok)r.data=clean(r.data);
 save(`${dir}/${type}_${id}.json`,r);
});
const recent=[];
for(const id of teamIds){const t=read(`${dir}/team_${id}.json`);if(!t.ok)continue;
 const fixtures=t.data.fixtures?.allFixtures?.fixtures??[];
 for(const f of fixtures.filter(f=>f.status?.finished&&!f.status?.cancelled&&!f.status?.awarded&&Date.parse(f.status.utcTime)<Date.parse('2026-11-07T00:00:00+02:00')&&!/friendly/i.test(f.tournament?.name??'')&&f.tournament?.leagueId!==489).sort((a,b)=>Date.parse(b.status.utcTime)-Date.parse(a.status.utcTime)).slice(0,10))recent.push({team_id:id,id:String(f.id),date:f.status.utcTime,competition:f.tournament?.name,home:f.home?.name,away:f.away?.name});
}
await pool([...new Set(recent.map(r=>r.id))],async id=>{
 if(![goal+'/MatchData/',goal+'/OfficialRecent/'].some(p=>fs.existsSync(p+id+'.json')))await match(id,dir+'/Recent');
});
const recentRows=recent.map(r=>{const p=[goal+'/MatchData/',goal+'/OfficialRecent/',dir+'/Recent/'].map(p=>p+r.id+'.json').find(p=>fs.existsSync(p));const d=read(p);return {...r,...observation(d),source:d.source};});
writeCsv(dir+'/last10_official.csv',recentRows);
const history=read(goal+'/historical_observations.json');
function sampleStats(h,id){const totals=h.map(x=>+x.total1h),mean=totals.reduce((s,x)=>s+x,0)/h.length;return {n:h.length,cf:h.reduce((s,h)=>s+(h.home_id===id?h.home_cf:h.away_cf),0),ca:h.reduce((s,h)=>s+(h.home_id===id?h.away_cf:h.home_cf),0),wins:h.filter(h=>h.under).length,seven:totals.filter(x=>x===7).length,eightplus:totals.filter(x=>x>=8).length,mean,variance:totals.length>1?totals.reduce((s,x)=>s+(x-mean)**2,0)/(totals.length-1):null,max:Math.max(...totals),totals,series:h.map(x=>({id:x.id,date:x.date,home:x.home,away:x.away,home_cf:x.home_cf,away_cf:x.away_cf,total:x.total1h,source:x.source,goals:x.goals,cards:x.cards}))};}
const cdf=l=>{let v=Math.exp(-l),sum=v;for(let k=1;k<=7;k++){v*=l/k;sum+=v;}return sum;};
const matches=models.map(row=>{
 const sets={};for(const [key,id,venue]of[['home',row.home_id,null],['away',row.away_id,null],['home_venue',row.home_id,'home'],['away_venue',row.away_id,'away']]){
  const h=history.filter(h=>h.league===row.competition&&h.corner_ok&&(venue?h[venue+'_id']===id:h.home_id===id||h.away_id===id)).sort((a,b)=>Date.parse(a.date)-Date.parse(b.date));
  sets[key]=sampleStats(h,id);for(const [a,b]of[['n','n'],['cf','gf'],['ca','ga'],['seven','seven'],['eightplus','eight']])assert.equal(sets[key][a],Number(row[key+'_'+b]));
 }
 const {home:a,away:b,home_venue:av,away_venue:bv}=sets;
 const ls=(a.cf/a.n+b.ca/b.n+b.cf/b.n+a.ca/a.n)/2,lv=(av.cf/av.n+bv.ca/bv.n+bv.cf/bv.n+av.ca/av.n)/2,l=(ls+lv)/2;
 assert.ok(Math.abs(cdf(l)-Number(row.p_raw))<1e-12);assert.ok(Math.abs(cdf(l*1.2)-Number(row.p_stress))<1e-12);
 const r=read(`${dir}/match_${row.id}.json`),d=r.data;
 const teamTable=id=>{const t=read(`${dir}/team_${id}.json`);return t.data?.table?.[0]?.data?.table?.all?.find(x=>String(x.id)===id);};
 return {row,sets,calculation:{lambda_season:ls,lambda_venue:lv,lambda_combined:l,p:cdf(l),stress:cdf(l*1.2)},context:{ok:r.ok,source:r.source,verified_at_utc:r.verified_at_utc,general:d?.general,status:d?.header?.status,lineup_type:d?.content?.lineup?.lineupType,weather:d?.content?.weather,stadium:d?.content?.matchFacts?.infoBox?.Stadium,table:[teamTable(row.home_id),teamTable(row.away_id)],teams:['homeTeam','awayTeam'].map(s=>{const t=d?.content?.lineup?.[s];return {name:t?.name,formation:t?.formation,coach:t?.coach?.name,starters:(t?.starters??[]).map(p=>p.name),unavailable:(t?.unavailable??[]).map(p=>({name:p.name,reason:p.unavailability}))};})},recent:recentRows.filter(x=>[row.home_id,row.away_id].includes(x.team_id))};
});
save(dir+'/comparison.json',{updated_at_ro:ro(),scope:'Four user-requested alternatives for CORNERS_1H_UNDER_7_5, within the archived daily inventory; not a global daily reranking. Canonical user choices and original model values unchanged.',matches});
console.log(JSON.stringify(matches.map(x=>({id:x.row.id,home:x.row.home,away:x.row.away,p:x.calculation.p,stress:x.calculation.stress,lineup:x.context.lineup_type,status:x.context.status,venue:Object.fromEntries(['home_venue','away_venue'].map(k=>[k,{mean:x.sets[k].mean,variance:x.sets[k].variance,totals:x.sets[k].totals}])),teams:x.context.teams,table:x.context.table,recent:x.recent.map(r=>({team:r.team_id,date:r.date.slice(0,10),match:r.home+'–'+r.away,competition:r.competition,corners:r.actual_corners_1h,result:r.corner_result}))})),null,2));
