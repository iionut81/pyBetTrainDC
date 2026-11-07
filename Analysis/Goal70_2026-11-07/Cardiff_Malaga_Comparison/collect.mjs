import fs from 'node:fs';
import assert from 'node:assert/strict';
import {read,csv,save,writeCsv,pool,fetch,api,match,observation,ro} from '../../Automation/research_2026_11_07.mjs';
const base='Analysis/Goal70_2026-11-07',dir=base+'/Cardiff_Malaga_Comparison';
const ids=['5836925','5868125','5836923'],teams=['8344','8191','9864','8696','8655','8654'];
const clean=x=>Array.isArray(x)?x.map(clean):x&&typeof x==='object'?Object.fromEntries(Object.entries(x).filter(([k])=>!/odds|betting|bookmaker/i.test(k)).map(([k,v])=>[k,clean(v)])):x;
await pool([...ids.map(id=>['match',id]),...teams.map(id=>['team',id])],async([type,id])=>{
 const r=await fetch(api(type==='match'?'matchDetails?matchId='+id:'teams?id='+id));
 if(r.ok)r.data=clean(r.data);
 save(`${dir}/${type}_${id}.json`,r);
});
const recent=[];
for(const id of teams){
 const t=read(`${dir}/team_${id}.json`);
 if(!t.ok)continue;
 const fixtures=t.data.fixtures?.allFixtures?.fixtures??[];
 for(const f of fixtures.filter(f=>f.status?.finished&&!f.status?.cancelled&&!f.status?.awarded&&Date.parse(f.status.utcTime)<Date.parse('2026-11-07T00:00:00+02:00')&&!/friendly/i.test(f.tournament?.name??'')&&f.tournament?.leagueId!==489).sort((a,b)=>Date.parse(b.status.utcTime)-Date.parse(a.status.utcTime)).slice(0,10))recent.push({team_id:id,id:String(f.id),date:f.status.utcTime,competition:f.tournament?.name,home:f.home?.name,away:f.away?.name});
}
await pool([...new Set(recent.map(r=>r.id))],async id=>{
 if(!fs.existsSync(base+'/MatchData/'+id+'.json'))await match(id,dir+'/Recent');
});
const rows=recent.map(r=>{const p=base+'/MatchData/'+r.id+'.json';const d=read(fs.existsSync(p)?p:dir+'/Recent/'+r.id+'.json');return {...r,...observation(d),source:d.source};});
writeCsv(dir+'/last10_official.csv',rows);
const history=read(base+'/historical_observations.json'),models=csv(base+'/all_models.csv');
const summaries=ids.map(id=>{
 const row=models.find(r=>r.id===id),r=read(`${dir}/match_${id}.json`),d=r.data;
 const sets={};
 for(const [prefix,team,venue] of [['home',row.home_id,null],['away',row.away_id,null],['home_venue',row.home_id,'home'],['away_venue',row.away_id,'away']]){
  const h=history.filter(h=>h.league===row.competition&&h.goal_ok&&(venue?h[venue+'_id']===team:h.home_id===team||h.away_id===team));
  const gf=h.reduce((s,h)=>s+(h.home_id===team?h.home70:h.away70),0),ga=h.reduce((s,h)=>s+(h.home_id===team?h.away70:h.home70),0);
  assert.equal(h.length,Number(row[prefix+'_n']));assert.equal(gf,Number(row[prefix+'_gf70']));assert.equal(ga,Number(row[prefix+'_ga70']));
  sets[prefix]={n:h.length,gf,ga,wins:h.filter(h=>h.success70).length,failures:h.filter(h=>!h.success70).map(h=>({id:h.id,date:h.date,home:h.home,away:h.away,score:h.score,first_goal:h.first_goal,source:h.source})),chronology:h.map(h=>({date:h.date,id:h.id,home:h.home,away:h.away,score:h.score,first_goal:h.first_goal,goals:h.goals,cards:h.cards}))};
 }
 const a=sets.home,b=sets.away,av=sets.home_venue,bv=sets.away_venue;
 const ls=(a.gf/a.n+b.ga/b.n+b.gf/b.n+a.ga/a.n)/2,lv=(av.gf/av.n+bv.ga/bv.n+bv.gf/bv.n+av.ga/av.n)/2,l=(ls+lv)/2;
 assert.ok(Math.abs(1-Math.exp(-l)-Number(row.p_raw))<1e-12);
 return {row,sets,calculation:{lambda_season:ls,lambda_venue:lv,lambda_combined:l,p:1-Math.exp(-l),stress:1-Math.exp(-.8*l)},current:{ok:r.ok,source:r.source,verified_at_utc:r.verified_at_utc,general:d?.general,status:d?.header?.status,lineup_type:d?.content?.lineup?.lineupType,stadium:d?.content?.matchFacts?.infoBox?.Stadium,weather:d?.content?.weather,infoBox:d?.content?.matchFacts?.infoBox,table:d?.content?.table,teams:['homeTeam','awayTeam'].map(s=>{const t=d?.content?.lineup?.[s];return {name:t?.name,formation:t?.formation,coach:t?.coach?.name,starters:(t?.starters??[]).map(p=>p.name),unavailable:(t?.unavailable??[]).map(p=>({name:p.name,reason:p.unavailability}))};})},recent:rows.filter(r=>[row.home_id,row.away_id].includes(r.team_id))};
});
save(dir+'/comparison.json',{updated_at_ro:ro(),scope:'Cardiff versus Malaga, then Blackburn-West Ham versus Malaga as requested by the user. Within the saved 102-fixture inventory, not a new global ranking. Original probabilities unchanged; no bookmaker odds.',matches:summaries});
console.log(JSON.stringify(summaries.map(s=>({id:s.row.id,p:s.calculation.p,status:s.current.status,recent:s.recent.length})),null,2));
