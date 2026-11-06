import fs from 'node:fs';
import path from 'node:path';
import {read,csv,save,writeCsv,pool,fetch,api,match,observation,ro} from './research_2026_11_06.mjs';
const base='Analysis/Goal70_2026-11-06', date='2026-11-06';
const finalists=['5739547','5868128','5881718','5803312'];
const candidates=read(base+'/candidate_fixtures.json');
const allTeams=new Set();
for(const id of finalists){const f=candidates.find(f=>f.id===id);allTeams.add(f.home_id);allTeams.add(f.away_id);}
await pool([...allTeams],async id=>{const p=base+'/Context/team_'+id+'.json';if(!fs.existsSync(p))save(p,await fetch(api('teams?id='+id)));});
await pool(finalists,async id=>{const p=base+'/Context/match_'+id+'.json';if(!fs.existsSync(p))save(p,await fetch(api('matchDetails?matchId='+id)));});
const recent=[];
for(const id of allTeams){const t=read(base+'/Context/team_'+id+'.json');if(!t.ok)continue;
const fixtures=t.data.fixtures?.allFixtures?.fixtures??[];
for(const f of fixtures.filter(f=>f.status?.finished&&!f.status?.cancelled&&!f.status?.awarded&&Date.parse(f.status.utcTime)<Date.parse(date+'T00:00:00+02:00')&&!/friendly/i.test(f.tournament?.name??'')&&f.tournament?.leagueId!==489).sort((a,b)=>Date.parse(b.status.utcTime)-Date.parse(a.status.utcTime)).slice(0,10))recent.push({team_id:id,id:String(f.id),date:f.status.utcTime,competition:f.tournament?.name,home:f.home?.name,away:f.away?.name,home_id:String(f.home?.id),away_id:String(f.away?.id)});
}
await pool([...new Set(recent.map(r=>r.id))],async id=>{const existing=base+'/MatchData/'+id+'.json';if(!fs.existsSync(existing))await match(id,base+'/OfficialRecent');});
const rows=recent.map(r=>{const p=base+'/MatchData/'+r.id+'.json';const d=read(fs.existsSync(p)?p:base+'/OfficialRecent/'+r.id+'.json');const o=observation(d);return {...r,...o,goal_result:o.goal_result??'pending',corner_result:o.corner_result??'pending',source:d.source};});
writeCsv(base+'/last10_official.csv',rows);
const contexts=finalists.map(id=>{const r=read(base+'/Context/match_'+id+'.json');const d=r.data;return {id,source:r.source,verified_at_utc:r.verified_at_utc,status:d?.header?.status,lineup_type:d?.content?.lineup?.lineupType,stadium:d?.content?.matchFacts?.infoBox?.Stadium,weather:d?.content?.weather,table:d?.content?.table,teams:['homeTeam','awayTeam'].map(s=>{const t=d?.content?.lineup?.[s];return {name:t?.name,formation:t?.formation,coach:t?.coach?.name,starters:(t?.starters??[]).map(p=>p.name),unavailable:(t?.unavailable??[]).map(p=>({name:p.name,reason:p.unavailability}))};})};});
save(base+'/context_summary.json',{recorded_at_ro:ro(),matches:contexts});
// Fresh kickoff/status verification for the whole compared universe.
await pool(candidates,async f=>{save(base+'/StatusAtSelection/'+f.id+'.json',await fetch(api('matchDetails?matchId='+f.id)));});
const statusRows=candidates.map(f=>{const r=read(base+'/StatusAtSelection/'+f.id+'.json'),s=r.data?.header?.status;return {id:f.id,home:f.home,away:f.away,source:r.source,verified_at_utc:r.verified_at_utc,identity_ok:String(r.data?.general?.matchId)===f.id,utcTime:s?.utcTime,started:s?.started,finished:s?.finished,cancelled:s?.cancelled,unstarted_verified:!!(r.ok&&String(r.data?.general?.matchId)===f.id&&s&&!s.started&&!s.finished&&!s.cancelled&&Date.parse(s.utcTime)>Date.now())};});
writeCsv(base+'/status_at_selection.csv',statusRows);
console.log(JSON.stringify({last10_rows:rows.length,teams:allTeams.size,missing_recent_goals:rows.filter(r=>r.goal_result==='pending').length,missing_recent_corners:rows.filter(r=>r.corner_result==='pending').length,status_verified:statusRows.filter(r=>r.unstarted_verified).length,total:statusRows.length}));
