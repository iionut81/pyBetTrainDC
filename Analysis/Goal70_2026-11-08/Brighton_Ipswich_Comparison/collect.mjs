import fs from 'node:fs';
import {read,csv,save,writeCsv,pool,fetch,api,match,observation,ro} from '../../Automation/research_2026_11_08.mjs';
const base='Analysis/Goal70_2026-11-08',dir=base+'/Brighton_Ipswich_Comparison';
fs.mkdirSync(dir,{recursive:true});
const candidates=csv(base+'/all_models.csv').filter(r=>['5795506','5795510'].includes(r.id));
const teamIds=[...new Set(candidates.flatMap(r=>[r.home_id,r.away_id]))];
await pool(teamIds,async id=>save(dir+'/team_'+id+'.json',await fetch(api('teams?id='+id))));
await pool(candidates,async r=>save(dir+'/match_'+r.id+'.json',await fetch(api('matchDetails?matchId='+r.id))));
const recent=[];
for(const tid of teamIds){const t=read(dir+'/team_'+tid+'.json');if(!t.ok)throw Error('Team unavailable '+tid);
 for(const f of (t.data.fixtures?.allFixtures?.fixtures??[]).filter(f=>f.status?.finished&&!f.status?.cancelled&&!f.status?.awarded&&Date.parse(f.status.utcTime)<Date.parse('2026-11-08T00:00:00+02:00')&&!/friendly/i.test(f.tournament?.name??'')&&f.tournament?.leagueId!==489).sort((a,b)=>Date.parse(b.status.utcTime)-Date.parse(a.status.utcTime)).slice(0,10))recent.push({team_id:tid,id:String(f.id),date:f.status.utcTime,competition:f.tournament?.name,home:f.home?.name,away:f.away?.name});
}
await pool([...new Set(recent.map(r=>r.id))],async id=>{if(!fs.existsSync(base+'/MatchData/'+id+'.json'))await match(id,dir+'/Recent');});
const rows=recent.map(r=>{const p=base+'/MatchData/'+r.id+'.json',d=read(fs.existsSync(p)?p:dir+'/Recent/'+r.id+'.json');return {...r,...observation(d),source:d.source};});
writeCsv(dir+'/last10_official.csv',rows);
const history=read(base+'/historical_observations.json');
const contexts=candidates.map(r=>{const d=read(dir+'/match_'+r.id+'.json');return {id:r.id,source:d.source,verified_at_utc:d.verified_at_utc,model:r,status:d.data?.header?.status,identity_ok:String(d.data?.general?.matchId)===r.id,info:d.data?.content?.matchFacts?.infoBox,lineup:d.data?.content?.lineup,table:d.data?.content?.table,weather:d.data?.content?.weather,recent:rows.filter(x=>[r.home_id,r.away_id].includes(x.team_id))};});
save(dir+'/comparison.json',{recorded_at_ro:ro(),contexts});
console.log(JSON.stringify({recorded_at_ro:ro(),rows:rows.length,teams:teamIds.length,missing:rows.filter(r=>!r.goal_result).length,historyType:Array.isArray(history)?'array':Object.keys(history),contexts:contexts.map(x=>({id:x.id,status:x.status,lineupType:x.lineup?.lineupType,teams:['homeTeam','awayTeam'].map(s=>{const t=x.lineup?.[s];return {name:t?.name,coach:t?.coach?.name,formation:t?.formation,unavailable:t?.unavailable?.map(p=>({name:p.name,reason:p.unavailability}))};})}))},null,2));
