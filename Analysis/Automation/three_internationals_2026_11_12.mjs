import fs from 'node:fs';
import path from 'node:path';
import {read,csv,save,writeCsv,pool,fetch,api,extract,observation,ro} from './research_2026_11_12.mjs';
const base='Analysis/Goal70_2026-11-12/Three_Internationals_Comparison',cutoff=Date.parse('2026-11-12T00:00:00+02:00');
const fixtures=[{id:'5181796',home:'Turkiye',away:'Belgium',home_id:'6595',away_id:'8263'},{id:'5181797',home:'Italy',away:'France',home_id:'8204',away_id:'6723'},{id:'5181858',home:'England',away:'Croatia',home_id:'8491',away_id:'10155'}];
async function cached(url,file){if(fs.existsSync(file)){const r=read(file);if(r.ok)return r;}const r=await fetch(url);save(file,r);return r;}
async function probe(){
 if(!fs.existsSync(base+'/request.json'))save(base+'/request.json',{at_ro:ro(),request:'Deep GOAL_BY_70 analysis for Turkiye–Belgium, Italy–France, England–Croatia; no bookmaker odds, no inferred user selection',previous_review:'Analysis/Review_2026-11-11/LESSONS_FOR_NEXT_SCAN.md',existing_eligibility:'Current edition has4overall/2venue, below5/3; do not alter official model eligibility or frozen predictions'});
 await pool(fixtures,async f=>{const r=await cached(api('matchDetails?matchId='+f.id),base+'/Current/'+f.id+'.json');if(!r.ok)return;const d=r.data;if(String(d.general?.matchId)!==f.id)throw Error('Fixture identity mismatch');const stadium=d.content?.matchFacts?.infoBox?.Stadium;save(base+'/Status/'+f.id+'.json',extract(r,f.id));
  if(stadium?.lat&&stadium?.long){const url=`https://api.open-meteo.com/v1/forecast?latitude=${stadium.lat}&longitude=${stadium.long}&hourly=temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m&start_date=2026-11-12&end_date=2026-11-12&timezone=UTC`;await cached(url,base+'/Weather/'+f.id+'.json');}
 },3);
 const teams=[...new Map(fixtures.flatMap(f=>[['home',f.home_id,f.home],['away',f.away_id,f.away]]).map(([side,id,name])=>[id,{id,name}])).values()];
 await pool(teams,async t=>{await cached(api('teams?id='+t.id),base+'/Teams/'+t.id+'.json');},4);
 console.log(JSON.stringify({at_ro:ro(),fixtures:fixtures.map(f=>{const r=read(base+'/Current/'+f.id+'.json');return {id:f.id,ok:r.ok,kickoff:r.data?.general?.matchTimeUTCDate??r.data?.header?.status?.utcTime,status:r.data?.header?.status,stadium:r.data?.content?.matchFacts?.infoBox?.Stadium,coach:r.data?.content?.lineup?.coach};}),teams:teams.map(t=>{const r=read(base+'/Teams/'+t.id+'.json');return {...t,ok:r.ok,fixtureKeys:Object.keys(r.data?.fixtures??{}),overviewKeys:Object.keys(r.data?.overview??{}),allFixturesKeys:Object.keys(r.data?.fixtures?.allFixtures??{})};})}));
}
if(process.argv[2]==='probe-three')await probe();
async function history(){
 const selections=[];
 for(const id of [...new Set(fixtures.flatMap(f=>[f.home_id,f.away_id]))]){
  let d=read(base+'/Teams/'+id+'.json').data,all=[...d.fixtures.allFixtures.fixtures],previous=d.fixtures.previousFixturesUrl;
  const eligible=()=>all.filter(m=>m.status.finished&&!m.status.cancelled&&!m.status.awarded&&Date.parse(m.status.utcTime)<cutoff&&!/friendl/i.test(m.tournament.name)).sort((a,b)=>Date.parse(b.status.utcTime)-Date.parse(a.status.utcTime));
  for(let p=0;eligible().length<10&&previous&&p<4;p++){const r=await cached(previous,base+'/Teams/'+id+'_previous_'+p+'.json');if(!r.ok)break;let x=r.data;console.log('Pagination',id,Object.keys(x));const batch=x.fixtures??x.allFixtures?.fixtures??[];if(!Array.isArray(batch))throw Error('Unknown pagination schema');all.push(...batch);previous=x.previousFixturesUrl;}
  selections.push({team_id:id,fixtures:[...new Map(eligible().map(m=>[m.id,m])).values()].slice(0,10)});
 }
 save(base+'/last10_selection.json',selections);
 const unique=[...new Map(selections.flatMap(t=>t.fixtures).map(m=>[m.id,m])).values()];
 await pool(unique,async m=>{const dest=base+'/History/'+m.id+'.json',prior='Analysis/Goal70_2026-11-12/International/MatchData/'+m.id+'.json';if(fs.existsSync(dest)&&read(dest).ok)return;if(fs.existsSync(prior)&&read(prior).ok){save(dest,{...read(prior),reused_from:prior});return;}const r=await cached(api('matchDetails?matchId='+m.id),base+'/RawHistory/'+m.id+'.json');save(dest,extract(r,m.id));},5);
 const rows=selections.flatMap(t=>t.fixtures.map(m=>{const d=read(base+'/History/'+m.id+'.json'),o=observation(d);let valid=!!o.goal_result;const extended=d.goals?.some(g=>g.minute>90);if(!valid&&extended&&d.status?.finished){valid=d.goals.every(g=>Number.isFinite(g.minute)&&g.minute<=120)&&['home','away'].every((s,i)=>d.goals.filter(g=>g.side===s).length===Number(d.teams[i].score));}
 const goals=valid?d.goals.filter(g=>g.minute<=90).sort((a,b)=>a.minute-b.minute||a.added-b.added):[],first=goals[0],side=String(m.home.id)===t.team_id?'home':'away';
 return {team_id:t.team_id,id:String(m.id),date:m.status.utcTime,competition:m.tournament.name,venue:side,home:m.home.name,away:m.away.name,score:m.status.scoreStr,verified:valid,first_goal:first?.display??'',success70:valid?!!first&&first.minute<=70:'',own70:valid?goals.filter(g=>g.side===side&&g.minute<=70).length:'',conceded70:valid?goals.filter(g=>g.side!==side&&g.minute<=70).length:'',early_red:valid?d.cards.filter(c=>/red/i.test(c.card)&&c.time<(first?.minute??71)).map(c=>JSON.stringify(c)).join('|'):'',method:extended?'Goals reconciled to extra-time score; only minutes <=90 retained':o.goal_reason,source:d.source};}));
 writeCsv(base+'/last10_official.csv',rows);save(base+'/last10_summary.json',selections.map(t=>{let a=rows.filter(r=>r.team_id===t.team_id),v=a.filter(r=>r.verified);return {team_id:t.team_id,expected:a.length,verified:v.length,success:v.filter(r=>r.success70).length,late:v.filter(r=>!r.success70&&r.first_goal).length,nil:v.filter(r=>!r.first_goal).length,own70:v.reduce((s,r)=>s+r.own70,0),conceded70:v.reduce((s,r)=>s+r.conceded70,0),matches:a};}));
 console.log(JSON.stringify(read(base+'/last10_summary.json')));
}
if(process.argv[2]==='history-three')await history();
export {base,fixtures,cutoff,cached};
