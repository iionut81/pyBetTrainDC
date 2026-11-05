import fs from 'node:fs';
import {read,csv,save,writeCsv,pool,fetch,api,match,observation,ro} from './research_2026_11_05.mjs';
const G='Analysis/Goal70_2026-11-05',C='Analysis/Corners1H_2026-11-05';
const goals=csv(G+'/all_models.csv'),corners=csv(C+'/all_models.csv'),fixtures=read(G+'/candidate_fixtures.json');
const shortlist=[...new Map([...goals.filter(x=>x.p_raw!=='').slice(0,3),...corners.filter(x=>x.p_raw!=='').slice(0,3)].map(x=>[x.id,x])).values()];
const teamIds=[...new Set(shortlist.flatMap(f=>[f.home_id,f.away_id]))];
await pool(teamIds,async id=>{
 const r=await fetch(api('teams?id='+id));if(!r.ok){save(G+'/Finalists/team_'+id+'.json',r);return;}
 const fixtures=(r.data.fixtures?.allFixtures?.fixtures??[]).filter(m=>m.status.finished&&!m.status.cancelled&&!m.status.awarded&&!/Friendl/i.test(m.tournament?.name??'')&&Date.parse(m.status.utcTime)<Date.parse('2026-11-05T00:00:00+02:00')).sort((a,b)=>Date.parse(b.status.utcTime)-Date.parse(a.status.utcTime)).slice(0,10);
 save(G+'/Finalists/team_'+id+'.json',{source:r.source,verified_at_utc:r.verified_at_utc,fixtures});
 await pool(fixtures,async f=>{await match(f.id,G+'/Finalists/MatchData');},3);
 writeCsv(G+'/Finalists/'+id+'_last10.csv',fixtures.map(f=>{const d=read(G+'/Finalists/MatchData/'+f.id+'.json'),o=observation(d);return {team_id:id,id:f.id,date:f.status.utcTime,competition:f.tournament.name,home:f.home.name,away:f.away.name,score_ft:o.score_ft,first_goal_minute:o.first_goal_minute,result:o.goal_result??o.result,actual_corners_1h:o.actual_corners_1h,corner_result:o.corner_result??o.result,source:d.source};}));
},3);
await pool(fixtures,async f=>{await match(f.id,G+'/LiveStatus');},5);
writeCsv(G+'/live_status.csv',fixtures.map(f=>{const d=read(G+'/LiveStatus/'+f.id+'.json');return {id:f.id,home:f.home,away:f.away,kickoff_utc:d.status?.utcTime??'',unstarted:d.ok&&!d.status.started&&!d.status.finished&&!d.status.cancelled&&Date.parse(d.status.utcTime)>Date.now(),identity_agrees:d.ok&&String(d.general.homeTeam.id)===f.home_id&&String(d.general.awayTeam.id)===f.away_id,source:d.source,verified_at_utc:d.verified_at_utc};}));
// Independently count goals in domestic schedules for all shortlist teams.
const slugs=[...new Set(shortlist.flatMap(f=>[f.home_league,f.away_league]))];
await pool(slugs,async slug=>{save(G+'/Crosscheck/'+slug+'.json',await fetch(`https://site.api.espn.com/apis/site/v2/sports/soccer/${slug}/scoreboard?dates=2026&limit=1000`));});
const aliases={'heartofmidlothian':'hearts','unionstgilloise':'unionsaintgilloise','tsghoffenheim':'hoffenheim','sportingbraga':'braga','agf':'aarhus','asmonaco':'monaco','fcnordsjaelland':'nordsjaelland','afcbournemouth':'bournemouth','spartapraha':'spartaprague','celta':'celtavigo','fcstgallen':'stgallen','fcwinterthur':'winterthur','fcbasel1893':'basel','grasshoppers':'grasshopper','fczurich':'zurich','fcstgallen1879':'stgallen'};
function nm(s){s=s.normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase().replace(/[^a-z0-9]/g,'');return aliases[s]??s;}
const history=read(G+'/historical_observations.json'),checks=[];
for(const m of history.filter(m=>teamIds.some(id=>[m.home_id,m.away_id].includes(id)))){
 const es=read(G+'/Crosscheck/'+m.league+'.json');const found=(es.data?.events??[]).filter(e=>{const c=e.competitions[0],h=c.competitors.find(x=>x.homeAway==='home'),a=c.competitors.find(x=>x.homeAway==='away');return e.date.slice(0,10)===m.date.slice(0,10)&&nm(h.team.displayName)===nm(m.home)&&nm(a.team.displayName)===nm(m.away);});
 if(found.length!==1){checks.push({id:m.id,home:m.home,away:m.away,status:'independent_identity_unresolved',source:es.source});continue;}
 const e=found[0],c=e.competitions[0],h=c.competitors.find(x=>x.homeAway==='home'),a=c.competitors.find(x=>x.homeAway==='away');const gs=(c.details??[]).filter(g=>g.scoringPlay&&!g.shootout).map(g=>({minute:parseInt(g.clock.displayValue),display:g.clock.displayValue,side:String(g.team.id)===String(h.id)?'home':'away'}));
 const valid=gs.length===Number(h.score)+Number(a.score)&&gs.every(g=>Number.isFinite(g.minute)),gh=gs.filter(g=>g.side==='home'&&g.minute<=70).length,ga=gs.filter(g=>g.side==='away'&&g.minute<=70).length;
 const row={id:m.id,espn_id:e.id,home:m.home,away:m.away,date:m.date,goal_status:valid?(gh===m.home70&&ga===m.away70?'agree':'conflict'):'timeline_incomplete',fotmob_home70:m.home70,fotmob_away70:m.away70,espn_home70:valid?gh:'',espn_away70:valid?ga:'',source:es.source};
 // Primary corners finalist: validate complete event timelines rather than counting absent commentary as zero.
 if([m.home_id,m.away_id].some(id=>[corners[0].home_id,corners[0].away_id].includes(id))){
 const r=await fetch(`https://site.api.espn.com/apis/site/v2/sports/soccer/${m.league}/summary?event=${e.id}`);if(r.ok){const d=r.data,plays=[...new Map((d.commentary??[]).filter(x=>x.play?.id).map(x=>[x.play.id,x.play])).values()],cs=plays.filter(p=>p.type?.type==='corner-awarded'),teams=d.boxscore?.teams??[];
 const complete=teams.length===2&&teams.every(t=>{const ft=t.statistics?.find(x=>x.name==='wonCorners')?.displayValue;return ft!==undefined&&Number(ft)===cs.filter(p=>p.team?.id===t.team.id).length;})&&(d.commentary??[]).some(x=>/First Half ends/i.test(x.text??''));
 const nh=cs.filter(p=>p.period?.number===1&&String(p.team?.id)===String(h.id)).length,na=cs.filter(p=>p.period?.number===1&&String(p.team?.id)===String(a.id)).length;
 row.corner_status=complete?(nh===m.home_cf&&na===m.away_cf?'agree':'conflict'):'timeline_incomplete';row.espn_home1h=complete?nh:'';row.espn_away1h=complete?na:'';row.fotmob_home1h=m.home_cf;row.fotmob_away1h=m.away_cf;
 save(C+'/ESPN/'+e.id+'.json',{source:r.source,verified_at_utc:r.verified_at_utc,id:e.id,fotmob_id:m.id,timeline_complete:complete,first_half:complete?[nh,na]:null,corners:cs.map(p=>({minute:p.clock?.displayValue,period:p.period?.number,team:p.team?.id})),boxscore:teams.map(t=>({id:t.team.id,name:t.team.displayName,corners:t.statistics?.find(x=>x.name==='wonCorners')?.displayValue}))});
 }else row.corner_status=r.error;
 }
 checks.push(row);
}
writeCsv(G+'/independent_provider_checks.csv',checks);
const cov=csv(G+'/universe_coverage.csv');for(const c of cov){const s=read(G+'/Schedules/'+c.league+'.json');const valid=s.ok&&s.fixtures?.length>0;const daily=valid?s.fixtures.filter(f=>new Date(f.date).toLocaleDateString('en-CA',{timeZone:'Europe/Bucharest'})==='2026-11-05'):[];c.season_schedule_daily=valid?daily.length:'';c.season_schedule_source=s.source;c.coverage=valid&&daily.length===Number(c.identified_fixtures)?'global_and_valid_season_schedule_agree':valid?'schedule_count_conflict':'season_schedule_unavailable';if(c.coverage==='global_and_valid_season_schedule_agree')c.limit=c.espn_supported==='true'?'':'ESPN unsupported; two FotMob schedules agree (same provider)';}
writeCsv(G+'/universe_coverage.csv',cov);
console.log(JSON.stringify({shortlist:shortlist.map(f=>f.home+' - '+f.away),teams:teamIds.length,live:csv(G+'/live_status.csv').filter(x=>x.unstarted==='true'&&x.identity_agrees==='true').length,checks:checks.reduce((a,r)=>(a[r.corner_status??r.goal_status??r.status]=(a[r.corner_status??r.goal_status??r.status]||0)+1,a),{}),coverage:cov.reduce((a,r)=>(a[r.coverage]=(a[r.coverage]||0)+1,a),{})},null,2));
