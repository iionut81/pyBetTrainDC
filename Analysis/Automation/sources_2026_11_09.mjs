import fs from 'node:fs';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {read,csv,save,writeCsv,fetch,pool,ro} from './research_2026_11_09.mjs';
const exec=promisify(execFile),b='Analysis/Goal70_2026-11-09';
// Restore every original prediction field of the first three fixtures.
for(const market of ['goal','corner']){
 const dir=market==='goal'?b:'Analysis/Corners1H_2026-11-09';
 const rows=csv(dir+'/all_models.csv'),old=csv(b+'/Revisions/initial_three_fixture_scan/'+market+'_models.csv');
 for(const r of rows){const prior=old.find(x=>x.id===r.id);if(prior)Object.assign(r,prior);}
 writeCsv(dir+'/all_models.csv',rows);writeCsv('Analysis/Selection_History/2026-11-09_'+(market==='goal'?'goal70':'corners1h')+'_candidates.csv',rows);
}
const sources=[
 ['uefa2026','https://www.uefa.com/nationalassociations/uefarankings/?year=2026'],
 ['nff_postponement','https://www.fotball.no/turneringer/eliteserien/2026/kristiansund---valerenga-utsatt/'],
 ['vif_health','https://www.vif-fotball.no/nyheter/sykdomsutbrudd-i-valerenga'],
 ['vif_reschedule','https://www.vif-fotball.no/nyheter/kristiansund-kampen-flyttes-til-mandag'],
 ['kbk_schedule','https://www.kristiansundbk.no/lag/import/tournament/eliteserien-2026/season/fotballsesongen-2026/match/round-26-kristiansund-x-valerenga'],
 ['sanse_preview','https://www.realsociedad.eus/es/noticias/detalle/nuevo-desafio-1'],
 ['sanse_press','https://www.realsociedad.eus/es/noticias/detalle/siguela-en-directo-634'],
 ['sanse_schedule','https://www.realsociedad.eus/eu/taldea/hurrengo-partidak/sanse'],
 ['sabadell_risks','https://www.mundodeportivo.com/futbol/sabadell/20261106/1004247992/claves-sabadell-frenar-sanse.html'],
 ['sabadell_press_efe','https://www.infobae.com/espana/agencias/2026/11/06/ferran-costa-sabadell-defiende-que-se-tendria-que-haber-suspendido-la-jornada/'],
 ['laliga_sabadell','https://www.laliga.com/es-GB/partido/temporada-2026-2027-laliga-hypermotion-ce-sabadell-r-sociedad-b-13'],
 ['botosani_schedule','https://fcbt.ro/'],
 ['botosani_press','https://fcbt.ro/video-conferinta-de-presa-inainte-de-partida-dintre-fc-botosani-si-otelul-galati-4/'],
 ['botosani_coach','https://botosani24.ro/marius-croitoru-fc-botosani-avem-un-meci-dificil-cu-otelul-galati-dar-vrem-sa-castigam-146969.html'],
 ['laspalmas_reschedule','https://www.udlaspalmas.es/noticias/el-partido-j13-laspalmaseldense-se-disputara-manana-lunes-9-de-noviembre-a-las-1930-horas-hora-canaria-en-el-estadio-gran-canaria'],
 ['brighton_result','https://www.brentfordfc.com/en/news/match-reports-brighton-and-hove-albion-2-brentford-0-premier-league-08-11-2026']
];
const evidence=[];fs.mkdirSync(b+'/Sources',{recursive:true});
await pool(sources,async([key,url])=>{try{const {stdout}=await exec('curl.exe',['--compressed','-L','--max-time','25','-sS','-w','\nHTTP_STATUS:%{http_code}',url],{maxBuffer:12*1024*1024,windowsHide:true});const ix=stdout.lastIndexOf('HTTP_STATUS:');fs.writeFileSync(b+'/Sources/'+key+'.html',stdout.slice(0,ix));evidence.push({key,url,status:stdout.slice(ix+12).trim(),verified_at_ro:ro()});}catch(e){evidence.push({key,url,error:String(e.message).slice(0,180),verified_at_ro:ro()});}},4);
save(b+'/sources.json',evidence);
const ctx=read(b+'/context_summary.json').matches;
for(const m of ctx){const s=m.stadium;save(b+'/Weather/'+m.id+'.json',await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${s.lat}&longitude=${s.long}&hourly=temperature_2m,precipitation_probability,wind_speed_10m&start_date=2026-11-09&end_date=2026-11-09&timezone=Europe%2FBucharest`));}
// Reconcile the program discrepancy explicitly.
const coverage=csv(b+'/universe_coverage.csv');
for(const c of coverage){
 c.fixtures_after_reconciliation=c.league==='esp.2'?'2':c.identified_fixtures;
 if(c.league==='esp.2'){c.coverage='official_and_ESPN_reschedule_vs_stale_FotMob';c.limit='Las Palmas–Eldense added from official club+ESPN; FotMob status still cancelled on8Nov';}
 if(c.league==='rou.1'){c.coverage='official_club_plus_FotMob';c.limit='ESPN empty conflicts with official20:00RO and FotMob';}
}
writeCsv(b+'/universe_coverage.csv',coverage);
const global=read(b+'/fixtures_global.json');
save(b+'/cups_coverage.json',{verified_at_ro:ro(),leagues:global.leagues.filter(l=>/cup|champions|europa|conference|copa/i.test(l.name)).map(l=>({name:l.name,id:l.primaryId??l.id,matches:l.matches.map(m=>({id:m.id,home:m.home.name,away:m.away.name,status:m.status}))})),limitation:'Only global provider inventory; no assertion of exhaustive all-world cups. Outside-scope fixtures retained in exclusions.csv.'});
console.log(JSON.stringify({sources:evidence.map(x=>({key:x.key,status:x.status,error:x.error})),coverage_corrected:true,original_three_snapshots_restored:true}));
