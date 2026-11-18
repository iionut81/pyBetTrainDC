import {execFile} from 'node:child_process';import {promisify} from 'node:util';import {write,pool,stamp} from '../daily_2026_11_18.mjs';
const run=promisify(execFile),B='Analysis/Goal70_2026-11-18';
const sources=[
 ['portugalete_calendar','https://clubportugalete.eus/primer-equipo/calendario/'],
 ['portugalete_coach','https://clubportugalete.eus/primer-equipo/cuerpo-tecnico/'],
 ['tenerife_preview','https://www.clubdeportivotenerife.es/noticias/alvaro-cervera-en-la-rueda-de-prensa-previa-al-debut-en-la-presente-edicion-copera'],
 ['tenerife_week','https://www.clubdeportivotenerife.es/noticias/plan-semanal-para-los-encuentros-de-la-copa-del-rey-mapfre-y-laliga-hypermotion'],
 ['cartagena_preview','https://www.fccartagena.es/noticias/duelo-cartagenero-en-la-copa-del-rey'],
 ['badajoz_club','https://clubdeportivobadajoz.es/'],
 ['rfef_kickoffs','https://rfef.es/sites/default/files/designaciones_copa_sm_el_rey_-_temp_2026-27_-_eliminatoria_1_miercoles_18.pdf?cb=db636d87'],
 ['uefa_ranking','https://www.uefa.com/nationalassociations/uefarankings/country/?year=2026'],
 ['fai_result','https://www.fai.ie/latest/report-mu21-play-ireland-2-2-finland-2-3-agg/']
];
await pool(sources,async([name,url])=>{let file=`${B}/Context/${name}.${name==='rfef_kickoffs'?'pdf':'html'}`;try{let {stdout}=await run('curl.exe',['-L','--max-time','25','-sS',url],{maxBuffer:10e6,encoding:name==='rfef_kickoffs'?'buffer':'utf8',windowsHide:true});if(name==='rfef_kickoffs'){const fs=await import('node:fs');fs.writeFileSync(file,stdout);}else write(file,stdout);write(file+'.source.json',{url,at_ro:stamp(),size:stdout.length});}catch(e){write(file+'.error.json',{url,at_ro:stamp(),error:String(e).slice(0,400)});}},4);
write(B+'/sources.json',{at_ro:stamp(),sources:sources.map(([name,url])=>({name,url})),program:['https://www.fotmob.com/api/data/matches?date=20261118','https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard?dates=20261118&limit=100','https://footynomad.com/fixtures/2026-11-18','https://football-mania.com/agenda/2026-11-18'],notes:'Full match and league endpoint URLs/timestamps in raw .source.json. No bookmaker prices or tipster percentages used.'});
console.log('Sources archived',sources.length);
