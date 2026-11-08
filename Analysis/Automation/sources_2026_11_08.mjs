import fs from 'node:fs';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {read,save,fetch,pool,ro} from './research_2026_11_08.mjs';
const exec=promisify(execFile),base='Analysis/Goal70_2026-11-08';
const sources=[
 ['uefa2026','https://www.uefa.com/nationalassociations/uefarankings/?year=2026'],
 ['psv_press','https://www.psv.nl/media/artikel/persconferentie-verwacht-weer-een-open-wedstrijd'],
 ['knvb_schedule','https://www.knvb.nl/competities/eredivisie/programma'],
 ['girona_press','https://www.gironafc.cat/es/noticias/estoy-convencido-de-que-el-equipo-dara-manana-la-mejor-version'],
 ['almeria_press','https://www.udalmeriasad.com/noticias/nos-enfrentamos-a-un-equipo-que-ha-jugado-champions'],
 ['almeria_squad','https://www.udalmeriasad.com/noticias/convocatoria-para-el-partido-en-montilivi'],
 ['almeria_defense','https://www.udalmeriasad.com/noticias/el-almeria-pondra-a-prueba-su-labor-defensiva-en-montilivi'],
 ['almeria_match','https://www.udalmeriasad.com/partidos/temporada-2026-2027-laliga-hypermotion-13-girona-fc-vs-ud-almeria-3438'],
 ['viking_schedule','https://www.vikingfotball.no/terminliste'],
 ['viking_results','https://www.vikingfotball.no/resultater'],
 ['monza_press','https://www.acmonza.com/it/news/la-conferenza-di-mister-juric-prima-di-monza-atalanta/'],
 ['monaco_schedule','https://www.asmonaco.com/fr/news/le-planning-des-rouge-et-blanc-avant-nordsjaelland-et-le-mans'],
 ['review_blackburn','https://www.skysports.com/football/blackburn-rovers-vs-west-ham-united/8067885432187075232'],
 ['review_excelsior','https://www.skysports.com/football/excelsior-vs-go-ahead-eagles/556791']
];
const evidence=[];fs.mkdirSync(base+'/Sources',{recursive:true});
await pool(sources,async([key,url])=>{
 try{const {stdout}=await exec('curl.exe',['--compressed','-L','--max-time','25','-sS','-w','\nHTTP_STATUS:%{http_code}',url],{maxBuffer:15*1024*1024,windowsHide:true});const ix=stdout.lastIndexOf('HTTP_STATUS:');const status=stdout.slice(ix+12).trim();fs.writeFileSync(base+'/Sources/'+key+'.html',stdout.slice(0,ix));evidence.push({key,url,status,verified_at_ro:ro(),path:base+'/Sources/'+key+'.html'});}catch(e){evidence.push({key,url,error:String(e.message).slice(0,150),verified_at_ro:ro()});}
},4);
save(base+'/sources.json',evidence);
const ctx=read(base+'/context_summary.json').matches;
for(const id of ['5781796','5868530']){const m=ctx.find(m=>m.id===id),s=m.stadium;
 const url=`https://api.open-meteo.com/v1/forecast?latitude=${s.lat}&longitude=${s.long}&hourly=temperature_2m,precipitation_probability,wind_speed_10m&start_date=2026-11-08&end_date=2026-11-08&timezone=Europe%2FBucharest`;
 save(base+'/Weather/'+id+'.json',await fetch(url));
}
console.log(JSON.stringify(evidence.map(x=>({key:x.key,status:x.status,error:x.error}))));
