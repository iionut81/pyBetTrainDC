import fs from 'node:fs';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {pool,save,ro} from '../../Automation/research_2026_11_08.mjs';
const exec=promisify(execFile),dir='Analysis/Goal70_2026-11-08/Brighton_Ipswich_Comparison';
const sources=[
 ['brighton_preview','https://www.brightonandhovealbion.com/media-article/preview-hurzeler-hopes-chema-will-be-back-to-face-brentford'],
 ['brighton_recovery','https://www.brightonandhovealbion.com/media-article/hurzeler-we-have-to-be-ready-for-physical-test-against-brentford'],
 ['brentford_preview','https://www.brentfordfc.com/en/news/match-previews-brighton-and-hove-albion-v-brentford-premier-league-08-11-2026'],
 ['brentford_team_news','https://www.brentfordfc.com/en/news/first-team-news-brighton-v-brentford-premier-league-josh-dasilva'],
 ['ipswich_club_news','https://www.itfc.co.uk/news/2026/november/06/Gary-ONeil-early-team-news-Ipswich-Town-Bournemouth/'],
 ['ipswich_presser_report','https://t.twtd.co.uk/ipswich-town-news/53143/oneil-bournemouth-are-in-a-good-moment'],
 ['press_conference_index','https://fpldraft.me/premier-league-press-conferences']
];
const rows=[];
await pool(sources,async([key,url])=>{try{const {stdout}=await exec('curl.exe',['--compressed','-L','--max-time','25','-sS','-w','\nHTTP_STATUS:%{http_code}',url],{maxBuffer:20*1024*1024,windowsHide:true});const ix=stdout.lastIndexOf('HTTP_STATUS:');const status=stdout.slice(ix+12).trim();fs.writeFileSync(dir+'/'+key+'.html',stdout.slice(0,ix));rows.push({key,url,status,verified_at_ro:ro(),limitation:key==='ipswich_club_news'?'Readable extraction contains headline only; medical claims not independently confirmed from this body':''});}catch(e){rows.push({key,url,error:String(e.message).slice(0,150),verified_at_ro:ro()});}});
save(dir+'/sources.json',rows);console.log(JSON.stringify(rows.map(r=>({key:r.key,status:r.status,error:r.error}))));
