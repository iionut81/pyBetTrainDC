import {execFile} from 'node:child_process';import {promisify} from 'node:util';
import {write,stamp,pool} from '../daily_2026_11_18.mjs';
const run=promisify(execFile),b='Analysis/Goal70_2026-11-18/Supplemental';
await pool([
 ['footynomad','https://footynomad.com/fixtures/2026-11-18'],
 ['footballfixtures','https://www.footballfixtures.org/fixtures/2026-11-18'],
 ['footballmania','https://football-mania.com/agenda/2026-11-18'],
 ['sportpremi','https://sportpremi.com/football/2026-11-18'],
 ['UEFA_WU17','https://www.uefa.com/womensunder17/fixtures-results/'],
 ['UEFA_WU19','https://www.uefa.com/womensunder19/fixtures-results/'],
 ['FIFA_U17','https://www.fifa.com/en/tournaments/mens/u17worldcup/qatar-2026/articles/score-fixtures-match-schedule']
],async([name,url])=>{try{let {stdout}=await run('curl.exe',['-L','--max-time','25','-sS','-w','\nHTTP_STATUS:%{http_code}',url],{maxBuffer:12e6,windowsHide:true});write(`${b}/${name}.html`,stdout);write(`${b}/${name}.source.json`,{url,at_ro:stamp(),size:stdout.length,status:stdout.split('HTTP_STATUS:').at(-1).trim()});console.log(name,stdout.length);}catch(e){write(`${b}/${name}.error.json`,{url,at_ro:stamp(),error:String(e).slice(0,500)});}},4);
