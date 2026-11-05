import fs from 'node:fs';
import {read,csv,writeCsv,ro,save} from './research_2026_11_05.mjs';
const R='Analysis/Review_2026-11-02',current='Analysis/Review_2026-11-04';
const source='https://www.fotmob.com/api/data/matchDetails?matchId=5868523 | https://www.laliga.com/es-NG/partido/temporada-2026-2027-laliga-hypermotion-cd-tenerife-fc-andorra-12 | https://as.com/futbol/segunda/el-tenerife-andorra-se-da-por-finalizado-no-se-tendra-que-reanudar-f202611-n/';
for(const market of ['goal70','corners1h']){
 const p=`${R}/${market}_outcomes.csv`,backup=`${R}/Revisions/2026-11-05_before_official_conclusion/${market}_outcomes.csv`;
 fs.mkdirSync(backup.slice(0,backup.lastIndexOf('/')),{recursive:true});if(!fs.existsSync(backup))fs.copyFileSync(p,backup);
 const rows=csv(p),r=rows.find(x=>(x.fotmob_id||x.id)==='5868523');
 if(r){r.result='success';r.reason='official_conclusion_2_2_confirmed_LaLiga; FirstHalf_preserved; previous_abandoned_status_archived';r.score_ft='2-2';r.result_source=source;r.result_verified_at_ro=ro();if(market==='goal70'){r.first_goal_minute='3';r.first_goal_base='3';r.label='1';r.brier=(Number(r.p_raw)-1)**2;r.log_loss=-Math.log(Number(r.p_raw));}else r.actual_corners_1h=2;writeCsv(p,rows);}
}
const note=`\n\n## Reconciliere5 noiembrie\n\n${ro()}: LaLiga afiseaza Finalizado2-2; raportulAS4 noiembrie17:36CET descrie hotararea comisiei de a incheia partida, fara reluare. FotMob confirma acumfinished=true, cancelled=false, primul gol3 si FirstHalf1-1=2, reconciliat cuSecondHalf siAll. Rezultatele Tenerife-Andorra devin success pe ambele piete, observatii sportive; nu este decontare de bookmaker. Vechile stari amanat/abandonat sunt arhivate inRevisions/2026-11-05_before_official_conclusion, probabilitatile prospective raman intacte. [LaLiga](https://www.laliga.com/es-NG/partido/temporada-2026-2027-laliga-hypermotion-cd-tenerife-fc-andorra-12), [decizie relatataAS](https://as.com/futbol/segunda/el-tenerife-andorra-se-da-por-finalizado-no-se-tendra-que-reanudar-f202611-n/). Bilant actualizat2 noiembrie: Gol7023/27, cornere22/26; Radnicki-Zeleznicar ramane reprogramat25 noiembrie, oraTBD, exclus.\n`;
for(const p of [R+'/REVIEW.md',current+'/REVIEW.md',current+'/LESSONS_FOR_NEXT_SCAN.md'])fs.appendFileSync(p,note);
const h=read(current+'/handoff.json');h.review_handoff_at_ro=ro();h.missing_data=['Radnicki-Zeleznicar reprogramat25 noiembrie, oraTBD; cauzele tactice ale esecurilor nu sunt demonstrate.'];save(current+'/handoff.json',h);
