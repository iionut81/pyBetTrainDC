import fs from 'node:fs';
import {read,csv,save,writeCsv,ro} from './research_2026_11_09.mjs';
const R='Analysis/Review_2026-11-08',b='Analysis/Goal70_2026-11-09',stamp=ro();
const lists={goal:csv(R+'/goal70_outcomes.csv'),corner:csv(R+'/corners1h_outcomes.csv')};
const metrics=rows=>({total:rows.length,modeled:rows.filter(r=>r.p_raw).length,success:rows.filter(r=>r.result==='success').length,fail:rows.filter(r=>r.result==='fail').length,unscored:rows.filter(r=>!['success','fail'].includes(r.result)).map(r=>({id:r.id,home:r.home,away:r.away,result:r.result,reason:r.reason}))});
const p=R+'/REVIEW.md';let s=fs.readFileSync(p,'utf8');
s=s.split('\n').map(line=>{
 if(line.startsWith('Brighton?Brentford:'))return 'Brighton–Brentford: FT2–0, primul gol15, succes, confirmat și de raportul oficialBrentford. Girona–Almería: FT0–1, FirstHalf5cornere, succes; FotMob cu reconciliere pe reprize. Alegerile sunt distincte de candidații analistului și nu presupun pariuri.';
 if(line.startsWith('Univers verificat și lipsuri:')){const key=line.includes('"modeled":103')?'goal':'corner';return 'Univers verificat și lipsuri: '+JSON.stringify(metrics(lists[key]))+'. Separat modele prospective: '+JSON.stringify(metrics(lists[key].filter(r=>r.p_raw)))+'.';}
 if(line.startsWith('|')&&line.includes('Las Palmas–Eldense'))return line.replace('cancelled','postponed_not_scored');
 return line;
}).join('\n');
s+=`\n## Bilanț comparabil și restante\n\nGol70:87/102 verificate (85,29%),15fail; și în cohorta cu model87/102. Cornere:87/99 (87,88%),12fail; separat modele72/81 (88,89%),9fail. Trei eșecuri cornere nu aveau model prospectiv: nu li se inventează probabilități. Alegerile utilizatorului1/1 pe fiecare piață, distinct de lista completă și de finaliști. Nacional–Viseu, restant din7noiembrie și jucat8noiembrie: primul86FAILGol70,FirstHalf6SUCCEScornere,FT1–0; arhiva originală7noiembrie actualizată separat cu sursa și momentul verificării.\n\nSursele retrospective nu devin predictori disponibili anterior. Exemplu de succes compatibil, fără validare: Girona5cornere corespunde seriei reduse, dar riscurile de presing și schimbări defensive existau; Brighton15 corespunde argumentelor ofensive, dar scorurile mari și sensibilitatea fără roșii rămân avertismente pentru verificările viitoare.\n`;
fs.writeFileSync(p,s);
const h=read(R+'/handoff.json');h.metrics={goal:metrics(lists.goal),corner:metrics(lists.corner)};h.editorial_reconciled_at_ro=stamp;h.missing_data=['3FirstHalfCyprus8Nov pending','4olderCyprusFirstHalf unavailable; Serbia not final','Independent FirstHalf corroboration incomplete'];save(R+'/handoff.json',h);
const pick='Analysis/Corners1H_2026-11-09/prospective_selection.csv',rows=csv(pick);rows[0].kickoff_verified_at_ro=new Date(Date.parse(rows[0].kickoff_verified_at_ro)+7200000).toISOString().replace('Z','+02:00');writeCsv(pick,rows);
// Preserve the latest successful schedule verification in the coverage record.
const cov=csv(b+'/universe_coverage.csv');for(const r of cov)if(r.league==='esp.2'){r.coverage='official_ESPN_and_individual_FotMob_reconciled';r.limit='Global FotMob still omitted rescheduled Las Palmas–Eldense; individual endpoint now confirms9Nov19:30UTC';}writeCsv(b+'/universe_coverage.csv',cov);
const cups=read(b+'/cups_coverage.json');
for(const f of [b+'/EUROPE_ALL_MATCHES.md','Analysis/Corners1H_2026-11-09/ANALYSIS.md']){
 let text=fs.readFileSync(f,'utf8');
 text+='\n## Cupe și verificarea programului\n\nÎn programul global sunt inventariate FA Cup: PlymouthParkway–Berkhamsted; celelalte competiții numiteCup sunt listate cu identități și status încups_coverage.json și excluse explicit din universul campionatelor. Nu există model intern comparabil colectat pentru aceste echipe inferioare. EvenimentulUSLLeagueOne apare prin filtrul textualChampionship, dar nu este cupă europeană și rămâne exclus. Nicio proiecție peste competiții pentru cei4candidați de campionat. LasPalmas are acum status individualFotMob reconciliat cuESPNși clubul; doar lista globală a omis partida.\n';
 fs.writeFileSync(f,text);
}
save(b+'/close_evidence.json',{at_ro:stamp,review_metrics:h.metrics,cups: cups.leagues.map(l=>({name:l.name,matches:l.matches.length}))});
console.log(JSON.stringify({review_corrected:true,at_ro:stamp}));
