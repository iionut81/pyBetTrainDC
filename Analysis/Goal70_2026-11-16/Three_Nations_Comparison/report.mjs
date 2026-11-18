import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {read,save,writeCsv,pool,fetch,api,ro} from '../../Automation/research_2026_11_13.mjs';
const D='Analysis/Goal70_2026-11-16/Three_Nations_Comparison';
const data=read(D+'/comparison.json');
const protectedFiles=['Analysis/Selection_History/user_selections.csv','Analysis/Selection_History/selection_events.jsonl','Analysis/Goal70_2026-11-16/all_models.csv','Analysis/Selection_History/2026-11-16_goal70_candidates.csv'];
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const before=Object.fromEntries(protectedFiles.map(p=>[p,hash(p)]));
const names={'6383':'Grecia','8205':'Serbia','6717':'Elveția','8498':'Scoția','8265':'Kazahstan','8497':'Slovacia'};
const sources={
 slovak_absences:'https://futbalsfz.sk/muzi-a-weissov-tim-do-astany-bez-siedmich-hracov/',
 slovak_coach:'https://futbalsfz.sk/muzi-a-trener-vladimir-weiss-pred-zapasom-ligy-narodov-v-kazachstane-byt-plne-odhodlani-a-vela-behat/',
 greek_stakes:'https://www.epo.gr/el/arthra/psila-kefali-kai-niki-me-serbia',
 greek_venue:'https://www.epo.gr/el/arthra/ellada-serbia-enimerosi-media',
 greek_medical:'https://admin.novasports.gr/sport/podosfairo/event/nations-league/article/15232762/ethniki-ola-kala-me-tzoli-ektos-o-ioannidis-me-servia/',
 greek_coach:'https://www.in.gr/2026/11/15/sports/football/giovanovits-o-egoismos-mas-xaraktirizei-olous-antidrasi-me-servia',
 serbia_absences:'https://www.oraspor.gr/ethniki-ellados-me-syn-dyo-apousies-erchetai-i-servia/',
 scotland:'https://www.scottishfa.co.uk/en/news/scotland-round-off-uefa-nations-league-campaign-with-swiss-test-in-st-gallen',
 swiss_coach:'https://www.radiocentral.ch/news/schweiz/murat-yakin-kuendigt-wie-erwartet-viele-wechsel-an-166035948',
 kaz_red:'https://www.championat.ru/football/article-6698334-liga-nacij-uefa-2026-2027-obzor-matchej-5-go-tura-niderlandy-greciya-serbiya-germaniya-rejnders-13-noyabrya-2026.html'
};
// Extra time must be reconciled separately, never counted toward GOAL_BY_70.
const aet=data.comparisons.flatMap(c=>c.recent).find(r=>r.id==='4653854');
if(aet&&!aet.goal_result){
 const raw=read(D+'/Recent/'+aet.id+'.json');assert(raw.status.finished);
 for(const[i,side]of['home','away'].entries())assert.equal(raw.goals.filter(g=>g.side===side).length,Number(raw.teams[i].score));
 const regulation=raw.goals.filter(g=>g.minute<=90&&!/Extra/i.test(g.period||''));
 const first=[...regulation].sort((a,b)=>a.minute-b.minute||(a.added??0)-(b.added??0))[0];assert.equal(first.minute,10);
 assert.equal(regulation.filter(g=>g.side==='home').length,1);assert.equal(regulation.filter(g=>g.side==='away').length,1);
 aet.first_goal_minute=first.display;aet.goal_result='success';aet.goal_reason='all_goals_reconciled_to_AET_score_then_extra_time_excluded';aet.score_note='3-1 după prelungiri; 1-1 la 90; loviturile de departajare excluse';
 save(D+'/extra_time_validation.json',{id:aet.id,source:aet.source,all_goals:raw.goals,score_aet:aet.score_ft,score_90:'1-1',first_regulation_goal:first,production_helpers_unchanged:true});
}
const recent=data.comparisons.flatMap(c=>c.recent);
assert.equal(recent.length,46);assert(recent.every(r=>r.goal_result));assert(recent.every(r=>!/friendl/i.test(r.competition)));
writeCsv(D+'/last10_official.csv',recent);save(D+'/comparison.json',data);
const finals=[];
await pool(data.comparisons,async c=>{
 const r=await fetch(api('matchDetails?matchId='+c.model.id));assert(r.ok);
 assert.equal(String(r.data.general.matchId),c.model.id);
 const s=r.data.header.status;assert(!s.started&&!s.finished&&!s.cancelled);assert(Date.parse(s.utcTime)>Date.now());
 finals.push({id:c.model.id,source:r.source,status:s,verified_at_utc:r.verified_at_utc,at_ro:ro()});
});
save(D+'/final_status.json',finals);const updated=ro();
const order=['5181944','5181787','5181836'];
let md=`# Gol70: Kazahstan–Slovacia / Grecia–Serbia / Elveția–Scoția

Actualizat ${updated}. Toate sunt naționale de seniori masculin, UEFA Nations League 2026/27, etapa 6. Kazahstan–Slovacia începe la 17:00 RO; celelalte la 21:45 RO. Identitățile și statutul neînceput au fost reverificate separat la final.

**Prima preferință contextuală dintre cele trei: Kazahstan–Slovacia.** Apoi Elveția–Scoția și Grecia–Serbia. Aceasta este o comparație descriptivă, cu incertitudine importantă de lot; nu este o superioritate probabilistică demonstrată și nu înlocuiește candidatul modelat Grecia U21–Islanda U21 al dimineții.

**Decizia modelului salvat pentru toate trei: NO BET / neeligibil.** Fiecare echipă are cinci meciuri în ediția curentă, însă numai două în configurația relevantă acasă/deplasare, sub minimul 5/3. Nu inventăm procent, AR20 sau cotă echitabilă, nu adăugăm amicale/ediții vechi și nu relaxăm pragul fiindcă alegerea contextuală Slovacia–Feroe a reușit pe 13 noiembrie. Nicio alegere a utilizatorului nu este creată.

Evenimentul este minimum un gol valid al oricărei echipe la minut afișat ≤70, inclusiv 45+. Prelungirile eliminatorii și penaltyurile sunt excluse. Datele lipsă nu sunt zero.

## Comparația ediției curente

| Ordine contextuală | Partidă | Gol70 gazde | Gol70 oaspeți | Partide distincte favorabile | GF70 gazde acasă | GF70 oaspeți deplasare |
|---|---|---:|---:|---:|---:|---:|
`;
for(const[id,index]of order.map((id,index)=>[id,index])){
 const c=data.comparisons.find(c=>c.model.id===id),m=c.model;
 assert.equal(m.p_raw,'');assert.equal(m.eligible,'false');assert.equal(m.reason,'venue_sample_below_3');
 for(const k of['home_n','away_n'])assert.equal(Number(m[k]),5);
 for(const k of['home_venue_n','away_venue_n'])assert.equal(Number(m[k]),2);
 assert.equal(c.edition_history.length,9);assert.equal(new Set(c.edition_history.map(h=>h.id)).size,9);
 assert.equal(c.edition_history.filter(h=>h.success70).length,Number(m.first_goal_success));
 md+=`| ${index+1} | ${names[m.home_id]}–${names[m.away_id]} | ${m.home_wins}/5 | ${m.away_wins}/5 | ${m.first_goal_success}/${m.first_goal_n} | ${m.home_venue_gf}/2 | ${m.away_venue_gf}/2 |\n`;
}
md+=`
Turul comun este numărat o singură dată în uniunea fiecărui meci: 5+5−1=9. Frecvențele 9/9 și 7/9 nu sunt probabilități pentru astăzi. Coloanele GF70 reprezintă goluri marcate / meciuri, nu frecvența meciurilor cu gol. Nu reunim cele 46 de apariții recente într-un eșantion independent sau în model.

## 1. Kazahstan–Slovacia: cel mai bun istoric, dar lot profund schimbat

În ediția curentă ambele au 5/5 Gol70, iar uniunea are 9/9. Primele goluri au apărut la 8, 15, 17, 22, 26, 32, 38, 39 și 45+1. Nu există aici nici 0–0, nici deschidere după 70. Turul a fost 2–1, primul gol 26. Fiecare echipă a marcat ea însăși înainte de prag în toate cele cinci meciuri, fără ca numărul mare de goluri dintr-un singur scor să fie necesar pentru această constatare.

Kazahstan acasă: GF70 2/2=1,00; GA70 4/2=2,00. Slovacia în deplasare: GF70 4/2=2,00; GA70 1/2=0,50. Atacul slovac din deplasare este concentrat: trei dintre cele patru goluri au venit în Moldova; în Feroe a marcat unul. Nu tratăm 4–0 ca patru dovezi independente pentru deschiderea scorului.

Slovacia are 11 puncte și promovarea deja asigurată; Kazahstan are două puncte și caută prima victorie. Nu pretindem că oaspeții trebuie să atace pentru clasament. Federația slovacă confirmă că Škriniar, Hancko, Duda, Bero, Greif, Rusnák și Valjent nu fac deplasarea; Suslov este de asemenea indisponibil, iar Haraslín a rămas cu lotul. Această ruptură față de echipa care a produs istoricul este principalul risc. [SFZ](${sources.slovak_absences}).

Weiss spune că vor exista multe schimbări și cere mai multă alergare. Confirmă gazonul artificial și acoperișul închis la Astana Arena. Familiaritatea gazdelor cu suprafața contează; prognoza exterioară de aproximativ −4°C nu este o măsurătoare a condițiilor sub acoperiș. [Conferință oficială](${sources.slovak_coach}).

Samorodov este suspendat după eliminarea din Moldova, conform relatării meciului și listei de indisponibili; cronologia extrasă confirmă un cartonaș galben-roșu pentru Kazahstan, după deschiderea scorului. Satpaev este indisponibil în feed. [Cronologia relatată](${sources.kaz_red}). **XI probabil automat este inutilizabil ca listă actuală:** îl include pe Samorodov și numeroși slovaci care nu călătoresc. Nu transformăm 4-3-3 versus 4-3-3 din feed într-o formație confirmată.

Mecanism posibil: gazdele pot ataca o apărare fără relațiile obișnuite, iar Slovacia poate găsi spațiu la tranziție. Mecanismul contrar este la fel de concret: lipsa creatorilor slovaci și absența lui Samorodov pot reduce calitatea ultimelor pase și a finalizării. Slăbirea apărării nu înseamnă automat un meci cu gol rapid. Recuperarea scurtă după 13 noiembrie și călătoria sunt relevante, dar minutele și oboseala individuală nu sunt auditate complet.

**Verdict:** prima preferință descriptivă, nu un pick oficial de 90%. Riscul dominant este transferul istoricului unei echipe aproape complete către un XI foarte diferit. Scenariul nefavorabil: posesie fără ocazii clare și 0–0 până la 70.

## 2. Elveția–Scoția: atac propriu mai bun, dar rotație și apărare eficientă

Ambele au 4/5 Gol70; uniunea 7/9. Elveția are GF70 7/5 și GA70 1/5. Pe terenul relevant, Elveția GF70 3/2=1,50, Scoția 2/2=1,00. Cele două goluri scoțiene din deplasare vin ambele din Macedonia de Nord; celălalt meci în deplasare a fost 0–0 în Slovenia. Elveția a încheiat și ea 0–0 în Slovenia pe 13 noiembrie. Aceste contraexemple împiedică eticheta de selecție sigură.

Scoția trebuie să câștige și să aștepte ca Slovenia să piardă puncte în Macedonia de Nord. Elveția este deja promovată. Federația scoțiană confirmă și că turul 0–3 a fost influențat de eliminarea timpurie a lui Jack Hendry: nu îl tratăm ca o comparație normală 11 contra 11 pentru întregul joc. [Scottish FA](${sources.scotland}).

Yakin anunță multe schimbări. Akanji a părăsit lotul cu probleme fizice ușoare, Ndoye pentru gestionarea efortului; Xhaka, Muheim și Manzambi lipsesc. Freuler este anunțat căpitan la selecția 100. XI vechi din feed îi include încă pe Akanji/Ndoye și nu poate fi folosit ca lot actual. [Conferința Yakin, relatată de SDA](${sources.swiss_coach}).

Elveția este proiectată în 4-2-3-1, însă formula exactă rămâne probabilă. Pentru Scoția există diferență între 3-4-1-2 din feed și 3-4-2-1 sugerat de presa elvețiană: nu inventăm certitudine. Scoția poate folosi benzile și sosirile lui McGinn/McTominay, dar nevoia victoriei nu dovedește pressing neîntrerupt din minutul 1. Elveția poate controla mingea fără să forțeze, iar rotația poate slăbi atât apărarea, cât și construcția.

Teren propriu real, Berit Sitterstadion, St. Gallen. Prognoza din feed: aproximativ 8°C și ploaie ușoară, nu stare efectivă a gazonului. Scoțienii se deplasează după partida de vineri; odihna individuală nu este cuantificată. Principalul risc: Elveția controlează ritmul, iar Scoția creează insuficient contra unei apărări care a primit un singur gol înainte de prag în cinci meciuri.

## 3. Grecia–Serbia: miză pentru gazde, producție timpurie proprie slabă

Grecia are 3/5 Gol70; Serbia 5/5; uniune 7/9. Dar Grecia a marcat ea însăși numai două goluri până la 70 în cinci meciuri, ambele în 2–2 cu Olanda. Victoria cu Germania s-a deschis la 74; cu Germania acasă a fost 0–0. În victoria 2–1 din Serbia, golul de la 4 a fost al sârbilor, iar golurile grecești au venit după prag. Așadar, rezultatele bune la final nu demonstrează atac grec precoce.

Grecia acasă GF70 2/2=1,00; Serbia în deplasare GF70 1/2=0,50. Din cele nouă goluri primite de Grecia până la 70 în ediție, șapte provin din dezastrul recent 0–7 cu Olanda. Nu extrapolăm acea partidă ca ritm obișnuit contra Serbiei.

Grecia are opt puncte, Serbia zero și retrogradarea deja certă. EPO precizează că victoria Greciei, combinată cu Germania care nu învinge Olanda, asigură calificarea; egalitatea în trei nu o favorizează. Deci nu afirmăm că simpla victorie îi garantează calificarea în orice scenariu. [Federația greacă](${sources.greek_stakes}).

Ioannidis și Kourbelis sunt indisponibili; Tzolis a fost declarat disponibil după investigațiile de duminică seară. XI automat cu Ioannidis nu este actual. [Actualizarea medicală Novasports](${sources.greek_medical}). Jovanović cere reacție după eșec și îi recuperează după suspendare pe Saliakas și Zafeiris. [Conferința](${sources.greek_coach}). Pentru Serbia, Dragojević și Živković sunt raportați suspendați după eliminările contra Germaniei; nu confundăm absențele meciului anterior cu disponibilitatea actuală. [Actualizare Serbia](${sources.serbia_absences}).

Feedul propune 4-4-2 Grecia versus 5-3-2 Serbia, ambele neconfirmate. Grecia poate presa prin benzi și încerca să fixeze linia de cinci, dar Serbia poate apăra compact și contraataca. Nevoia victoriei nu compensează automat lipsa unui atacant și producția GF70 redusă. Panthessaliko din Volos este teren propriu, confirmat de [EPO](${sources.greek_venue}); prognoza din feed este aproximativ 15°C și averse slabe.

**De ce locul 3:** dependență mai mare de un gol advers și de interpretarea unui scor extrem. Scenariul nefavorabil este exact cel observat deja: Grecia în control, dar primul gol abia după 70 sau deloc.

## Metodă, lecții și limite

Citite review-ul și LESSONS_FOR_NEXT_SCAN din 15 noiembrie înaintea comparației. Aplicate: GF70 propriu pe teren, toate primele goluri tardive/0–0, concentrarea în scoruri ample, ediția și categoria exacte, minimele 5/3, diferența XI probabil/lot disponibil, statutul neînceput și ora României. Eliminarea timpurie din turul Scoția–Elveția este un avertisment identificat; auditul tuturor fazelor, eliminărilor și cauzelor tactice nu este complet. Nu pretindem analiză video completă sau xG comparabil pentru aceste trei meciuri.

Formula salvată rămâne Poisson 50% sezon/50% teren și stres cu intensitate −20%, dar nu o publicăm ca probabilitate eligibilă pentru eșantioane sub minim. Nu există ajustări procentuale pentru motivație sau accidentări. Fără cote de bookmaker, fără schimbare de model, fără alegere în numele utilizatorului. Clasamentul contextual nu este un top global al zilei.

## Istoricul ediției curente, fără dublarea turului
`;
for(const id of order){const c=data.comparisons.find(c=>c.model.id===id);md+=`\n### ${names[c.model.home_id]}–${names[c.model.away_id]}\n\n| Data RO | Meci | Scor | Primul gol | Gol70 | Dovadă |\n|---|---|---|---|---|---|\n`;for(const h of [...c.edition_history].sort((a,b)=>a.date.localeCompare(b.date)))md+=`| ${new Intl.DateTimeFormat('en-CA',{timeZone:'Europe/Bucharest'}).format(new Date(h.date))} | ${h.home}–${h.away} | ${h.score} | ${h.first_goal==='none'?'Fără gol':h.first_goal} | ${h.success70?'Succes':'Fail'} | [Sursa](${h.source}) |\n`;}
md+=`\n## Ultimele maximum zece oficiale disponibile, diagnostic separat\n\n46 de apariții în tabelele pe echipă; H2H poate apărea la ambele echipe și nu este numărat drept două partide independente. Feedul nu oferă zece oficiale pentru toate echipele. Amicalele sunt excluse; edițiile vechi din această secțiune nu completează eșantionul modelului. Argentina–Elveția este 3–1 după prelungiri, 1–1 la 90; primul gol 10, golurile din prelungiri excluse după reconciliere separată.\n`;
for(const c of data.comparisons)for(const id of[c.model.home_id,c.model.away_id]){const rows=c.recent.filter(r=>r.team_id===id);md+=`\n### ${names[id]}: ${rows.filter(r=>r.goal_result==='success').length}/${rows.length} Gol70\n\n| Data RO | Competiție | Meci | Scor final | Primul gol | Gol70 | Dovadă |\n|---|---|---|---|---|---|---|\n`;for(const r of rows)md+=`| ${new Intl.DateTimeFormat('en-CA',{timeZone:'Europe/Bucharest'}).format(new Date(r.date))} | ${r.competition} | ${r.home}–${r.away} | ${r.score_ft}${r.score_note?' ('+r.score_note+')':''} | ${r.first_goal_minute||'Fără gol'} | ${r.goal_result==='success'?'Succes':'Fail'} | [Sursa](${r.source}) |\n`;}
fs.writeFileSync(D+'/COMPARISON.md',md);
save(D+'/decision.json',{at_ro:updated,market:'GOAL_BY_70',contextual_order:order,contextual_pick:order[0],official_model_decision:'NO_BET_ALL_THREE_VENUE_2_BELOW_MINIMUM_3',p_raw:null,p_stress:null,fair_odds:null,user_selected:false,sources,scope:'Only the three senior mens fixtures requested',limitations:['Small current-edition venue samples','Multiple stale predicted lineups','Final starting lineups not confirmed','Full historical red-card and video audit incomplete']});
const note=`\n\n## Comparație naționale solicitată ulterior — ${updated}\n\nCerere: Kazahstan–Slovacia, Grecia–Serbia, Elveția–Scoția, Gol70. Seniori masculin, Nations League 2026/27 etapa 6, ore RO 17:00/21:45/21:45, statut neînceput reconfirmat. Preferință contextuală Kazahstan–Slovacia (9/9 distincte), apoi Elveția–Scoția și Grecia–Serbia (ambele 7/9). Toate neeligibile formal: 5 general/2 teren sub 5/3; fără procent, AR20 sau fair inventate. SFZ confirmă șapte slovaci fără deplasare, acoperiș închis și artificial; Samorodov suspendat. Akanji/Ndoye retrași, Ioannidis indisponibil: XI din feed sunt vechi. Grecia GF70 propriu 2/5, șapte GA70 într-un singur 0–7; Scoția trebuie să câștige, turul influențat de eliminare. 46 apariții oficiale recente; Argentina–Elveția AET reconciliat, primul 10, prelungirile excluse. Raport ${D}/COMPARISON.md; surse și audit salvate. Alegerile canonice/events recitite și păstrate, candidatul modelat Grecia U21–Islanda U21 rămâne separat. Nu s-a creat alegere nouă și nu s-au modificat modele/programare/Git. Urmează rebuild/verificare dashboard și export privat de continuitate.\n`;
for(const p of['Analysis/Selection_History/2026-11-16.md','Analysis/SESSION_RESUME.md'])fs.appendFileSync(p,note);
fs.appendFileSync('Analysis/Goal70_2026-11-16/EUROPE_ALL_MATCHES.md',`\n\n## Comparație contextuală suplimentară — ${updated}\n\n[Kazahstan–Slovacia / Elveția–Scoția / Grecia–Serbia](Three_Nations_Comparison/COMPARISON.md). Prima preferință contextuală Kazahstan–Slovacia; toate trei au 5/2 sub 5/3, fără procent oficial. Candidatul modelat Grecia U21 și snapshoturile zilnice sunt păstrate.\n`);
const unchanged=Object.fromEntries(protectedFiles.map(p=>[p,hash(p)===before[p]]));assert(Object.values(unchanged).every(Boolean));
save(D+'/verification.json',{at_ro:updated,unstarted_verified:finals,official_recent_rows:recent.length,missing_goal_outcomes:0,current_edition_counts:data.comparisons.map(c=>({id:c.model.id,distinct:9,successes:Number(c.model.first_goal_success),overall_each:5,venue_each:2})),canonical_files_unchanged:unchanged,extra_time_validation:'extra_time_validation.json'});
console.log(JSON.stringify({at_ro:updated,contextual_pick:'Kazakhstan-Slovakia',official_decision:'NO_BET_5_2_BELOW_5_3',report:D+'/COMPARISON.md',unchanged}));
