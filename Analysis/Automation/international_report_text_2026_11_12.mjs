import fs from 'node:fs';
import {csv,read,ro} from './research_2026_11_12.mjs';
const b='Analysis/Goal70_2026-11-12',c='Analysis/Corners1H_2026-11-12',ib=b+'/International',rev=b+'/Revisions/before_international_extension',at=ro();
const goals=csv(ib+'/goal_models.csv'),corners=csv(ib+'/corner_models.csv'),all=csv(b+'/all_models.csv'),sources=read(ib+'/sources.json').urls;
const pct=x=>(100*Number(x)).toFixed(2).replace('.',',')+'%',table=(headers,rows)=>'| '+headers.join(' | ')+' |\n| '+headers.map(()=> '---').join(' | ')+' |\n'+rows.map(r=>'| '+r.map(v=>String(v??'—').replaceAll('|','/')).join(' | ')+' |').join('\n');
const sensitivities=read(ib+'/sensitivity.json');
fs.writeFileSync(ib+'/ICELAND_GREECE_CONTEXT.md',`# Islanda U21–Grecia U21: analiza prospectivă Gol70

Actualizare: ${at}. Candidat al analistului, fără alegere nouă în numele utilizatorului.

**12 noiembrie, 15:00 RO (13:00 UTC), Thróttarvöllur / PEPSI völlurinn, Reykjavík.** Ora este confirmată de [KSÍ](${sources.ksi_preview}) și [EPO](${sources.epo}); meciul a fost reconfirmat neînceput la 09:41 RO în final_status.json.

## Date și calcul

Model brut **93,92%**, stres AR20 **89,35%**, cotă echitabilă matematică **1,0647**. Nu sunt probabilități calibrate sau prețuri oferite de bookmakeri.

Eșantion: calificarea U21 pentru EURO 2027, etichetată 2025/2026 de furnizor. Sunt zece meciuri pentru fiecare echipă și câte cinci în configurația terenului. Frecvența Gol70 este **18/20**, respectiv **9/10** pe teren. λ sezon = 2,7; λ teren = 2,9; media 50/50 = 2,8. Probabilitatea este 1−exp(−2,8); stresul folosește intensitatea 2,24.

- Islanda acasă: GF70 propriu 11/5 = 2,2; primele goluri ale partidelor la 3, 27, 2, 31 și 11; 5/5 succese.
- Grecia în deplasare: GF70 propriu 10/5 = 2,0; primele goluri la 15, 13, **83**, 5 și 32; 4/5 succese.
- Contraexemple din aceeași ediție: Elveția–Islanda 0–0, 10 octombrie 2025; Letonia–Grecia, primul gol 83, 14 octombrie 2025.

Islanda a înscris șapte dintre cele 11 goluri proprii ≤70 acasă în două partide, cu Estonia și Elveția. Grecia a înscris patru dintre cele zece goluri proprii ≤70 în deplasare împotriva Maltei. Aceste concentrări pot ridica intensitatea Poisson fără a demonstra aceeași probabilitate a primului gol împotriva adversarului de azi.

## Context, loturi și contraargumente

Este turul unui baraj cu retur în Atena pe 16 noiembrie. Egalul din tur lasă calificarea deschisă. O abordare mai prudentă este o posibilitate contextuală, nu o tactică demonstrată înaintea jocului.

[KSÍ](${sources.ksi_squad}) îi identifică pe Lúðvík Gunnarsson și Ólafur Helgi Kristjánsson drept antrenorii U21 și publică lotul convocat. [EPO](${sources.epo}) confirmă selecționerul Giannis Taousianis, lotul și disponibilitatea tuturor celor convocați în comunicatul din 8 noiembrie. Acestea nu sunt formulele de start confirmate astăzi. Etichetele medicale FotMob pentru Kosidis/Tzimas nu sunt suficiente pentru a-i prezenta drept absențe actuale ale acestui lot; informațiile sunt păstrate ca neconcordanță de acoperire, fără penalizare numerică.

[UEFA](${sources.uefa_u21}) confirmă stadionul; API indică suprafață artificială. Open-Meteo estimează, între 13 și 15 UTC, −0,2…−0,6°C, precipitații 0,1…0,7 mm/oră, vânt 19,1…19,4 km/h și rafale 32…33,1 km/h. FotMob afișează 1°C; prognozele nu sunt identice. Frigul și umezeala pot influența execuția, dar efectul nu este cuantificat în model. Grecia are deplasare; încărcarea individuală de la cluburi nu a fost auditată exhaustiv. Faptul că ultimul meci al naționalelor a fost pe 6 octombrie nu înseamnă 37 de zile de odihnă.

## Sensibilități separate

${table(['Scenariu diagnostic','Probabilitate','n sezon / teren'],sensitivities.map(s=>[s.label,pct(s.p),s.samples.map(a=>a.n).join(' / ')]))}

Excluderile sunt simetrice pentru ambele echipe. Sunt diagnostice, nu intervale de încredere sau prognoze care înlocuiesc 93,92%. Eliminarea scorurilor cu diferență finală de minimum patru goluri reduce estimarea la aproximativ **89,98%**, un contraargument relevant.

## Verdict și limite

Este principalul candidat calculabil după extinderea solicitată, înaintea Navalcarnero–Betis (88,25%) și Puertollano–Albacete (82,36%). Argumentele sunt istoricul bilateral și contribuția proprie ofensivă pe teren. Riscurile rămân: tur de baraj, adversari eterogeni, selecționate U21, condiții reci, loturi înaintea anunțării XI și sensibilitatea la scoruri ample. Nu este o rată demonstrată de succes de 93,92%.

Pentru cornere: **NO BET**; toate cele 20 de observații FirstHalf ale celor două echipe lipsesc în feedul numeric consultat. Nu sunt înlocuite cu totalurile FT sau cu statisticile seniorilor.

Dovezi locale: historical_observations.csv, MatchData/, StatusAtExtension/6416832.json, Context/, sensitivity.json și final_status.json. Inventarul complet și excluderile sunt în [REPORT.md](REPORT.md).
`);
const report=`# Extindere la internaționale — 12 noiembrie 2026

Cerere: includerea oricărei partide dintre naționale de azi, cu aceleași reguli ca la cluburi, pentru Gol70 și maximum șapte cornere în prima repriză. Actualizare: ${at}.

## Acoperire

Inventarul conține **43 de înregistrări internaționale: 24 de seniori și 19 de juniori**. Sunt 41 de partide identificate în programe și două semnalări neconfirmate: Egipt U20–Tanzania U20 și Rusia U21–Kârgâzstan U23. Aceste două semnalări nu au model. Nu am identificat partide feminine între naționale în sursele consultate; absența lor globală nu este demonstrată.

Au fost confruntate trei date UTC adiacente FotMob (11/12/13), șapte programe de competiții, ESPN, programe independente și surse UEFA, CAF, federații și cluburi. Din 28 de identități FotMob au fost colectate **153 de observații istorice**; 15 înregistrări suplimentare au surse web, dar istoricul lor numeric nu este complet colectat. Lipsesc patru liste exacte de goluri și 56 de observații FirstHalf din cele 153. **Cercetare parțială, nu top global complet.**

Serbia U21–Macedonia de Nord U21 a fost recuperat din programul competiției, fiind omis din programul global. Patru U17/U19 au fost reprogramate în Portugalia din 11 în 12 noiembrie; comunicatul UEFA are prioritate față de calendarele vechi. CAF confirmă separat trei meciuri U23. Partidele CONCACAF de la 22:00/23:00 UTC pe 12 noiembrie și 00:30 UTC pe 13 noiembrie aparțin zilei de **13 noiembrie în România**. Noua Zeelandă–India era deja începută și nu primește predicție retrospectivă.

## Aplicarea acelorași reguli

Se păstrează pragurile 5 observații generale și 3 în configurația terenului, lipsa datelor nu este zero, iar formula rămâne 50/50 Poisson cu AR20 separat. Se folosesc ediția curentă a competiției și, separat, amicalele din anul calendaristic curent. Seniorii, U21, U20, U19, U17 și U23 nu își înlocuiesc reciproc istoricul. Nu adăugăm ediții vechi sau date de club pentru eligibilitate.

- UEFA Nations League, opt partide: numai patru meciuri anterioare și două pe teren pentru fiecare națională; sub pragurile 5/3.
- CONCACAF Nations League, trei partide în data RO: eșantion curent sub praguri.
- AFCON, nouă partide: în general două meciuri anterioare; preliminariile unora nu fac bilateralitatea suficientă.
- Uzbekistan–Rusia: o listă de goluri incompletă și o singură deplasare pentru Rusia în amicalele 2026. Celelalte amicale au eșantioane sau observații insuficiente.
- Islanda U21–Grecia U21: singurul internațional calculabil la Gol70; niciun internațional calculabil la cornere.

## Comparația calculabilă extinsă

${table(['Rang extins','Meci','Brut','AR20','Istoric','Teren'],all.filter(r=>r.p_raw!=='').map(r=>[r.rank,r.home+'–'+r.away,pct(r.p_raw),pct(r.p_stress),r.first_goal_success+'/'+r.first_goal_n,r.venue_success+'/'+r.venue_n]))}

**Candidat principal Gol70: Islanda U21–Grecia U21, 15:00 RO.** [Analiza completă și contraargumentele](ICELAND_GREECE_CONTEXT.md). Model brut 93,92%, stres 89,35%, frecvență 18/20 și 9/10 pe teren, fair matematic 1,0647. Cornere: **NO BET**. Nu s-a creat o alegere în numele utilizatorului.

Versiunea inițială de club este arhivată în ../Revisions/before_international_extension/. Probabilitățile, intensitățile și timestampurile ei sunt păstrate; rangul vechi apare în original_screening_rank, iar rangul extins compară universul nou. NO BET din raportul inițial de club descrie versiunea anterioară cererii, nu această extindere.

## Lecții aplicate și limite

Review-ul și lecțiile din 11 noiembrie au fost predate la 09:20:29 RO, înaintea scanării inițiale de la 09:20:41. Au fost recitite pentru extindere: brut/frecvență/AR20 separate; GF70 propriu pe teren; 0–0 și primul gol 83 păstrate; FirstHalf lipsă rămâne NULL; competiții și categorii distincte; grupă versus baraj; loturi și antrenori confirmați de federații, XI încă neconfirmat; prognoze și teren fără ajustări arbitrare; kickoff și neîncepere reconfirmate.

Rămân conflicte de afișaj la Guineea–Eritreea (18/19 RO), Noua Zeelandă–India (08:00/08:10 RO) și unele calendare de juniori. Pentru 15 suplimente nu există încă istoricul numeric complet. Aceste limite împiedică o afirmație de acoperire globală completă. Modelele de producție și selection_engine/ sunt neschimbate.

## Inventarul internațional și motivele neeligibilității

Motivele tehnice de mai jos sunt trasabile la verificări: overall_sample_below_5 = sub cinci observații; venue_sample_below_3 = sub trei pe teren; missing_exact = observații exacte lipsă; one_or_both_team_histories_unavailable = istoric bilateral indisponibil; identity_or_live_status_unconfirmed = identitate/status neconfirmat. Nu sunt veto-uri noi de model.

${table(['Meci','Competiție','Ora RO','Gol70 / motiv','Cornere / motiv'],goals.map(r=>[r.home+'–'+r.away,r.competition,r.kickoff_ro?String(r.kickoff_ro).slice(11,16):'TBD',r.p_raw!==''?pct(r.p_raw):r.reason,corners.find(c=>c.id===r.id)?.reason]))}

## Surse

${Object.entries(sources).map(([k,v])=>'- ['+k+']('+v+')').join('\n')}

Dovezi: Global/, Schedules/, ESPN/, MatchData/, StatusAtExtension/, Context/, competition_coverage.json, supplementary_inventory.csv și scope.json. Nu sunt folosite cote de bookmaker sau procente de tipsteri.
`;
fs.writeFileSync(ib+'/REPORT.md',report);
for(const [dir,name]of [[b,'EUROPE_ALL_MATCHES.md'],[c,'ANALYSIS.md']]){
 const prior=fs.readFileSync(rev+'/'+dir+'/'+name,'utf8');
 const header=`# Actualizare: cluburi și internaționale — ${at}

${dir===b?'**Candidat analist Gol70: Islanda U21–Grecia U21, 15:00 RO.** Model brut 93,92%, AR20 89,35%, frecvență 18/20 și 9/10 pe teren; fair matematic 1,0647.':'**CORNERS_1H_UNDER_7_5: NO BET.** Niciun internațional nu are datele FirstHalf bilaterale suficiente pentru model.'}

[Inventarul extins, calculele și excluderile](${dir===b?'International/REPORT.md':'../Goal70_2026-11-12/International/REPORT.md'}). Total: 46 de înregistrări, dintre care 3 de club și 43 internaționale; 3 modele Gol70, 0 cornere. Acoperire parțială; două semnalări de juniori rămân neconfirmate. Nicio alegere a utilizatorului inferată.

Mai jos este păstrat raportul inițial de club. NO BET din acea versiune precedă cererea de extindere și nu descrie decizia ulterioară. Probabilitățile și timestampurile originale nu au fost schimbate.

---

`;
 fs.writeFileSync(dir+'/'+name,header+prior);
}
console.log('Saved readable international reports; numerical inputs unchanged.');
