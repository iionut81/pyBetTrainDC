import fs from 'node:fs';
import crypto from 'node:crypto';
import {read,csv,save,fetch,api,pool,ro} from '../../Automation/research_2026_11_08.mjs';
const dir='Analysis/Goal70_2026-11-08/Brighton_Ipswich_Comparison';
const comparison=read(dir+'/comparison.json'),diagnostics=read(dir+'/diagnostics.json'),recent=csv(dir+'/last10_official.csv');
const pct=p=>(Number(p)*100).toFixed(2)+'%',num=n=>Number(n).toFixed(4);
const sources=read(dir+'/sources.json'),url=key=>sources.find(s=>s.key===key)?.url;
const link=(key,label)=>`[${label}](${url(key)})`;
const statuses=[];
await pool(comparison.contexts,async c=>{const r=await fetch(api('matchDetails?matchId='+c.id));save(dir+'/final_status_'+c.id+'.json',r);const s=r.data?.header?.status;statuses.push({id:c.id,verified_at_utc:r.verified_at_utc,source:r.source,identity_ok:String(r.data?.general?.matchId)===c.id,utcTime:s?.utcTime,started:s?.started,cancelled:s?.cancelled,unstarted:!!(r.ok&&String(r.data?.general?.matchId)===c.id&&!s?.started&&!s?.finished&&!s?.cancelled&&Date.parse(s?.utcTime)>Date.now())});});
save(dir+'/final_status.json',statuses);if(statuses.some(s=>!s.unstarted))throw Error('Fresh unstarted status not verified');
const table='| Indicator | Brighton–Brentford | Ipswich–Bournemouth |\n|---|---:|---:|\n'+[
 ['Ora România','16:00','16:00'],['Rang brut în inventarul zilei','20','26'],
 ['Model brut',...comparison.contexts.map(c=>pct(c.model.p_raw))],['AR20: intensitate redusă cu20%',...comparison.contexts.map(c=>pct(c.model.p_stress))],
 ['Doar sezon',...comparison.contexts.map(c=>pct(c.model.season_only))],['Doar teren',...comparison.contexts.map(c=>pct(c.model.venue_only))],
 ['Cotă exclusiv matematică',...comparison.contexts.map(c=>num(c.model.fair_odds))],['Primul gol≤70 în sezon','18/18','18/18'],['Primul gol≤70 pe terenurile relevante','8/8','8/8'],
 ['Goluri proprii≤70: gazda acasă','11/4 =2,75/meci','2/4 =0,50/meci'],['Goluri proprii≤70: oaspetele afară','5/4 =1,25/meci','6/4 =1,50/meci'],
 ['Ultimele10 oficiale: gazdă/oaspete','9/10 și10/10','10/10 și10/10']
].map(r=>'| '+r.join(' | ')+' |').join('\n');
const formulas=comparison.contexts.map(c=>{const r=c.model;return `### ${r.home}–${r.away}\n\nλsezon = (${r.home_gf}/${r.home_n} + ${r.away_ga}/${r.away_n} + ${r.away_gf}/${r.away_n} + ${r.home_ga}/${r.home_n})/2 = ${num(r.lambda_season)}.\n\nλteren = (${r.home_venue_gf}/${r.home_venue_n} + ${r.away_venue_ga}/${r.away_venue_n} + ${r.away_venue_gf}/${r.away_venue_n} + ${r.home_venue_ga}/${r.home_venue_n})/2 = ${num(r.lambda_venue)}.\n\nλcombinat = 0,5λsezon + 0,5λteren = ${num(r.lambda_combined)}. P=1−exp(−λ)=${pct(r.p_raw)}; AR20 folosește0,8λ și dă${pct(r.p_stress)}.\n`;}).join('\n');
const historyTables=comparison.contexts.flatMap(c=>['home','away'].map(side=>{const tid=c.model[side+'_id'],r=recent.filter(x=>x.team_id===tid);return `### ${c.model[side]}\n\n| Data | Competiție | Partidă | Scor90 | Primul gol | Gol70 | Dovadă |\n|---|---|---|---|---:|---|---|\n`+r.map(x=>`| ${x.date.slice(0,10)} | ${x.competition} | ${x.home}–${x.away} | ${x.score_ft} | ${x.first_goal_minute||'Fără gol'} | ${x.goal_result==='success'?'Succes':x.goal_result==='fail'?'Fail':'Neverificat'} | [date](${x.source}) |`).join('\n');})).join('\n\n');
const diagTable='| Diagnostic simetric | Brighton–Brentford | Ipswich–Bournemouth |\n|---|---:|---:|\n'+[
 ['Fără FT cu diferență≥4',...diagnostics.map(d=>pct(d.without_big_margins.p))],
 ['n sezon gazdă/oaspete; n teren gazdă/oaspete',...diagnostics.map(d=>d.without_big_margins.samples.join('/'))],
 ['Fără cartonaș roșu până la70 inclusiv',...diagnostics.map(d=>pct(d.without_red_by70.p))],
 ['n sezon gazdă/oaspete; n teren gazdă/oaspete',...diagnostics.map(d=>d.without_red_by70.samples.join('/'))]
].map(r=>'| '+r.join(' | ')+' |').join('\n');
const content=`# Brighton–Brentford versus Ipswich–Bournemouth: Gol70

Actualizare ${ro()}, Europe/Bucharest. Ambele partide sunt astăzi8noiembrie2026,16:00RO/14:00UTC și au fost reverificate ca neîncepute; vezi final_status.json. Premier League, etapa10, meciuri de campionat cu trei puncte pentru victorie și egal permis, fără scor agregat, prelungiri eliminatorii sau departajare în această partidă.

## Verdict comparativ: Brighton–Brentford, avantaj mic

Prefer Brighton–Brentford dintre cele două cerute: producție proprie înainte de70 mai puternică la gazde și contribuție verificabilă a oaspeților. Nu afirm superioritate demonstrată: diferența modelului este numai0,60puncte procentuale, istoricul recent avantajează Ipswich–Bournemouth, iar diagnosticul fără eliminări inversează ușor ordinea. Nu este un nou top global și nu înlocuiește candidatul inițial Cambuur–PSV. Nicio alegere a utilizatorului nu este înregistrată automat.

${table}

18/18și8/8 sunt frecvențe, nu probabilități de100%. Eșantioanele sunt mici, suprapuse și nu independente. Nu folosim cote de bookmaker; cota1/pnu este ofertă disponibilă și nu confirmă preț≥1,10.

## Brighton–Brentford: argumente și riscuri

Brighton≤70acasă:4,0,3,4goluri proprii în ordine cronologică; a marcat în3/4. Brentford≤70afară:1,2,1,1; a marcat în4/4. Gazda nu depinde numai de goluri primite. Dar8din11goluriBrighton provin din două succese ample, cuAstonVillașiPalace; meciulVilla are eliminare40dupăprimulgol8. Nu confundăm volumul de după primul gol cu probabilitatea de a deschide meciul.

${link('brighton_preview','Brighton, avancronică oficială7noiembrie')} confirmă revenirea acasă și absența unor probleme medicale noi; Chema este doar sperat disponibil. Clubul indică faptul că Brentford a marcat în fiecare etapă și a deschis scorul de opt ori; acestea sunt statisticiFT/primulgol, nu minute≤70. Brighton a păstrat poarta intactă în cele șase victorii de campionat: această soliditate poate limita contribuția Brentford.

${link('brentford_preview','Avancronica Brentford5noiembrie')} susține un profil de pressing înalt și faze fixe pentruBrighton. Interpretarea mea: recuperările sus, pasele verticale și duelurile pot produce ocazii fără un atac pozițional îndelungat. Nu deduc stilul numai din4-2-3-1 și nu adaug procente. Brentford este descris deHürzeler ca adversar direct, fizic, orientat spre mingea a doua.

${link('brighton_recovery','Hürzeler despre recuperare7noiembrie')} confirmă deplasareaGetafejoi5noiembrie și revenirea înSussexvinerea la prânz. Acesta este riscul contextual principal pentru intensitateaBrighton. Au fost cinci schimbări înXI de joi, ceea ce atenuează dar nu elimină incertitudinea. Brentford a jucat ultima dată31octombrie:8zile între date, versus3pentruBrighton. Nu pretind odihnă completă măsurată sau traseulBrentfordverificat.

${link('brentford_team_news','Lot Brentford, sursă oficială5noiembrie')}: Yarmoliuk revine după suspendare; Henry șiCarvalho rămân absenți; Milambo nu era încă pregătit, Dasilva în recuperare. XI-urileAPI sunt probabile, ambele4-2-3-1, nu confirmate. APIlistează Mitoma,Ferguson,Tzimas,Yohanna,AzeezșiincertitudiniKadıoğlu/Struijk; confirmarea primară individuală a întregii liste lipsește. Chema apare înXIprobabil, dar clubul încă îl evaluează.

Context de clasament înaintea etapei:Brightonlocul3,Brentford6conformavancronicilorcluburilor. Ambele au motive să urmărească victoria, dar nu există obligație de a ataca imediat sau dovadă că un egal ar fi respins de la început.

## Ipswich–Bournemouth: de ce rămâne alternativă apropiată

Ipswich≤70acasă:1,0,0,1; Bournemouthafară:1,2,1,2. Atacul oaspeților susține selecția, însă contribuția gazdei este mai redusă înaintea pragului. Ipswich–Fulham a fost2–1FT,dar la70Ipswichavea0goluri: primul55a fost alFulham, iar cele două goluri ale gazdei sunt ulterioare pragului. Acesta este motivul pentru care nu folosimFTîn loculGF70.

În ultimele10oficiale, ambele echipe au10/10partide deschise până la70; Bournemouthare șapte victorii consecutive. Seria este favorabilă, dar nu garantează continuarea. Bournemouth vine și ea după deplasarea europeană laSpartaPrague5noiembrie, în timp ceIpswichnu a mai jucat din31octombrie. Diferența dintre odihna lotului și minutele titularilor contează: relatările indică rotație, iar XI de azi rămâne probabil.

${link('ipswich_presser_report','Conferința O’Neil relatată deTWTD6noiembrie')} descrie construcția directă spreEvanilson/Rayan și schimbări de organizare fără minge subMarcoRose. Acestea pot genera tranziții; organizarea compactă poate și blocaIpswich. Același articol numește adversarul europeanSlavia, în timp ce datele de meci îl identificăSpartaPrague; păstrăm conflictul și folosim identitatea din fișa meciului. Nu folosim această eroare editorială pentru inventarea itinerarului.

GaryO’NeilșiMarcoRose sunt antrenorii actuali înAPI și relatările conferințelor; nu aplicăm descrierile vechiMcKenna/Iraola. Formații probabile4-2-3-1pentruambele, nu titulari confirmați. APIaratăJackTaylorincert, în timp ce conferințaO’Neil relatată în ${link('press_conference_index','indexul conferințelor6noiembrie')} afirmă lot disponibil; conflictul rămâne explicit. Lukic este relatat revenit la antrenament. Bournemouth:APIaratăJuanlu/Araujo/Tóthindisponibili,Milosavljevićincert,Scottla antrenamente; indexulconferinței spuneScottîncănepregătitcomplet șiKroupiînrefacere. Nu echivalăm revenirea la antrenament cu titularizarea. Corpul lizibil al articoluluiIpswichoficial nu confirmă aceste detalii medicale, deși titlul/data au fost verificate.

Clasamentele relatate diferă după momentul publicării:Ipswich14/Bournemouth8înaintea etapei, Bournemouth9înaltărelatare; diferența de un punct este consemnată, fără utilizare în formulă. Este campionat, egalul este permis; motivația nu dovedește un meci deschis imediat.

## Condiții și limite

APIpentruambele:9°C, probabilitateploaie2%/3%,AmexșiPortmanRoadgazon natural. Vânt1/2fărăunitate explicită; nu îl compar cu un prag meteorologic. Nu am confirmare meteo independentă sau inspecție a terenului pentru această comparație, deci nu aplic ajustări. Miza și scorurile din campionat nu țin loc de observații privind mindsetul real sau planul din primul minut.

## Calcul păstrat din snapshotul original

Numărătorile au fost recalculate din istoricul sezonului, anterior00:00RO8noiembrie; valorilecoincidcuCSVoriginal. Gol70înseamnăminimumungol valid aloricăreiechipe la minutulafișat≤70,inclusiv45+. Toate golurile folosite sunt reconciliate cu scorul90. Fără extra-time/pens, fără inventarea datelor lipsă.

${formulas}

Ponderarea50/50este euristică, eșantioanele sezon/teren se suprapun; modelulPoissonnu este validat sau calibrat. Golurile apărute după deschiderea scorului pot umflaλ. Nu aplic bonusuri pentru stil/motivație/accidentări.

## Scenarii diagnostice, fără modificarea prognozelor

${diagTable}

Filtrele sunt aplicate identic ambelor perechi, numai ca sensibilități exploratorii. Fără scorurile ample,Brightonrămâne înainte, dar are numai2observațiiacasă, subminimul3al screeningului: rezultatul nu este model eligibil nou. Fărăeliminări≤70,Ipswich–Bournemouthiese ușor înainte. Concluzia: preferințaBrightonnu este robustă la toate alegerile de eșantion. Niciun diagnostic nu este interval de încredere, predicție calibrată sau motiv să înlocuim probabilitățile arhivate.

## Lecții aplicate

Am citit review7noiembrie șiLESSONS_FOR_NEXT_SCAN înaintea comparației. Am separatnumărători≤70deFT, propria producție de golurile cedate, intern deEuropa/cupe, istoricul de model și scenariile de verdict. Am inclus0–0Jablonec–Brighton și perspectiva contrară20/20Ipswich/Bournemouth. Nu am selectat doar argumentele favorabile și nu am modificat modelul după rezultatele de ieri. Necunoscute: titularii, impactul oboselii, verificarea completă medicală independentă și calibrarea.

## Ultimele10oficiale pentru fiecare echipă

Cele40rânduri furnizate sunt înaintea zilei de azi, fără amicale și fără observațiiGol70lipsă. În fiecare pereche agregămpartideleunice; seriileechipelor nu sunt independente. Formula rămâne exclusivcampionatulcurent, cupele șiEuropa sunt diagnostice separate.

${historyTables}

## Fișiere și pas următor

comparison.json include loturile/contextul și snapshotul original;diagnostics.json calcule separate;last10_official.csv istoriccomplet;sources.json șiHTMLpublice dovezi;final_status.json status final. Cererea de comparație nu este o alegere a utilizatorului și nu autorizează pariu/miză/combinare/commit/push sau schimbarea programării. La următoarea verificare se citeșteCSV/evenimentele canonice, nu se deduce o alegere din acest verdict.
`;
fs.writeFileSync(dir+'/COMPARISON.md',content);
save(dir+'/recommendation.json',{recorded_at_ro:ro(),role:'analyst_comparison_preference',market:'GOAL_BY_70',match_id:'5795506',home:'Brighton & Hove Albion',away:'Brentford',p_raw:Number(comparison.contexts[0].model.p_raw),p_stress:Number(comparison.contexts[0].model.p_stress),confidence_in_relative_order:'small edge; not robust to red-card sensitivity',not_user_choice:true,not_global_reranking:true});
const protectedFiles=['Analysis/Selection_History/user_selections.csv','Analysis/Selection_History/selection_events.jsonl','Analysis/Selection_History/2026-11-08_goal70_candidates.csv','Analysis/Selection_History/2026-11-08_corners1h_candidates.csv'];
save(dir+'/preserved_before_build.json',Object.fromEntries(protectedFiles.map(p=>[p,crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex')])));
console.log(JSON.stringify({report:dir+'/COMPARISON.md',history_rows:recent.length,statuses,preferred:'5795506'}));
