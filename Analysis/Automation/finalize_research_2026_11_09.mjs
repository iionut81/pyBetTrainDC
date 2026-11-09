import fs from 'node:fs';
import {read,csv,save,writeCsv,ro} from './research_2026_11_09.mjs';
const date='2026-11-09',b='Analysis/Goal70_'+date,c='Analysis/Corners1H_'+date,R='Analysis/Review_2026-11-08',H='Analysis/Selection_History',stamp=ro();
const hist=read(b+'/historical_observations.json'),recent=csv(b+'/last10_official.csv'),ctx=read(b+'/context_summary.json').matches,status=csv(b+'/status_at_selection.csv');
const all={goal:csv(b+'/all_models.csv'),corner:csv(c+'/all_models.csv')};
const pct=x=>x===''||x==null?'lipsă':(100*Number(x)).toFixed(2)+'%';
const table=(keys,rows)=>'| '+keys.join(' | ')+' |\n| '+keys.map(()=>'---').join(' | ')+' |\n'+rows.map(r=>'| '+r.map(x=>x??'lipsă').join(' | ')+' |').join('\n')+'\n';
for(const market of ['goal','corner']){
 const rows=all[market];for(const r of rows){r.analyst_selected=market==='corner'&&r.id==='5868534'?'true':'false';r.user_selected='false';}
 writeCsv((market==='goal'?b:c)+'/all_models.csv',rows);writeCsv(`${H}/${date}_${market==='goal'?'goal70':'corners1h'}_candidates.csv`,rows);
}
const chosen=all.corner.find(r=>r.id==='5868534');
writeCsv(c+'/prospective_selection.csv',[{...chosen,market:'CORNERS_1H_UNDER_7_5',analysis_path:c+'/ANALYSIS.md',status:'analyst_candidate_unvalidated',kickoff_verified_at_ro:status.find(r=>r.id===chosen.id).verified_at_utc}]);
save(b+'/selection_decision.json',{at_ro:stamp,market:'GOAL_BY_70',decision:'NO BET',statistical_leader:'5105046',reason:'Confirmed12-player illness outbreak at Vålerenga; recovery and available squad not established. Other compared models have lower probabilities and contextual uncertainty. No numeric penalty or new automatic filter applied.'});
writeCsv(c+'/historical_corners_1h.csv',hist.map(({goals,cards,status,...x})=>x));
const start=read(b+'/research_start.json');
const common=`Actualizat ${stamp}, Europe/Bucharest. [Review anterior](../Review_2026-11-08/REVIEW.md) și [lecții](../Review_2026-11-08/LESSONS_FOR_NEXT_SCAN.md) predate la${start.review_handoff_at_ro}; cercetare începută${start.research_started_at_ro}. Corecția editorială a numelor din preambulul lecțiilor este documentată; rezultatele și verificările cerute existau înaintea scanării. Fără modificări de modele, ponderi, calibrare, cote bookmaker sau xG inventat.

Inventar: **29 ligi,4 partide,4 modeleGol70 și3cornere**, ${hist.length} istorice unice anterioare00:00RO9noiembrie. Primele20asociații din [clasamentul UEFA încheiat în2026](https://www.uefa.com/nationalassociations/uefarankings/?year=2026), plusRomânia,Serbia și7divizii secunde urmărite. Lista celor29 de ligi, inclusiv toate cele26 fără partide identificate, este în [universe_coverage.csv](../Goal70_2026-11-09/universe_coverage.csv). Anglia,Italia,Spania,Germania,Franța,Portugalia,Olanda,Belgia,Turcia,Cehia,Grecia,Polonia,Danemarca,Norvegia,Cipru,Elveția,Austria,Scoția,Suedia,Croația sunt cele20; suplimenteRomânia/Serbia șiE1,D2,SP2,I2,F2,B2,TR2.

FotMob global/calendar sezon șiESPN au fost confruntate. Norvegia1,România1,Spania2 sunt ligile cu partide. ESPN nesuportat pentruPolonia,Croația,Serbia,Belgia2: absențele lor sunt acoperite de calendarul sezonului aceluiași furnizorFotMob, fără confirmare independentă exhaustivă. ESPN gol pentruRomânia contrazice clubulBotoșani șiFotMob. LasPalmas–Eldense a fost omis dinFotMob global; [clubul confirmă9noiembrie19:30Canare=21:30RO](https://www.udlaspalmas.es/noticias/el-partido-j13-laspalmaseldense-se-disputara-manana-lunes-9-de-noviembre-a-las-1930-horas-hora-canaria-en-el-estadio-gran-canaria), la felESPN; endpointul individualFotMob a confirmat ulterior noua oră și neînceperea. Inventarul inițial3partide și toate predicțiile inițiale sunt păstrate înRevisions/; cele3valori/ranguri/timestampuri nu au fost rescrise.

[Excluderi](../Goal70_2026-11-09/exclusions.csv):54partide din afara universului convenit, fără a le declara inexistente sau calculate. [Cupe](../Goal70_2026-11-09/cups_coverage.json): programul global consultat este inventariat separat; nu aparUCL/UEL/UECL eligibile și nu proiectăm formulele naționale automat peste cupe. Calendarul global nu este dovadă exhaustivă pentru toate cupele. [Statusuri individuale](../Goal70_2026-11-09/status_at_selection.csv):4/4 neîncepute, ultima verificare09:03RO;3partide au fost confirmate și prin surse oficiale distincte, iarLasPalmas prin comunicatul oficial reprogramării.

**Cercetare parțială**, cu calcul complet pentru eligibilele identificate: o observațieFirstHalf lipsește în istoriculLasPalmas/Eldense, deci nu există modelcornere pentru ea. ContextAPI pentru4partide,8echipe și80rânduri din ultimele10oficiale;9FirstHalf și1listă de goluri nereconciliate rămân lipsă în aceste serii. Titularii nu sunt confirmați; cronologia completă a schimbărilor de antrenor și recuperarea bolnavilorVålerenga nu sunt verificate. Confirmarea independentă exhaustivăFirstHalf lipsește. Nu numim cercetarea top global complet verificat.

Metodă: numai campionatul actual, minimum5observații sezon și3pe teren per echipă, fără observații cerute lipsă. λsezon=(GF_A/n_A+GA_B/n_B+GF_B/n_B+GA_A/n_A)/2; analog pe teren; λ=0,5λsezon+0,5λteren. Gol70:1−exp(−λ),stres0,8λ. Cornere:PoissonCDF(7,λ),stres1,2λ. GF70 numără TOATE golurile valide≤70, inclusiv45+; nu aplicăm70/90. FirstHalf≤7inclusiv45+; lipsa nu devinezero. Procente exploratorii necalibrate, frecvențele istorice separate și eșantioanele suprapuse;1/p este exclusiv cotă matematică.

`;
const goalContext=`## GOAL_BY_70: NO BET

**Lider statistic Kristiansund–Vålerenga,20:00RO:90,90%brut,85,30%stres,1,1001matematic.**47/49istoric și23/24teren; GF70propriu11/12acasă și12/12afară. Nu îl transform în recomandare: [NFF](https://www.fotball.no/turneringer/eliteserien/2026/kristiansund---valerenga-utsatt/) confirmă documente medicale pentru12jucători și mutarea partidei din7în9noiembrie. [Vålerenga](https://www.vif-fotball.no/nyheter/kristiansund-kampen-flyttes-til-mandag) confirmă motivul. Nu avem lista celor recuperați sau aprobarea medicală individuală de azi; istoricul de sezon nu descrie această situație. Nu deducem că apărarea slăbită garantează goluri: atacul, ritmul și selecția sunt și ele incerte. Este o judecată contextuală pentru această partidă, fără penalizare numerică sau filtru nou automat.

Etapa26,Nordmøre,gazonartificial. [ClubulKBK](https://www.kristiansundbk.no/lag/import/tournament/eliteserien-2026/season/fotballsesongen-2026/match/round-26-kristiansund-x-valerenga) indică gazdele pe15cu23puncte după25meciuri, deci puncte importante în lupta de jos; miza nu impune gol≤70. API:Skiri4-3-3/Moesgaard4-4-2, XIprobabil; Igor/Tufekčić șiDrammeh/Rashani în listaAPI, care omite focarul colectiv. Formația nu dovedește tactica. Verificările actuale de lot au prioritate față de o listăAPI aparent scurtă.

Alternative: Sabadell–RealSociedadB87,70%/81,30%,19/24și10/11;3meciuri0–0Sabadell în sezon,2deschideri tardiveSanse, probabilitate mai mică și risc de control defensiv. Botoșani–Oțelul85,72%/78,93%,24/28și11/12,GFpropriu8/6și4/6; contribuție ofensivă mai modestă a oaspeților și echipă în schimbare. LasPalmas–Eldense84,01%/76,93%,20/24; alertă meteo ridicată și reprogramare, context/loturi incomplet confirmate. Nu forțez înlocuirea liderului cu un meci mai slab documentat.

`;
const cornerContext=`## Candidat analist: Sabadell–RealSociedadB,21:30RO

**Maximum7cornereFirstHalf:94,84%brut,88,57%stres, cotă matematică1,0544; rang1.** Istoric24/24 și11/11teren,2reprizeexact7,zero≥8; nu reprezintă100%pentruazi. Gazdele au5observațiiacasă, oaspeții6afară: eșantioane mici. Modelul nu este calibrat. Este singura preferință individuală a zilei; nu a fost înregistrată ca alegere a utilizatorului.

LaLiga2etapa13,NovaCreuAlta,gazon natural. [LaLiga](https://www.laliga.com/es-GB/partido/temporada-2026-2027-laliga-hypermotion-ce-sabadell-r-sociedad-b-13) arată17puncte/loc10 și11puncte/loc19 după12meciuri. [RealSociedad](https://www.realsociedad.eus/es/noticias/detalle/nuevo-desafio-1) confirmă20:30local=21:30RO și intenția de a continua victoria din etapa trecută. Campionat de puncte, fără retur sau scor de apărat obligatoriu.

Principalul contraargument: [avancronica jurnalistică](https://www.mundodeportivo.com/futbol/sabadell/20261106/1004247992/claves-sabadell-frenar-sanse.html) descrie verticalitateaSanse, amenințareaDaniDíaz și nevoiaSabadell de control prin posesie. Tranzițiile pe benzi pot genera blocaje/cornere. ÓscarSanz este suspendat;Edgar revine,Quadri este incert după lovitura la cap. APIlastStarting11 îi include peSanz/Quadri: NU esteXIconfirmat și suspendarea are prioritate. Nu deduc reducerea cornerelor din eventualul control al posesiei. ArticolulEFE6noiembrie confirmă incertitudineaQuadri și o săptămână emoțional dificilă; efectul asupra ritmului nu poate fi cuantificat.

Antrenori:Costa, confirmat în conferința relatatăEFE;JonAnsotegi confirmat de [conferința oficială7noiembrie](https://www.realsociedad.eus/es/noticias/detalle/siguela-en-directo-634). APIarată4-4-2ambele, dreptXIvechi. Nu avem cronologia completă a schimbărilor sau lista de azi a rezervelorSanse chemate la primaechipă. Istoricul filialelor poate fi sensibil la aceste mutări. Absența unor nume înAPI nu dovedește lot complet.

Ambele au ultimul campionat31octombrie,9zile între date; cupa este separată și nu presupunem oboseală europeană pentru filială din programul primeiechipe. DeplasareSanse laSabadell, fără itinerar/ore de somn verificate. Datele și competițiile celor10oficiale sunt în tabelele de mai jos.

Alternative: Botoșani–Oțelul91,34%/82,17%,26/28și12/12teren,2reprize≥8și3exact7; varianțele sezonului sunt5,94/7,82, mai mari decât laSabadell/Sanse. [ClubulBotoșani](https://fcbt.ro/) confirmă20RO; relatareaconferințeiCroitoru semnalează3indisponibili,inclusivIlașsuspendat, și adversar cu două module: listaAPIgoală nu este lot validat. Kristiansund84,75%/71,43%,44/49și21/24,5reprize≥8și8exact7, în plusfocarulmedical. LasPalmas nu are model eligibilcornere din cauza unei observațiiFirstHalf lipsă; nu estezero și nu îl clasific probabilistic.

`;
const lessons=`## Lecții aplicate din review

| Lecție | Verificare și dovadă nouă | Statut și efect |
| --- | --- | --- |
| Model/frecvență/AR20 | Toate4partidele calculate înaintea alegerii;Sabadell94,84/88,57versus24/24 | Aplicată; nu înlocuim modelul cu100%istoric |
| GF70propriu,tardive/0–0 |11/12și12/12la lider;Sabadell3zero-zero,Sanse2tardive în sezon; tabele exacte | Aplicată;NO BETGol70, fără procente modificate |
| Intern/Europa/cupă | Formula doar sezon intern;80rânduri oficiale cu competiție separată | Aplicată;9FirstHalf și1gol nereconciliate rămân explicite |
| Dispersie,exact7/≥8 |Sabadell2exact7/0≥8, varianțe teren1,70/1,87; alternative în tabele | Aplicată; susține preferința, eșantioane5/6 |
| Loturi și antrenori actuali |NFF12bolnaviVIF;Sanzsuspendat/Quadriincert contrazicXIvechi;Ansotegi confirmat primar | Aplicată parțial; focarul oprește recomandareaGol70, titulari neconfirmați |
| Program/status independent |LasPalmas omis în global identificat dinESPN+club;4/4neîncepute individual | Aplicată; revizia inventarului păstrată, fără predicție retrospectivă |
| Meteo/odihnă |PrognozeOpenMeteo șiAPI, ultimele date verificate; itinerare incomplete | Aplicată parțial; fără ajustări numerice |
| Cauze și calibrare |Brighton15înaintearoșului23; eșecurile nu au cauze tactice demonstrate | Neverificate exhaustiv; fărătuning sau promisiuni |

`;
function sampleTable(r,k){return table(['Eșantion','n / lipsă','GF / GA','reușite','tardiv /0–0','medie /varianță','exact7 /≥8'],['home','away','home_venue','away_venue'].map(p=>[p,r[p+'_n']+' /'+r[p+'_missing'],r[p+'_gf']+' /'+r[p+'_ga'],r[p+'_wins'],r[p+'_late']+' /'+r[p+'_nil'],r[p+'_mean']+' /'+r[p+'_variance'],k==='corner'?r[p+'_seven']+' /'+r[p+'_eight']:'—']));}
function series(r,k){let out='';for(const side of ['home','away']){
 const team=r[side+'_id'],season=hist.filter(m=>m.league===r[side+'_league']&&(m.home_id===team||m.away_id===team)).sort((a,b)=>Date.parse(a.date)-Date.parse(b.date));
 out+='\n### '+r[side]+' — sezonul curent\n\n'+table(['Dată','Partidă','teren','primul gol','Gol70','FirstHalf','cornere'],season.map(m=>[m.date?.slice(0,10),m.home+'–'+m.away,m.home_id===team?'acasă':'afară',m.goal_ok?(m.first_goal||'0–0'):'lipsă',m.goal_ok?(m.success70?'success':'fail'):'pending',m.corner_ok?m.total1h:'lipsă',m.corner_ok?(m.under?'success':'fail'):'pending']));
 out+='\n### '+r[side]+' — ultimele10oficiale (separat de formula campionatului)\n\n'+table(['Dată','Competiție','Partidă','FT90','primul gol','Gol70','FirstHalf','cornere','Sursă'],recent.filter(x=>x.team_id===team).map(m=>[m.date?.slice(0,10),m.competition,m.home+'–'+m.away,m.score_ft,m.goal_result==='pending'?'lipsă':m.first_goal_minute||'0–0',m.goal_result,m.actual_corners_1h===''?'lipsă':m.actual_corners_1h,m.corner_result,`[date](${m.source})`]));
 }return out;}
function weather(r){const p=b+'/Weather/'+r.id+'.json';if(!fs.existsSync(p))return 'Prognoză independentă lipsă.';const w=read(p);if(!w.ok)return 'OpenMeteo indisponibil; prognozaAPI încontext_summary.json.';const hour=r.kickoff_ro.slice(0,13)+':00',ix=w.data.hourly.time.indexOf(hour);if(ix<0)return 'Ora prognozei indisponibilă.';return `OpenMeteo la${hour}RO:${w.data.hourly.temperature_2m[ix]}°C, ploaie${w.data.hourly.precipitation_probability[ix]}%,vânt${w.data.hourly.wind_speed_10m[ix]}km/h. Prognoză, fără inspecție teren. APIpoate diferi; unitatea vântuluiAPI nu este explicită.`;}
for(const k of ['goal','corner']){
 const rows=all[k];let out='# '+(k==='goal'?'GOAL_BY_70':'CORNERS_1H_UNDER_7_5')+' —9noiembrie2026\n\n'+common+(k==='goal'?goalContext:cornerContext)+lessons;
 out+='## Clasament complet al universului identificat\n\n'+table(['Rang original','Meci /RO','Brut','AR20','1/p matematic','Istoric unic','Teren','Rol/lipsă'],rows.map(r=>[r.rank,r.home+'–'+r.away+' '+r.kickoff_ro.slice(11,16),pct(r.p_raw),pct(r.p_stress),r.fair_odds?Number(r.fair_odds).toFixed(4):'lipsă',k==='goal'?r.first_goal_success+'/'+r.first_goal_n:r.unique_under+'/'+r.unique_n,k==='goal'?r.venue_success+'/'+r.venue_n:r.venue_under+'/'+r.venue_unique_n,r.analyst_selected==='true'?'candidat analist':r.p_raw?'comparat':'model indisponibil;FirstHalf lipsă']));
 for(const r of rows){out+='\n## '+r.home+'–'+r.away+' — numărători și sensibilitate\n\n'+sampleTable(r,k)+`\nλsezon=${r.lambda_season||'indisponibil'},λteren=${r.lambda_venue||'indisponibil'},λ50/50=${r.lambda_combined||'indisponibil'}. Numai-sezon${pct(r.season_only)},numai-teren${pct(r.venue_only)},brut${pct(r.p_raw)},stres${pct(r.p_stress)}. Fără recalibrare contextuală.\n\n`+weather(r)+'\n'+series(r,k);}
 out+='\n## Surse și limite\n\nJSON/API cu timestamp înMatchData,OfficialRecent,Context șiStatusAtSelection; HTMLpublic înSources/, index[sources.json](../Goal70_2026-11-09/sources.json). Istoric complet[historical_observations.csv](../Goal70_2026-11-09/historical_observations.csv); probabilitățile originale și reviziile inventarului sunt păstrate. FotMob este sursa principală a observațiilorFirstHalf; reconciliereaFirstHalf+SecondHalf=All nu echivalează cu verificare independentă. Nu există preț disponibil/profit demonstrat, pariu plasat sau nouă alegere în numele utilizatorului.\n';
 fs.writeFileSync((k==='goal'?b+'/EUROPE_ALL_MATCHES.md':c+'/ANALYSIS.md'),out);
}
save(b+'/final_status.json',status.map(x=>({...x,verified_at_ro:new Date(Date.parse(x.verified_at_utc)+7200000).toISOString().replace('Z','+02:00')})));
const missing=['Review:3FirstHalfCyprus8Nov pending;4olderCyprusFirstHalf andSerbia not final; independent exact FirstHalf corroboration incomplete','ESPN unsupported4leagues; Romania empty discrepancy; complete independent schedule absence checks unavailable','LasPalmas/Eldense1historicalFirstHalf missing; only3corner models from4fixtures','80recent rows:9FirstHalf and1goal-list observation unavailable; lineups and full coach-change chronology not confirmed','Vålerenga12-player illness recovery not confirmed; NO BETGOAL70; probabilities uncalibrated'];
save('Analysis/Automation/run_report_'+date+'.json',{date,review_status:'partial',goal_research_status:'partial',corner_research_status:'partial',dashboard_status:'partial',missing_data:missing,review_path:R+'/REVIEW.md',lessons_path:R+'/LESSONS_FOR_NEXT_SCAN.md',goal_path:b+'/EUROPE_ALL_MATCHES.md',corner_path:c+'/ANALYSIS.md',...start,inventory_leagues:29,fixtures:4,goal_models:4,corner_models:3,historical_matches:hist.length,goal_decision:'NO BET',corner_candidate:'5868534',principal_individual:'5868534',original_prediction_values_preserved:true});
fs.writeFileSync(H+'/'+date+'.md',`# Jurnal PRED —9noiembrie2026

Cerere: execuția zilnică autorizată pentru ambele piețe, fără agenți paraleli, cote bookmaker, pariuri, mize, notificări externe, modele/selection_engine modificate, programare, commit/push. Ora verificată inițial08:57RO.

CSV/evenimente citite: alegeri8noiembrieBrighton–BrentfordGol70 șiGirona–AlmeríaCornere. Rezultate verificate: primul15 șiFirstHalf5, ambele succes. CSV actualizat sublock și scriereatomică, probabilități/ranguri/timestampuri păstrate. UniversGol7087/102,15fail;cornere87/99,12fail,3pending;2cancelledfurnizor șiLasPalmasreprogramat9noiembrie. Nacional–Viseurestant7noiembrie actualizat separat, alteFirstHalfCipru/Serbia rămânrestante.

Review/lecții predate${start.review_handoff_at_ro}; cercetare${start.research_started_at_ro}. Corectat editorial preambulul lecțiilor copiat din7noiembrie, fără modificarea rezultatelor/checklistului deja salvat; inițialul păstrat. Review complet include fiecarefail, succese, aproape de prag și riscurile originale. Nicio cauză tactică inventată saucalibrare nouă.

Inventar29ligi: inițial3partide, apoi4 după descoperireaLasPalmas–Eldense dinESPN+club;4modeleGol70/3cornere,${hist.length}istorice. Toate eligibilele calculate înaintea finalistului. NO BETGol70: liderKristiansund90,90%/85,30%are focar12bolnaviVIFconfirmatNFF, recuperare/lotneconfirmate. CornereSabadell–RealSociedadB21:30RO94,84%/88,57%,fair1,0544,rang1,24/24și11/11; principalindividual. Riscuri:Sanzsuspendat,Quadriincert,verticalitateSanse. Nicio alegere nouă a utilizatorului.

Lecții aplicate: brut/istoric/AR20,GF70propriu,tardive/0–0,intern separat de cupe,FirstHalfexact7/≥8/dispersie,loturiactualeversusXIvechi,status/programindependent,odihnă/meteo. Focarmedical și diferența de program schimbă argumentarea, fără procente ajustate.

Cercetarepartial: surseindependenteincomplete,1FirstHalfistoricLasPalmaslipsă,80oficialerecente cu9FirstHalfși1gollipsă, titulari/cronologieantrenori/recuperareVIFneconfirmate. Surse și calcule în rapoarte și run_report. Urmeazărebuild/verificare/exportmesaje vizibile, apoi10noiembrie rezultate→review/lecții→scanare. Nu presupunemprofit sauprețdisponibil.
`);
console.log(JSON.stringify({goal:'NO BET',corner:{id:chosen.id,p:chosen.p_raw,stress:chosen.p_stress,fair:chosen.fair_odds},start,missing}));
