param([string]$RecordedAtUtc)
$ErrorActionPreference='Stop'
$root=$PSScriptRoot;$ci=[Globalization.CultureInfo]::InvariantCulture
function Num($v){[double]::Parse([string]$v,$ci)}
function Fmt($v,[int]$n=2){([double]$v).ToString(('F'+$n),$ci).Replace('.',',')}
function Pct($v){(Fmt (100*(Num $v)))+'%'}
function Cdf([double]$lambda){$v=[math]::Exp(-$lambda);$sum=$v;for($i=1;$i -le 7;$i++){$v*=$lambda/$i;$sum+=$v};$sum}
$tz=[TimeZoneInfo]::FindSystemTimeZoneById('GTB Standard Time')
$stamp=[TimeZoneInfo]::ConvertTime([DateTimeOffset]::Parse($RecordedAtUtc),$tz).ToString('yyyy-MM-ddTHH:mm:sszzz')
$models=@(Import-Csv (Join-Path $root 'all_models.csv') -Encoding UTF8)
$h=@(Import-Csv (Join-Path $root 'historical_corners_1h.csv') -Encoding UTF8)
$coverage=@(Import-Csv (Join-Path $root 'coverage.csv') -Encoding UTF8)
$sens=@(Import-Csv (Join-Path $root 'sensitivity.csv') -Encoding UTF8)
$x=@($models|Where-Object id -eq '5836912')[0]
$official=@($sens|Where-Object variant -eq 'exclude_derby_stoke_conflicting_observation')[0]
$sampleAll=@($h|Where-Object {$_.league -eq 'eng.2' -and ($_.home_id -in @('10170','9850') -or $_.away_id -in @('10170','9850'))})
$sample=@($sampleAll|Where-Object id -ne '5836877')
if($sample.Count -ne 25 -or @($sample|Where-Object under_7_5 -eq 'True').Count -ne 24){throw 'Final sample integrity mismatch'}
$sample|Export-Csv (Join-Path $root 'selected_history.csv') -NoTypeInformation -Encoding UTF8
[pscustomobject]@{id='5836877';home='Derby County';away='Stoke City';date='2026-10-17';reason='Unresolved provider first-half count: FotMob 2-3, ESPN 3-3. Third source supports 5 total, but observation excluded from final estimate.';source_fotmob='https://www.fotmob.com/api/data/matchDetails?matchId=5836877';source_espn='https://site.api.espn.com/apis/site/v2/sports/soccer/eng.2/summary?event=401880217';source_third='https://playerstats.football/fixture/derby-county/stoke-city/2026-10-17'}|Export-Csv (Join-Path $root 'selected_exclusions.csv') -NoTypeInformation -Encoding UTF8
function Stat($rows,[string]$team){
 $cf=0;$ca=0;foreach($r in $rows){if($r.home_id -eq $team){$cf += [int]$r.home_corners_1h;$ca += [int]$r.away_corners_1h}else{$cf += [int]$r.away_corners_1h;$ca += [int]$r.home_corners_1h}}
 $n=@($rows).Count;$mean=($cf+$ca)/$n;$ss=0;foreach($r in $rows){$ss += [math]::Pow([int]$r.total_1h-$mean,2)}
 [pscustomobject]@{n=$n;cf=$cf;ca=$ca;cf_mean=$cf/$n;ca_mean=$ca/$n;mean=$mean;variance=$ss/($n-1);under=@($rows|Where-Object under_7_5 -eq 'True').Count;seven=@($rows|Where-Object exactly_7 -eq 'True').Count;eight=@($rows|Where-Object eight_plus -eq 'True').Count}
}
$aRows=@($sample|Where-Object {$_.home_id -eq '10170' -or $_.away_id -eq '10170'})
$bRows=@($sample|Where-Object {$_.home_id -eq '9850' -or $_.away_id -eq '9850'})
$avRows=@($sample|Where-Object home_id -eq '10170');$bvRows=@($sample|Where-Object away_id -eq '9850')
$a=Stat $aRows '10170';$b=Stat $bRows '9850';$av=Stat $avRows '10170';$bv=Stat $bvRows '9850'
$l=New-Object System.Collections.Generic.List[string]
function Add([string]$s=''){$l.Add($s)}
Add '# Derby-Norwich: sub 7,5 cornere prima repriza, 3 noiembrie 2026'
Add
Add ('**Candidatul ales pentru analiza cornerelor: Derby County - Norwich City, 21:45 RO / 19:45 GMT, Pride Park, Championship etapa 14.** Eveniment: maximum 7 cornere cumulate in prima repriza, inclusiv 45+. Analiza salvata la '+$stamp+'. Model brut pe esantionul final verificat: '+(Pct $official.p_raw)+'; stres intensitate +20%: '+(Pct $official.p_stress)+'. Probabilitatile nu sunt calibrate; stresul nu este un interval de incredere.')
Add
Add 'Cererea curenta autorizeaza aceasta analiza suplimentara. QPR-Burnley GOAL_BY_70 ramane alegerea deja confirmata de utilizator. Derby-Norwich la cornere este alegerea analistului pentru cererea curenta, nu o noua selectie confirmata de utilizator, nu o inlocuire a QPR si nu un bilet combinat. Protocolul general nu este modificat permanent; nu presupun miza ori pariu plasat.'
Add
Add '## Acoperirea cercetarii'
Add
Add 'Am pornit de la inventarul zilei pentru primele ligi masculine ale topului UEFA 2026, Romania, Serbia si cele 7 divizii secunde urmarite: 29 ligi. Sunt 10 partide in universul de baza (9 Championship si una Allsvenskan); am adaugat 9 Champions League si una Cupa Danemarcei. Inventarul si limitarile lui sunt in `fixture_universe_reference.csv` si `fixture_exclusions_reference.csv`, cu sursele din analiza Gol70 a aceleiasi zile. Pentru patru ligi fara suport ESPN, absenta partidelor este bazata pe FotMob, nu pe HTTP 400.'
Add
Add 'Au fost cercetate toate cele 20 de partide identificate, inaintea alegerii. Au rezultat 19 calcule de screening din statistici explicite FirstHalf; Shakhtar-Sporting este exclus numeric, fiindca in cele 11 meciuri Shakhtar lipsesc cornerele 1H. Nu inlocuiesc lipsurile cu zero. 417 partide istorice distincte asteptate din 15 ligi pentru cele 40 echipe: 406 au 1H, 2H si FT reconciliate separat pe echipa; 11 nu au 1H. Nu numesc aceste 19 modele un top al tuturor diviziilor europene.'
Add
Add '[Program FotMob](https://www.fotmob.com/api/data/matches?date=20261103). Statusurile tuturor celor 20 de meciuri au fost reverificate la 11:15:27 RO: toate neincepute, ore neschimbate, `current_status.json`. Prima selectie rezulta dupa compararea celor 19 modele eligibile; verificarile contextuale aprofundate se concentreaza pe finaliste.'
Add
Add '## Controlul surselor si observatia exclusa'
Add
Add 'FotMob este sursa statistica principala: content.stats.Periods.FirstHalf, cheia corners, incluzand timpul aditional. FirstHalf + SecondHalf = All este o verificare interna; nu demonstreaza independent corectitudinea cifrelor.'
Add
Add 'Pentru toate cele 26 de meciuri de campionat Derby/Norwich am colectat si cronologia ESPN. Evenimentele unice corner-awarded sunt numarate pe perioada 1, inclusiv 45+, si reconciliate cu totalurile FT ale ESPN. In 25/26 totalurile 1H coincid. In 23/26 coincid si ambele reprize si FT; la Derby-Wrexham si Norwich-WBA diferentele sunt exclusiv in a doua repriza/FT, iar 1H coincide. Sunt pastrate in `provider_comparison.csv` si `ESPN/*.json`; doua fluxuri concordante pot avea surse comune.'
Add
Add 'La Derby-Stoke, 17 octombrie, FotMob are 2-3 la pauza, ESPN 3-3. [Cronologia PlayerStats/Sportmonks](https://playerstats.football/fixture/derby-county/stoke-city/2026-10-17) arata al cincilea corner la 45+3 si apoi repriza a doua, sustinand totalul 5. ESPN include un corner suplimentar Derby la 45+4. Nu am verificat video daca a fost acordat si neexecutat. **Exclud aceasta partida din estimarea finala**, fara sa aleg valoarea mai avantajoasa. Pastrez separat screeningul FotMob, varianta ESPN si varianta cu observatia exclusa. Clasamentul de screening de mai jos nu este rescris retrospectiv.'
Add
Add '## Datele folosite in estimarea finala'
Add
Add 'Dupa excludere: Derby 12/13 meciuri de sezon verificabile fara conflict, Norwich 13/13; pe teren Derby 5/6 acasa, Norwich 6/6 deplasare. Acestea depasesc minimul exploratoriu fixat de 5 pe sezon si 3 pe teren, dar sezonul Derby nu este complet in estimarea finala. Valorile lipsa nu sunt zero.'
Add
Add '| Esantion | N | Obtinute 1H | Cedate 1H | Media obtinute | Media cedate | Media TOTALULUI | >=8 | Exact 7 | Varianta totalului |'
Add '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
foreach($pair in @(@('Derby sezon verificat',$a),@('Norwich sezon',$b),@('Derby acasa verificat',$av),@('Norwich deplasare',$bv))){$s=$pair[1];Add ('| '+$pair[0]+' | '+$s.n+' | '+$s.cf+' | '+$s.ca+' | '+(Fmt $s.cf_mean)+' | '+(Fmt $s.ca_mean)+' | '+(Fmt $s.mean)+' | '+$s.eight+'/'+$s.n+' | '+$s.seven+'/'+$s.n+' | '+(Fmt $s.variance)+' |')}
Add
Add 'Meciuri distincte finale: 24/25 sub 7,5 (96,00%), unul cu minimum 8 si unul cu exact 7. Configuratia terenului: 10/11 sub 7,5 (90,91%), un esec, niciun total exact 7. Acestea sunt frecvente observate, nu probabilitati pentru azi. Derby-Norwich nu s-a jucat in acest sezon, deci nu exista o intalnire directa dubla in agregare.'
Add
Add 'Singurul esec este Derby-Cardiff, 22 august: 5-5 la pauza, total 10. Sheff Utd-Derby, 24 octombrie, a avut 1-6 la pauza, total 7, deci succes aproape de limita. Aceste partide raman in date; nu elimin esecul sau succesul fragil pentru a imbunatati procentul.'
Add
Add '## Calculul fix si toate sensibilitatile'
Add
Add '```text'
Add 'lambda_sezon = ((21/12 + 19/13) + (23/13 + 24/12))/2 = 3,490384615'
Add 'lambda_teren = ((10/5 + 11/6) + (10/6 + 11/5))/2 = 3,85'
Add 'lambda_combinat = 0,5*lambda_sezon + 0,5*lambda_teren = 3,670192308'
Add 'P(total 1H <=7) = exp(-lambda)*SUM(lambda^k/k!, k=0..7) = 96,6128976%'
Add 'Stres: lambda*1,2 = 4,404230769; P(total 1H <=7) = 92,1091154%'
Add 'Cota echitabila matematica = 1/P = aproximativ 1,0351'
Add '```'
Add
Add ('Numai sezon: '+(Pct (Cdf (Num $official.lambda_season)))+'; numai teren: '+(Pct (Cdf (Num $official.lambda_venue)))+'. Ponderarea 50/50 este euristica, esantioanele se suprapun. Cota echitabila nu este o oferta disponibila. Nu am consultat cote sau probabilitati ale tipsterilor.')
Add
Add '| Varianta, aceeasi formula | N sezon A/B | N teren A/B | Model brut | Stres +20% |'
Add '|---|---:|---:|---:|---:|'
foreach($s in $sens){Add ('| '+$s.variant+' | '+$s.home_n+'/'+$s.away_n+' | '+$s.home_venue_n+'/'+$s.away_venue_n+' | '+(Pct $s.p_raw)+' | '+(Pct $s.p_stress)+' |')}
Add
Add 'Randul oficial pentru estimarea finala este exclude_derby_stoke_conflicting_observation. Celelalte sunt diagnostice. Excluderea observatiei creste numeric modelul de la 96,35% la 96,61%; **nu creste automat increderea** si nu este o imbunatatire demonstrata a modelului. Folosind valoarea ESPN, rezultatul ar fi 96,05% si stresul 90,96%. Selectia ramane prima numeric si in aceasta varianta fata de QPR-Burnley (95,93%), dar diferenta este prea mica pentru a demonstra superioritate reala.'
Add
Add 'Sensibilitatile la eliminari din tabel folosesc screeningul FotMob ca baza. Unicul rosu in prima repriza a fost WBA la 34 in Derby-WBA, care a avut total 4 la pauza. Eliminarea de la 66 din Sheff Utd-Derby si cele din Norwich-Burnley/Millwall-Norwich, la 77/64, sunt dupa pauza; nu pot explica cauzal cornerele primei reprize. Nu modific procentul printr-un bonus/penalizare arbitrar.'
Add
Add '**Riscul de model:** la Derby acasa, dupa excludere, varianta totalurilor este 13,20 fata de media 4,20. Seria este mai dispersata decat presupune Poisson; o medie mica poate ascunde reprize cu multe cornere grupate. Modelul 96,61% este peste frecventa pe teren de 90,91%, iar esantionul este mic. Nu pot echivala aceste calcule cu o rata reala validata de 96%.'
Add
Add '## Istoric la pauza: ultimele 10 meciuri de campionat'
foreach($id in @('10170','9850')){
 Add
 Add $(if($id -eq '10170'){'### Derby'}else{'### Norwich'})
 Add
 Add '| Data | Partida | Cornere 1H gazda-oaspete | Total 1H | Incadrare |'
 Add '|---|---|---:|---:|---|'
 foreach($r in $sampleAll|Where-Object {$_.home_id -eq $id -or $_.away_id -eq $id}|Sort-Object date -Descending|Select-Object -First 10){
  $conflict=$r.id -eq '5836877'
  Add ('| '+$r.date.Substring(0,10)+' | ['+$r.home+' - '+$r.away+']('+$r.source+') | '+$(if($conflict){'2-3 / 3-3'}else{$r.home_corners_1h+'-'+$r.away_corners_1h})+' | '+$(if($conflict){'5 / 6'}else{$r.total_1h})+' | '+$(if($conflict){'Exclus: conflict intre surse'}elseif($r.under_7_5 -eq 'True'){'Sub 7,5'}else{'Esec: >=8'})+' |')
 }
}
Add
Add 'In ultimele 10 de campionat Derby exista 9 observatii utilizate si una exclusa; cele 9 au reusit, una cu exact 7. Norwich are 10/10 sub prag, maxim 5. Nu prezint aceste meciuri drept ultimele 10 din toate competitiile: Norwich a jucat si in EFL Cup. Predictia foloseste numai Championship 2026/27.'
Add
Add 'Seria Derby acasa de la cel mai recent, cu lipsa marcata: [exclus 5/6], 4, 3, 4, 0, 10. Norwich deplasare: 2, 3, 4, 5, 2, 5. Observatia 0 a lui Derby este confirmata de perioada FirstHalf si cronologia ESPN, nu un camp lipsa.'
Add
Add '## Context, tactica, loturi si conditii'
Add
Add 'Campionat regulat, etapa 14: Derby locul 23, 7 puncte; Norwich locul 15, 15 puncte. Gazdele cauta puncte pentru salvare, oaspetii a treia victorie consecutiva. Nu exista retur eliminatoriu, scor agregat sau prelungiri; egalul este posibil, dar nu o calificare. [Context si clasament contemporan](https://stadiro.com/matches/england/championship/2026-11-03/derby-v-norwich/). Nici motivatia, nici golurile nu determina singure numarul cornerelor.'
Add
Add 'John Eustace/Philippe Clement sunt antrenorii actuali. Ultimele module verificate sunt 4-2-3-1 pentru ambele; titularii de azi nu sunt confirmati. Clement descrie Derby ca echipa directa, cu dueluri importante, si cere control cu mingea; recunoaste nevoia evaluarii prospetimii si posibilitatea schimbarilor dupa Watford. Papa Diallo lipseste, Crnac nu este pregatit sa inceapa dupa accidentarea lunga, iar Brooks are un volum neobisnuit de minute. [Conferinta oficiala Norwich, 2 noiembrie](https://www.canaries.co.uk/news/press-conference-clement-im-looking-forward-game-its-going-be-hard-battle).'
Add
Add 'Eustace anunta ca Ward, Ozoh si Meister necesita evaluare, iar Clarke si Agyemang au revenit la antrenament complet; revenirea la antrenament nu confirma titularizarea. [Relatarea actualizata cu declaratiile antrenorului](https://staging-aws.sports.yahoo.com/articles/derby-county-duo-closing-returns-170400754.html). Agregatoarele care ii dau in continuare pe toti absentii drept cert indisponibili sunt in conflict cu aceasta actualizare; nu folosesc o lista veche ca dovada a unei reduceri sigure a cornerelor.'
Add
Add 'Interpretare tactica, fara ajustari procentuale: jocul direct Derby poate trimite atacurile in dueluri centrale si faze relativ scurte; controlul Norwich ar putea tempera schimburile. Invers, daca Norwich fixeaza gazdele jos si foloseste repetat benzile, ori Derby urmareste scorul prin centrari, respingerile pot produce cornere in rafale. Nu presupun ca 4-2-3-1 garanteaza underul si nu confund un gol timpuriu cu reducerea presiunii.'
Add
Add 'Insight verificabil din singurul esec: [cronologia ESPN Derby-Cardiff](https://site.api.espn.com/apis/site/v2/sports/soccer/eng.2/summary?event=401880326) arata cornere la 1, 3, 4, 7, 15, 17, 18, 18, 44 si 45+3. Pragul era depasit la 18, inaintea primului gol de la 19. Riscul nu se reduce la scenariul unui gol timpuriu: cornerele se pot grupa chiar la 0-0.'
Add
Add 'Ambele echipe au jucat pe 31 octombrie, aproximativ trei zile pana azi; meci intern, fara schimbare de fus orar. Pride Park are gazon natural; nu am inspectie a starii suprafetei ori itinerarul exact al oaspetilor. [Met Office, Derby, 3 noiembrie 04:00 UTC](https://weather.metoffice.gov.uk/forecast/gcqvn6pq4?new-design=false) indica 11 C maxima, 7 C minima, seara adesea uscata, posibilitate de averse izolate. Nu exista un motiv meteo verificat de veto; prognoza nu garanteaza conditiile efective.'
Add
Add '## De ce aceasta partida fata de alternative'
Add
Add 'Derby-Norwich conducea deja screeningul la 96,35%, inaintea filtrarii observatiei conflictuale. Are 25/26 sub prag in screening si 24/25 dupa excludere; QPR-Burnley are 23/26, dar 12/12 pe teren fata de 10/11 la Derby-Norwich. Acesta este un contraargument pentru QPR, nu il ascund. QPR are si schimbarea recenta de antrenor Burnley, care reduce comparabilitatea istoricului. Diferenta de model este mica, nu o dovada de superioritate reala a lui Derby-Norwich.'
Add
Add 'Manchester United-Roma: 94,51% / 87,94%, 2/19 reprize >=8; Atletico-Bayern 93,62% / 86,27%, 3/19 >=8. Sunt proiectii ale datelor interne in Champions League, cu esantioane mici pe teren, nu probabilitati UCL validate. Nu am facut pentru ele aceeasi verificare contextuala aprofundata ca pentru Derby. Portsmouth-Swansea: 92,06% / 83,44%, 4/26 esecuri si 3/12 in configuratia terenului. Sirius-Halmstad: 75,45% / 58,27%, 9/51 esecuri si 14/51 reprize cu exact 7; nu este o alternativa buna in aceasta comparatie.'
Add
Add '## Screening integral: statistica principala FotMob, inaintea verificarii finale'
Add
Add 'UCL/cupa folosesc datele campionatelor interne, nu un model validat pentru competitia zilei. Toate calculele eligibile sunt facute; doar finalistul are filtrul suplimentar documentat. Procentele sunt conditionate de sursa si de model, nu recomandari multiple.'
Add
Add '| Rang | Partida | Competitie | Ora RO | Model brut | Stres +20% | >=8 distincte | Exact 7 distincte |'
Add '|---:|---|---|---|---:|---:|---:|---:|'
$rank=0
foreach($m in $models){$rank++;$time=[TimeZoneInfo]::ConvertTime([DateTimeOffset]::Parse($m.date),$tz).ToString('HH:mm');Add ('| '+$rank+' | '+$m.home+' - '+$m.away+' | '+$m.competition+' | '+$time+' | '+$(if($m.eligible -eq 'True'){Pct $m.p_raw}else{'N/A: lipsa 1H Shakhtar'})+' | '+$(if($m.eligible -eq 'True'){Pct $m.p_stress}else{'N/A'})+' | '+$(if($m.eligible -eq 'True'){$m.unique_8plus+'/'+$m.unique_n}else{'N/A'})+' | '+$(if($m.eligible -eq 'True'){$m.unique_7+'/'+$m.unique_n}else{'N/A'})+' |')}
Add
Add '## Evidenta prospectiva si limite'
Add
Add 'Estimarea finala oficiala este pe cele 25 observatii fara conflict 1H, cu Derby-Stoke exclus: 96,61% brut, 92,11% stres. Nu este un procent calibrat; frecventa pe teren este 90,91%, varianta mare Derby ramane riscul principal. Cota echitabila este circa 1,0351; disponibilitatea unui pret si profitabilitatea nu sunt demonstrate. Nu stabilesc miza si nu combin cu GOAL_BY_70.'
Add
Add 'Fisiere: all_models.csv (screening initial al celor 20), historical_corners_1h.csv (406 statistici reconciliate intern), rejected_corners.csv (11 lipsuri Shakhtar), coverage.csv, selected_history.csv (25 observatii finale), selected_exclusions.csv (conflict Derby-Stoke), provider_comparison.csv, sensitivity.csv, current_status.json, verification.json. Nicio observatie ori selectie nu contine rezultat viitor; productia si selection_engine raman neschimbate.'
$l|Set-Content (Join-Path $root 'ANALYSIS.md') -Encoding UTF8
$selection=[pscustomobject]@{analysis_date='2026-11-03';recorded_at_ro=$stamp;selected_by='analyst';market='CORNERS_1H_UNDER_7_5';event_definition='Both teams combined first-half corners <=7, including added time';home='Derby County';away='Norwich City';match_id='5836912';goal70_espn_id='401880182';kickoff_ro='2026-11-03T21:45:00+02:00';screening_p_raw=$x.p_raw;screening_p_stress=$x.p_stress;final_p_raw=$official.p_raw;final_p_stress=$official.p_stress;fair_odds=1/(Num $official.p_raw);historical_success=24;historical_n=25;venue_success=10;venue_n=11;excluded_conflict_id='5836877';calibrated=$false;user_selected=$false;market_price='';stake='';actual_corners_1h='';result='';status='prospective_additional_analysis_only';analysis_path='Analysis/Corners1H_2026-11-03/ANALYSIS.md'}
$selection|Export-Csv (Join-Path $root 'prospective_selection.csv') -NoTypeInformation -Encoding UTF8
$verification=Get-Content (Join-Path $root 'verification.json') -Raw -Encoding UTF8|ConvertFrom-Json
$verification|Add-Member -NotePropertyName final_prediction_basis -NotePropertyValue 'Derby-Stoke excluded from final prediction; source-specific screening preserved' -Force
$verification|Add-Member -NotePropertyName final_p_raw -NotePropertyValue (Num $official.p_raw) -Force
$verification|Add-Member -NotePropertyName final_p_stress -NotePropertyValue (Num $official.p_stress) -Force
$verification|Add-Member -NotePropertyName final_unique_n -NotePropertyValue 25 -Force
$verification|ConvertTo-Json -Depth 5|Set-Content (Join-Path $root 'verification.json') -Encoding UTF8
'Saved final analysis: Derby-Norwich; 25 observations after excluding the first-half count conflict.'
