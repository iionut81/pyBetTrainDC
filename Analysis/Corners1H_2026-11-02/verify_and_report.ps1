$ErrorActionPreference='Stop'
$culture=[Globalization.CultureInfo]::InvariantCulture
function Number($v){[double]::Parse([string]$v,$culture)}
function Pct($v){(100*[double]$v).ToString('F2',$culture).Replace('.',',')+'%'}
function Decimal($v){([double]$v).ToString('F4',$culture).Replace('.',',')}
function Check($v,[string]$s){if(-not $v){throw $s}}
function Cdf([double]$lambda){$sum=0;$factorial=1;for($k=0;$k -le 7;$k++){if($k -gt 0){$factorial*=$k};$sum += [math]::Pow($lambda,$k)/$factorial};[math]::Exp(-$lambda)*$sum}
$h=@(Import-Csv (Join-Path $PSScriptRoot 'historical_corners_1h.csv') -Encoding UTF8)
$models=@(Import-Csv (Join-Path $PSScriptRoot 'all_models.csv') -Encoding UTF8)
$coverage=@(Import-Csv (Join-Path $PSScriptRoot 'coverage.csv') -Encoding UTF8)
$current=Get-Content (Join-Path $PSScriptRoot 'current_status.json') -Raw -Encoding UTF8|ConvertFrom-Json
Check ($models.Count -eq 27 -and $coverage.Count -eq 15 -and $h.Count -eq 575) 'Coverage count changed'
Check (@($coverage|Where-Object {[int]$_.not_fetched -ne 0 -or [int]$_.rejected -ne 0}).Count -eq 0) 'Incomplete history'
Check (@($h|Group-Object league,id|Where-Object Count -gt 1).Count -eq 0) 'Duplicate observations'
foreach($r in $h){
 Check ($r.period -eq 'FirstHalf' -and [DateTimeOffset]::Parse($r.date) -lt [DateTimeOffset]::Parse('2026-11-02T00:00:00+02:00')) ('Invalid period/cutoff '+$r.id)
 $total=[int]$r.home_corners_1h+[int]$r.away_corners_1h
 Check ($total -eq [int]$r.total_1h -and ($r.under_7_5 -eq 'True') -eq ($total -le 7) -and ($r.eight_plus -eq 'True') -eq ($total -ge 8)) ('Outcome mismatch '+$r.id)
 foreach($side in @('home','away')){Check (([int]$r.($side+'_corners_1h')+[int]$r.($side+'_corners_2h')) -eq [int]$r.($side+'_corners_ft')) ('Period reconciliation '+$r.id)}
}
function Counts($rows,[string]$team){
 $cf=0;$ca=0;foreach($r in $rows){if($r.home_id -eq $team){$cf += [int]$r.home_corners_1h;$ca += [int]$r.away_corners_1h}else{$cf += [int]$r.away_corners_1h;$ca += [int]$r.home_corners_1h}}
 [pscustomobject]@{n=@($rows).Count;cf=$cf;ca=$ca}
}
foreach($x in $models){
 Check ($x.eligible -eq 'True') ('Ineligible candidate '+$x.id)
 $aRows=@($h|Where-Object {$_.league -eq $x.league -and ($_.home_id -eq $x.home_id -or $_.away_id -eq $x.home_id)})
 $bRows=@($h|Where-Object {$_.league -eq $x.league -and ($_.home_id -eq $x.away_id -or $_.away_id -eq $x.away_id)})
 $a=Counts $aRows $x.home_id;$b=Counts $bRows $x.away_id
 $av=Counts @($aRows|Where-Object home_id -eq $x.home_id) $x.home_id
 $bv=Counts @($bRows|Where-Object away_id -eq $x.away_id) $x.away_id
 foreach($pair in @(@('home',$a),@('away',$b),@('home_venue',$av),@('away_venue',$bv))){
  foreach($field in @('n','cf','ca')){Check ([int]$x.($pair[0]+'_'+$field) -eq [int]$pair[1].$field) ('Input mismatch '+$x.id)}
 }
 $ls=(($a.cf+$a.ca)/$a.n+($b.cf+$b.ca)/$b.n)/2
 $lv=(($av.cf+$av.ca)/$av.n+($bv.cf+$bv.ca)/$bv.n)/2
 $lc=($ls+$lv)/2
 Check ([math]::Abs((Cdf $lc)-(Number $x.p_raw)) -lt 1e-12 -and [math]::Abs((Cdf (1.2*$lc))-(Number $x.p_stress)) -lt 1e-12) ('Probability mismatch '+$x.id)
 $s=@($current|Where-Object id -eq $x.id)
 Check ($s.Count -eq 1 -and $s[0].started -ne $true -and $s[0].finished -ne $true -and $s[0].cancelled -ne $true -and
  [DateTimeOffset]::Parse($s[0].date) -eq [DateTimeOffset]::Parse($x.date) -and [DateTimeOffset]::Parse($s[0].date) -gt [DateTimeOffset]::Parse($s[0].verified_at_utc)) ('Status/kickoff changed '+$x.id)
}
$selected=@($models|Where-Object id -eq '1000015166')[0]
$rows=@($h|Where-Object {$_.league -eq $selected.league -and ($_.home_id -in @($selected.home_id,$selected.away_id) -or $_.away_id -in @($selected.home_id,$selected.away_id))})
Check ($rows.Count -eq 25 -and @($rows|Where-Object {[int]$_.total_1h -ge 7}).Count -eq 0) 'Selected sample mismatch'
$tz=[TimeZoneInfo]::FindSystemTimeZoneById('GTB Standard Time')
$timestamp=[TimeZoneInfo]::ConvertTime([DateTimeOffset]::Parse($current[0].verified_at_utc),$tz).ToString('yyyy-MM-ddTHH:mm:sszzz')
$lines=New-Object System.Collections.Generic.List[string]
function Add([string]$s=''){$lines.Add($s)}
Add '# Sub 7,5 cornere prima repriza: 2 noiembrie 2026'
Add
Add ('**Selectia: Radnicki Nis - Zeleznicar Pancevo, 19:00 RO / 18:00 CET.** Cel mult 7 cornere cumulate pana la pauza, inclusiv timpul aditional. Model Poisson brut **96,84%**; stres intensitate +20% **92,58%**. Actualizare program: '+$timestamp+'. Partida nu a inceput. O singura selectie pentru aceasta analiza; nu este combinata cu golul pana la 70 si nu se stabileste o miza.')
Add
Add '## Acoperire si sursa'
Add
Add 'Au fost colectate toate cele 27 de partide din programul Europei stabilit azi: 16 din primele ligi masculine ale primelor 20 tari UEFA 2026, plus 11 din Romania, Serbia si diviziile secunde europene urmarite. Sunt 15 ligi active; celelalte prime ligi nu au partide identificate azi. Toate cele 27 au inputuri complete pe sezonul curent pentru echipele implicate, minimum 5 meciuri pe sezon si 3 in configuratia terenului. 575 meciuri istorice DISTINCTE, toate cu cornere FirstHalf, SecondHalf si All reconciliate separat pe echipa. Niciun rand lipsa nu este inlocuit cu zero.'
Add
Add 'Istoricul local anterior se oprea in mai 2026 si nu a fost folosit pentru selectia curenta. Sursa noua: [FotMob, programul zilei](https://www.fotmob.com/api/data/matches?date=20261102), endpoint matchDetails, content.stats.Periods.FirstHalf, key=corners. Este sursa numerica principala pentru 1H; reconcilierea reprizelor este control intern, nu verificare independenta a fiecarei observatii. Pentru ultimele doua partide ale finalistilor, totalurile FT au fost confruntate cu un alt furnizor: 3-6 si 8-5, dar aceasta nu constituie verificare separata a tuturor cifrelor 1H. [Nis-Partizan](https://scoremondo.com/en/match/fk-radnicki-nis-fk-partizan-belgrade-31356004), [Radnicki 1923-Zeleznicar](https://scoremondo.com/en/match/fk-radnicki-1923-fk-zeleznicar-pancevo-31356001).'
Add
Add '## Inputurile selectiei'
Add
Add '| Esantion | N | Cornere obtinute 1H | Cornere cedate 1H | Media TOTALULUI 1H | Exact 7 | Minimum 8 |'
Add '|---|---:|---:|---:|---:|---:|---:|'
foreach($q in @(@('Radnicki sezon','home'),@('Zeleznicar sezon','away'),@('Radnicki acasa','home_venue'),@('Zeleznicar afara','away_venue'))){
 $p=$q[1];Add ('| '+$q[0]+' | '+$selected.($p+'_n')+' | '+$selected.($p+'_cf')+' | '+$selected.($p+'_ca')+' | '+(Decimal (Number $selected.($p+'_mean')))+' | '+$selected.($p+'_7')+' | '+$selected.($p+'_8plus')+' |')
}
Add
Add 'Radnicki: 13/13 reusite; Zeleznicar: 13/13. Meciul direct din 17 iulie apare in ambele serii, deci totalul distinct este **25/25**, nu 26. Configuratia terenului: **12/12**, sase pentru fiecare echipa. Nu exista reprize cu exact 7; maximul este 6 in seria Radnicki si 5 in seria Zeleznicar. Un asemenea istoric nu demonstreaza o probabilitate de 100% si nu elimina riscul unei serii neobisnuite de cornere.'
Add
Add '## Calculul fix, fara cote'
Add
Add '```text'
Add 'lambda_sezon = ((26/13 + 18/13) + (25/13 + 26/13))/2 = 3,653846'
Add 'lambda_teren = ((10/6 + 10/6) + (10/6 + 13/6))/2 = 3,583333'
Add 'lambda_combinat = 0,5*3,653846 + 0,5*3,583333 = 3,618590'
Add 'P(total <=7) = exp(-lambda) * SUM(lambda^k/k!, k=0..7) = 96,8414%'
Add 'Stres: lambda*1,2 = 4,342308; P(total <=7) = 92,5831%'
Add 'Cota echitabila matematica = 1/P = 1,032616'
Add '```'
Add
Add ('Numai sezon: '+(Pct (Cdf (Number $selected.lambda_season)))+'; numai teren: '+(Pct (Cdf (Number $selected.lambda_venue)))+'. Ponderarea 50/50 este euristica; esantioanele se suprapun. Poisson nu este calibrat sau validat pentru acest clasament zilnic, iar stresul nu este un interval de incredere. Cota echitabila nu este un pret disponibil si nu confirma pragul dorit de peste 1,10. Nu au fost folosite cote sau probabilitati ale tipsterilor.')
Add
Add 'Dispersie observata, varianta de esantion a totalurilor 1H: Radnicki sezon 1,50 fata de media 4,00; Zeleznicar sezon 1,56 fata de media 3,31. Pe teren: 1,77 fata de media 3,83 si 1,47 fata de media 3,33. In aceste serii variatia este mai mica decat media; nu adopt o alta distributie dupa observarea rezultatelor. Un meci cu ritm diferit poate depasi acest istoric restrans.'
Add
Add '## Ultimele 10 meciuri de campionat, totaluri la pauza'
foreach($id in @($selected.home_id,$selected.away_id)){
 Add
 Add $(if($id -eq $selected.home_id){'### Radnicki Nis'}else{'### Zeleznicar Pancevo'})
 Add
 Add '| Data | Partida | Cornere gazde-oaspeti 1H | Total 1H |'
 Add '|---|---|---:|---:|'
 foreach($r in ($rows|Where-Object {$_.home_id -eq $id -or $_.away_id -eq $id}|Sort-Object date -Descending|Select-Object -First 10)){
  Add ('| '+$r.date.Substring(0,10)+' | ['+$r.home+' - '+$r.away+']('+$r.source+') | '+$r.home_corners_1h+'-'+$r.away_corners_1h+' | '+$r.total_1h+' |')
 }
}
Add
Add 'Sunt meciuri de campionat, nu pretind o serie completa din toate competitiile. Restantele din 28 octombrie sunt incluse pe data REALA a disputarii, nu dupa numarul etapei. Ultima deplasare Zeleznicar a avut 13 cornere FT, dar 5 la pauza (2-3): nu am dedus 1H impartind totalul la doi.'
Add
Add '## Context si contraargumente'
Add
Add 'Superliga Serbia, etapa 14, campionat regulat; fara calificare pe scor agregat sau prelungiri. Radnicki are 8 puncte si cinci infrangeri consecutive, Zeleznicar 26 si cinci victorii consecutive. Scorul clasamentului nu impune automat multe cornere. [Program si statistici oficiale ale ligii](https://www.superliga.rs/utakmica/14-kolo-radnicki-zeleznicar/). Numele comercial al competitiei nu inseamna ca folosim cote de bookmaker.'
Add
Add 'Gazdele sunt antrenate acum de Nikola Drincic: declaratia din avancronica vorbeste despre noua antrenamente si trei partide de la venire. Acest lucru reduce comparabilitatea cu o parte din cele 13 meciuri. Sunt raportati absenti Bojan Kovacevic si Ranko Jokic. Nu exista inca un 11 confirmat. [Relatarea conferintei in presa locala](https://glasjuga.rs/radnicki-u-ponedeljak-docekuje-zeleznicar-ulaz-na-cair-besplatan-drincic-veruje-u-opstanak/).'
Add
Add 'Radomir Kokovic confirma al treilea meci consecutiv in deplasare intr-un interval scurt, oboseala si jucatori suspendati sau accidentati; nu nominalizeaza complet indisponibilii. Nu inventez o lista de absenti si nu presupun ca oboseala ajuta automat underul: poate reduce atacul sau poate provoca greseli, respingeri si presiune defensiva. [Conferinta oficiala Zeleznicar](https://fkzeleznicar.rs/kokovic-moramo-da-izadjemo-iz-zone-komfora/).'
Add
Add 'Modulele 4-2-3-1 Radnicki si 4-1-4-1 Zeleznicar apar ca ultime formatii cunoscute la meciurile din 28 octombrie in agregatoare; nu le prezint drept module confirmate pentru azi. Scenariul favorabil este o prima repriza cu atacuri limitate pe benzi si volume apropiate de istoricul sezonului. Scenariul nefavorabil: gol timpuriu, gazda nevoita sa urmareasca scorul, centrari repetate si suturi blocate, generand cornere in grupuri. Un gol nu reduce automat cornerele.'
Add
Add 'Ambele au jucat pe 28 octombrie, deci cinci zile calendaristice pana azi. S-a confirmat calatoria oaspetilor din Pancevo spre Nis. Ploaia la antrenamentul oaspetilor din 1 noiembrie este confirmata, dar NU este o prognoza de vreme la Nis in timpul partidei. Nu exista aici un raport verificat al gazonului sau prognoza orara; nu modific procentul pe aceste presupuneri. [Anuntul oficial despre deplasare](https://fkzeleznicar.rs/poslednji-trening-pred-polazak-u-nis/).'
Add
Add '## De ce aleg acest meci, desi este al treilea dupa procentul brut'
Add
Add 'Atromitos-Volos are raw 97,46%, stres 93,90%, dar numai 3 meciuri ale gazdelor acasa, 15 observatii distincte si un esec >=8. Volos si-a schimbat antrenorul pe 26 octombrie: intreg istoricul sau de campionat din inputuri este anterior lui Garrido. [Numire oficiala](https://www.volosfc.com/o-juan-carlos-garrido-neos-proponitis-toy-voloy-elabet/).'
Add
Add 'Sassuolo-Fiorentina are 96,8437%, practic egal cu 96,8414% al partidei alese: diferenta de 0,0023 puncte procentuale nu este relevanta pentru o asemenea precizie a datelor. Acolo sunt 18 observatii distincte si un esec >=8; aici sunt 25, niciun esec >=8 si nicio repriza de exact 7. Lorient-Brest are 96,81%, 16 observatii distincte si doar 8 pe teren. Voluntari-Farul are 96,45%, 26 observatii, dar doua reprize de exact 7 pe configuratia terenului; nu e la fel de confortabil fata de prag. Mallorca-Cadiz este alternativa cu raw 95,72%, stres 90,30%, dar un esec >=8 in sezon.'
Add
Add 'Alegerea finala foloseste aceasta judecata despre cantitatea datelor, proximitatea pragului si incertitudinile contextuale. Nu am modificat formula, procentele sau adaugat bonusuri. Nu afirm ca alegerea este demonstrata superior printr-un backtest al acestei reguli de preferinta.'
Add
Add '## Comparatia integrala, ordonata strict dupa model'
Add
Add '| Rang raw | Partida | P brut | Stres +20% | >=8 in sezon, meciuri distincte | Exact 7, sezon distinct |'
Add '|---:|---|---:|---:|---:|---:|'
$rank=0
foreach($x in $models){$rank++;Add ('| '+$rank+' | ['+$x.home+' - '+$x.away+']('+$x.source+') | '+(Pct (Number $x.p_raw))+' | '+(Pct (Number $x.p_stress))+' | '+$x.unique_8plus+'/'+$x.unique_n+' | '+$x.unique_7+'/'+$x.unique_n+' |')}
Add
Add '## Dovezi si reproducere'
Add
Add 'Date: historical_corners_1h.csv (575 meciuri), all_models.csv (27 inputuri si calcule), coverage.csv (15 ligi active), current_status.json (status actual pentru toate 27), FotMob/*.json (campuri sportive publice selectate, fara cote). Reproducere offline: powershell -NoProfile -File Analysis/Corners1H_2026-11-02/evaluate_corners.ps1; apoi verify_and_report.ps1. Nu se ruleaza retraining si nu se modifica productia.'
Add
Add 'Verificare PASS: 575 observatii unice in trecut, perioada FirstHalf explicita, reconcilierea 1H+2H=FT pe fiecare echipa, inputuri si probabilitati recalculate independent din CSV, 27 statusuri pre-meci si kickoff-uri in viitor la ora salvarii. Verificarea este a datelor salvate, nu monitorizare dupa analiza.'
$lines|Set-Content (Join-Path $PSScriptRoot 'ANALYSIS.md') -Encoding UTF8
[pscustomobject]@{recorded_at_ro=$timestamp;market='CORNERS_1H_UNDER_7_5';match_id=$selected.id;home=$selected.home;away=$selected.away;kickoff_ro='2026-11-02T19:00:00+02:00';raw_rank=3;
 p_raw=$selected.p_raw;p_stress=$selected.p_stress;fair_odds=$selected.fair_odds;historical_success=25;historical_n=25;calibrated=$false;market_price_verified=$false;status='prospective_analysis_only';result=''}|
 Export-Csv (Join-Path $PSScriptRoot 'prospective_selection.csv') -NoTypeInformation -Encoding UTF8
[pscustomobject]@{status='PASS';recorded_at_ro=$timestamp;historical_matches=575;candidate_matches=27;active_leagues=15;selected_id=$selected.id;
 p_raw=(Number $selected.p_raw);p_stress=(Number $selected.p_stress);p_season=(Cdf (Number $selected.lambda_season));p_venue=(Cdf (Number $selected.lambda_venue));calibrated=$false}|
 ConvertTo-Json|Tee-Object -FilePath (Join-Path $PSScriptRoot 'verification.json')
