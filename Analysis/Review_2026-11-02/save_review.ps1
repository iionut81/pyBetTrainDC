$ErrorActionPreference='Stop'
$root=$PSScriptRoot;$analysis=Split-Path $root -Parent
$historyRoot=Join-Path $analysis 'Selection_History'
$rows=@(Import-Csv (Join-Path $root 'goal70_outcomes.csv') -Encoding UTF8)
$metrics=Get-Content (Join-Path $root 'metrics.json') -Raw -Encoding UTF8|ConvertFrom-Json
$culture=[Globalization.CultureInfo]::InvariantCulture
function Num($v){[double]::Parse([string]$v,$culture)}
function Pct($v){(100*[double]$v).ToString('F2',$culture).Replace('.',',')+'%'}
function Check($b,[string]$m){if(-not $b){throw $m}}
$now=[DateTimeOffset]::UtcNow
$ro=[TimeZoneInfo]::ConvertTime($now,[TimeZoneInfo]::FindSystemTimeZoneById('GTB Standard Time')).ToString('yyyy-MM-ddTHH:mm:sszzz')
$c=Get-Content (Join-Path $root 'FotMob/5749738.json') -Raw -Encoding UTF8|ConvertFrom-Json
$cornerData=foreach($period in @('FirstHalf','SecondHalf','All')){
 $s=@($c.stats|Where-Object {$_.period -eq $period -and $_.key -eq 'corners'}|Select-Object -First 1)
 Check ($s.Count -eq 1 -and $null -ne $s[0].values[0] -and $null -ne $s[0].values[1]) ('Missing corner period '+$period)
 [pscustomobject]@{period=$period;home=[int]$s[0].values[0];away=[int]$s[0].values[1];total=[int]$s[0].values[0]+[int]$s[0].values[1]}
}
Check ($c.general.finished -eq $true -and $cornerData[0].home+$cornerData[1].home -eq $cornerData[2].home -and $cornerData[0].away+$cornerData[1].away -eq $cornerData[2].away) 'Corner reconciliation failed'
Check ($cornerData[0].total -eq 2) 'Unexpected Sassuolo corner result'
$cornerData|Export-Csv (Join-Path $root 'sassuolo_corner_result.csv') -NoTypeInformation -Encoding UTF8
$played=@($rows|Where-Object result -in @('success','fail'))
$methods=foreach($field in @('p_raw','season_only','venue_only','historical_frequency')){
 $brier=0;$ll=0
 foreach($r in $played){$p=Num $r.$field;$y=[int]$r.label;$brier += [math]::Pow($p-$y,2);$clamp=[math]::Max(1e-6,[math]::Min(1-1e-6,$p));$ll -= $y*[math]::Log($clamp)+(1-$y)*[math]::Log(1-$clamp)}
 [pscustomobject]@{method=$field;n=$played.Count;brier=$brier/$played.Count;log_loss=$ll/$played.Count;fitted_on_review=$false;logloss_clamp_epsilon=1e-6;interpretation='single-day descriptive comparison; not independent calibration'}
}
$methods|Export-Csv (Join-Path $root 'pre_existing_methods_comparison.csv') -NoTypeInformation -Encoding UTF8
$journalPath=Join-Path $historyRoot 'user_selections.csv'
$candidatesPath=Join-Path $historyRoot '2026-11-02_goal70_candidates.csv'
foreach($p in @($journalPath,$candidatesPath)){
 $backup=Join-Path $root (([IO.Path]::GetFileNameWithoutExtension($p))+'_before_review.csv')
 if(-not (Test-Path -LiteralPath $backup)){Copy-Item -LiteralPath $p -Destination $backup}
}
$candidates=@(Import-Csv $candidatesPath -Encoding UTF8)
$frozenFields=@($candidates[0].PSObject.Properties.Name|Where-Object {$_ -notin @('actual_first_goal_minute','result','result_source','result_verified_at_ro','actual_score_ft','result_reason')})
$before=$candidates|Select-Object $frozenFields|ConvertTo-Json -Depth 3 -Compress
foreach($p in $candidates){
 $r=@($rows|Where-Object prediction_id -eq $p.id)
 Check ($r.Count -eq 1) ('Missing outcome for saved candidate '+$p.id)
 $p.actual_first_goal_minute=$r[0].first_goal_minute;$p.result=$r[0].result;$p.result_source=$r[0].result_source;$p.result_verified_at_ro=$ro
 $p|Add-Member -NotePropertyName actual_score_ft -NotePropertyValue $r[0].score_ft -Force
 $p|Add-Member -NotePropertyName result_reason -NotePropertyValue $r[0].reason -Force
 if($p.result -eq 'postponed_not_scored'){$p.result_source='https://www.superliga.rs/en/odlozen-mec-14-kola-radnicki-nis-zeleznicar-novi-termin-25-novembar/'}
}
$candidates|Export-Csv $candidatesPath -NoTypeInformation -Encoding UTF8
$after=@(Import-Csv $candidatesPath -Encoding UTF8)|Select-Object $frozenFields|ConvertTo-Json -Depth 3 -Compress
Check ($before -ceq $after) 'Prediction fields changed while recording outcomes'
$journal=@(Import-Csv $journalPath -Encoding UTF8)
$journalFrozen=@('selection_id','selected_at_ro','home','away','market','kickoff_ro','match_id','raw_rank','p_raw','p_stress','fair_odds','model_snapshot_at_ro','market_price','stake','net_profit')
$journalBefore=$journal|Select-Object $journalFrozen|ConvertTo-Json -Compress
foreach($s in $journal){
 if($s.analysis_date -ne '2026-11-02'){continue}
 if($s.market -eq 'GOAL_BY_70' -and $s.match_id -eq '401883147'){
  $r=@($rows|Where-Object prediction_id -eq $s.match_id)[0]
  $s.actual_first_goal_minute=$r.first_goal_minute;$s.result=$r.result
  $s.result_source=$r.result_source+' | https://www.rcdmallorca.es/noticias/el-mallorca-culmina-la-remontada-ante-el-cadiz-2-1'
 }elseif($s.market -eq 'CORNERS_1H_UNDER_7_5' -and $s.match_id -eq '5749738'){
  $s.actual_corners_1h=[string]$cornerData[0].total;$s.result=if($cornerData[0].total -le 7){'success'}else{'fail'};$s.result_source=$c.source
 }else{throw 'Unexpected user selection'}
 $s.result_verified_at_ro=$ro;$s.status='user_selection_result_verified'
}
$journal|Export-Csv $journalPath -NoTypeInformation -Encoding UTF8
$journalAfter=@(Import-Csv $journalPath -Encoding UTF8)|Select-Object $journalFrozen|ConvertTo-Json -Compress
Check ($journalBefore -ceq $journalAfter) 'User prediction/stake fields changed'
$lines=New-Object System.Collections.Generic.List[string]
function Add([string]$s=''){$lines.Add($s)}
Add '# Retrospectiva Gol70 - meciurile din 2 noiembrie 2026'
Add
Add ('Rezultate verificate la '+$ro+'. Datele predictive sunt cele salvate INAINTE de meciuri, la 2 noiembrie 11:36:58 RO; nu sunt recalculate folosind rezultatele zilei.')
Add
Add '**Bilant:** 27 partide inventariate, 26 jucate, 22 succese, 4 esecuri, 1 amanare exclusa din numitor. Hit rate descriptiv **84,62%**. Top 1: 1/1; top 3: 3/3; top 5: 5/5; top 10: 9/10. Lista de 27 este un inventar clasificat, nu 27 recomandari de pariat. Finalistii analistului si alegerile utilizatorului raman distincte.'
Add
Add '## Alegerile utilizatorului'
Add
Add '| Alegere | Estimarea salvata | Observatia verificata | Rezultat |'
Add '|---|---:|---|---|'
Add '| Mallorca-Cadiz Gol70 | 82,66% brut; stres 75,39% | 2-1; primul gol Cadiz la 50, gol Mallorca la 63 | SUCCES |'
Add '| Sassuolo-Fiorentina sub 7,5 cornere 1H | 96,84% brut; stres 92,59% | Cornere la pauza 0-2, total 2; FT 5-3, total 8 | SUCCES |'
Add
Add '[Raportul oficial Mallorca](https://www.rcdmallorca.es/noticias/el-mallorca-culmina-la-remontada-ante-el-cadiz-2-1) confirma primul gol la 50. Modulul statistic al aceleiasi pagini afiseaza 49, dar textul cronologic si ambii furnizori folosesc 50; diferenta nu afecteaza pragul 70. Pentru cornere se foloseste FirstHalf explicit din [FotMob matchDetails](https://www.fotmob.com/api/data/matchDetails?matchId=5749738), nu totalul FT. Suma 1H+2H este verificata pe fiecare echipa. Mizele si preturile nu sunt cunoscute, deci nu se calculeaza profit sau ROI.'
Add
Add '## Toata lista, in ordinea salvata'
Add
Add '| Loc | Partida | P brut initial | Scor FT | Primul gol | Evaluare Gol70 |'
Add '|---:|---|---:|---:|---:|---|'
foreach($r in ($rows|Sort-Object {[int]$_.rank})){
 $event=if($r.result -eq 'postponed_not_scored'){'AMANAT - exclus'}elseif($r.result -eq 'success'){'SUCCES'}else{'ESEC'}
 $first=if($r.first_goal_minute){$r.first_goal_minute+"'"}elseif($r.result -eq 'postponed_not_scored'){'nu s-a jucat'}else{'fara gol'}
 $score=if($r.score_ft){$r.score_ft}else{'-'}
 Add ('| '+$r.rank+' | ['+$r.home+' - '+$r.away+']('+$r.result_source+') | '+(Pct (Num $r.p_raw))+' | '+$score+' | '+$first+' | '+$event+' |')
}
Add
Add 'Radnicki-Zeleznicar a fost amanat din cauza unei defectiuni electrice; liga a anuntat reprogramarea pe 25 noiembrie. Nu reprezinta 0-0 jucat, succes, esec sau presupusa rambursare a vreunui pariu. O analiza pentru 25 noiembrie necesita un nou instantaneu de date. [Anuntul oficial](https://www.superliga.rs/en/odlozen-mec-14-kola-radnicki-nis-zeleznicar-novi-termin-25-novembar/).'
Add
Add '## Ce au sustinut corect datele la reusite'
Add
Add 'Sassuolo-Fiorentina: primul gol la 25, compatibil cu cele 18/18 partide distincte anterioare cu gol <=70 si cu topul formulei. Nu demonstreaza ca 94,40% a fost bine calibrat; sensibilitatea la 2-6 cu Lazio si eliminarea din acel meci ramane o limita. Sonderjyske-Copenhaga: primul gol la 20; rezultatul corect nu elimina discrepanta observata ieri intre 94,36% brut si 86,96% frecventa sezonului. Benfica-Alverca: primul gol la 51; istoricul 18/18 era compatibil, fara sa garanteze un gol in prima repriza.'
Add
Add 'Mallorca-Cadiz: configuratia terenului avea 9/9 succese si p numai-teren 87,13%, fata de 82,66% combinat. Cadiz a marcat la 50, iar Mallorca la 63. Cerinta oricine marcheaza a fost corecta. Raportul oficial descrie dominare Mallorca si ocazii ratate in prima repriza, apoi gol Cadiz la tranzitie. Datele post-meci FotMob arata xG 1H 2,06-0,10: au existat ocazii, desi pauza era 0-0. Acest xG nu era disponibil inainte de meci si nu este reutilizat ca predictor pentru ieri. [Cronica oficiala](https://www.rcdmallorca.es/noticias/el-mallorca-culmina-la-remontada-ante-el-cadiz-2-1). Faptul ca a castigat nu il muta retrospectiv de pe locul 21 pe locul 1.'
Add
Add 'Atalanta-Parma: a castigat evenimentul cu golul Parmei la 41, desi Parma nu marcase <=70 in cele patru deplasari anterioare. Selectia oricare echipa a fost corecta, dar rezultatul nu confirma o poveste despre superioritatea ofensiva a Atalantei. Lorient-Brest: gol la 69, succes aproape de limita, nu motiv sa ignoram fragilitatea primei reprize (xG 0,41-0,12, un sut pe poarta cumulat). [Cronologia Lorient-Brest](https://www.ouest-france.fr/sport/football/resultats-matchs/match-brest-lorient-1831-1299/).'
Add
Add '## Cele patru esecuri si lectiile lor'
Add
Add '### U Cluj-FCSB: 89,76% brut, primul gol la 76'
Add
Add 'Este cel mai important esec pentru ordinea listei, fiind pe locul 7. Riscul era vizibil inainte: FCSB marcase doar 3 goluri <=70 in 6 deplasari (0,50/meci) si doua dintre acele deplasari nu avusesera gol pana la 70. Frecventa pe teren, combinata fara duplicate, era 11/13 = 84,62%, sub p brut. Nu era un meci fara suturi: in prima repriza au fost 3-8 suturi si 0-3 pe poarta, conform sursei numerice. Nu avem xG verificabil pentru a numi toate acele suturi ocazii mari. Golul decisiv a venit dintr-o faza fixa la 76. Imbunatatire de proces: arata explicit contributia ofensiva mica in deplasare si discrepanta p/frecventa, fara sa scazi un procent ales arbitrar. [Cronica oficiala SuperLiga](https://www.superliga.ro/meci/superliga-26-27-regular-ucj-fcsb?sectiune=cronica).'
Add
Add '### Mjallby-Kalmar: 84,06%, 0-0'
Add
Add 'Pe configuratia terenului, echipele marcasera numai 7/13 si 8/13 goluri <=70: 0,54 si 0,62/meci. Apararile care primesc gol nu inlocuiesc o productie ofensiva proprie consistenta. Frecventa anterioara pe teren era 20/26 = 76,92%; p numai-teren era mai mic decat p combinat. Post-meci: xG total 0,71-0,44, 1,15 cumulat in 90 minute; raportul Kalmar descrie sanse ale ambelor echipe, fara gol. Imbunatatire: diagnosticul celor doua atacuri trebuie afisat separat de totalul GF+GA si comparat cu calitatea ocaziilor, cand exista date comparabile dinainte de meci. Nu se elimina o liga dupa un singur 0-0. [Raport oficial Kalmar](https://kalmarff.se/oavgjort-pa-strandvallen/).'
Add
Add '### Mladost-Radnik: 80,43%, gol la 84'
Add
Add 'Semnalele anterioare nu sustineau o selectie de top: Mladost primise doar un gol <=70 in sase partide acasa; Radnik marcase trei in sase deplasari, iar gazda cinci in sase. Frecventa pe teren era 9/12 = 75%, stresul 72,89%. Prima repriza a avut 14 suturi, dar un singur sut pe poarta, fara ocazii mari in datele furnizorului; multe suturi nu implica gol devreme. Imbunatatire: diferentiaza volumul de eficienta si nu transforma meciurile de 80% intr-o lista de recomandari cu sanse foarte ridicate. [Relatarea radioteleviziunii sarbe, golul la 84](https://www.rts.rs/sport/fudbal/superliga_srbije/6074912/radnik-mladost-superliga-srbije-2026.html).'
Add
Add '### Gil Vicente-Vitoria: 75,06%, 0-0'
Add
Add 'Era ultimul in formula, nu recomandare principala. Istoricul anterioar al primului gol era 13/18 = 72,22%, iar gazdele marcasera numai 3 goluri <=70 in 5 partide acasa. Post-meci: la pauza xG 0,19-0,05, fara sut pe poarta; totalul xG la final 1,07. Profilul slab a ramas slab, nu avem dovada unei erori mari a modelului doar pentru ca a pierdut. Imbunatatire: delimiteaza inventarul de candidati de selectia finala si nu promova meciul doar pentru a gasi un pret dorit. [Confirmarea 0-0, RTP](https://www.rtp.pt/noticias/desporto/empate-sem-golos-no-jogo-entre-gil-vicente-e-vitoria-de-guimaraes_e1776397).'
Add
Add 'Statistica de prima repriza descrie primele 45+ minute. Statistica FT include si minutele de dupa 70; nu o prezint drept volum exact in intervalul 0-70. Datele post-meci ajuta explicatia rezultatului, nu sunt informatii care puteau fi folosite ieri.'
Add
Add '## Cum evaluam probabilitatile'
Add
Add 'Media probabilitatilor pentru cele 26 jucate era 86,18%: suma anticipata 22,41 succese si 3,59 esecuri. Au fost 22 si 4. Abaterea observata nu este o dovada ca modelul trebuie cosmetizat. In schimb, o singura zi nu poate demonstra calibrarea, rata reala de 90% sau profitabilitatea. Brier 0,1209 si log loss 0,3945 sunt scoruri descriptive ale acestei zile.'
Add
Add '| Varianta disponibila inainte de meci | Brier (mai mic e mai bun) | Log loss |'
Add '|---|---:|---:|'
foreach($m in $methods){Add ('| '+$m.method+' | '+([double]$m.brier).ToString('F4',$culture)+' | '+([double]$m.log_loss).ToString('F4',$culture)+' |')}
Add
Add 'Numai teren a fost usor mai bun azi (Brier 0,1185 fata de 0,1209), dar avantajul este mic si masurat pe 26 meciuri. Nu justific schimbarea ponderilor la 100% teren. Frecventa empirica este neajustata si poate fi 100% pe esantioane mici; log loss foloseste limitare numerica 1e-6, fara invatarea unui procent pe rezultatele zilei.'
Add
Add '## Imbunatatiri concrete, fara rescrierea modelului'
Add
Add '1. Pentru fiecare finalist, publica separat GF70/meci ale ambelor echipe pe teren, frecventa 0-0 la 70, deschiderile tarzii si diferenta dintre p brut si frecventa observata. Nu adopta un prag nou ales pe aceste patru esecuri.'
Add '2. Pastreaza un diagnostic de sensibilitate sezon/teren si la meciuri cu eliminari; ipotezele de verificat se aplica tuturor candidatilor, inclusiv celor care au castigat. Nu elimina retrospectiv doar observatii incomode.'
Add '3. Urmareste in paralel variantele deja definite (sezon, teren, 50/50, frecventa primului gol) pe zile urmatoare, fara a modifica procentele salvate. Alege ponderi numai prin evaluare cronologica pe un esantion separat.'
Add '4. Pentru un model viitor, testeaza direct timpul pana la PRIMUL gol / probabilitatea 0-0 la 70. Numarul tuturor golurilor <=70 include goluri dupa deschiderea scorului si nu este aceeasi tinta. Ramane ipoteza de cercetare, nu implementare validata sau modificare a productiei.'
Add '5. Marcheaza reusitele aproape de prag (Lorient 69) si amanarile separat. Nu confunda un succes cu validarea tuturor argumentelor tactice si nu masura profit fara pret si miza reale.'
Add
Add '## Integritate si fisiere'
Add
Add '26 scoruri finale reconciliate cu listele golurilor pe echipe. Pentru 19 partide, ESPN si FotMob confirma scorul si incadrarea <=70; celelalte sapte folosesc FotMob, cu surse suplimentare pentru esecurile importante si amanare. Nicio cota nu a fost folosita. Probabilitatile, rangurile, orele de selectie si alegerile originale nu au fost schimbate. Fisierele dinaintea completarii sunt arhivate aici cu sufixul before_review.csv.'
Add
Add 'Jurnalul central si lista candidatilor au fost completate doar cu rezultatele si sursele. Raportul zilei initiale ramane disponibil ca instantaneu pre-meci. Reproducere offline: evaluate_results.ps1 si save_review.ps1, folosind JSON-urile sportive selectate din FotMob/ si ESPN/. Sunt actualizari documentare; productia si motorul de selectie nu au fost modificate.'
$lines|Set-Content (Join-Path $root 'REVIEW.md') -Encoding UTF8
$index=Join-Path $historyRoot '2026-11-02.md'
$old=[IO.File]::ReadAllText($index,[Text.Encoding]::UTF8)
if(-not $old.Contains('## Rezultate verificate la 3 noiembrie')){
 $note="`r`n`r`n## Rezultate verificate la 3 noiembrie`r`n`r`nMallorca-Cadiz Gol70: SUCCES, primul gol la 50. Sassuolo-Fiorentina cornere 1H: SUCCES, total la pauza 2 (0-2). Lista completa: 22 succese / 26 jucate (84,62%), 4 esecuri si Radnicki-Zeleznicar amanat, exclus. Textul pre-meci de mai sus este pastrat ca instantaneu; statutul actual si sursele sunt in [retrospectiva](../Review_2026-11-02/REVIEW.md) si in CSV-uri. Procentele initiale nu au fost rescrise.`r`n"
 [IO.File]::WriteAllText($index,$old+$note,(New-Object Text.UTF8Encoding($true)))
}
$checks=[pscustomobject]@{status='PASS';verified_at_ro=$ro;predictions=27;played=26;success=22;fail=4;postponed=1;user_selections_verified=2;original_prediction_fields_preserved=$true;stake_and_price_unchanged=$true;sassuolo_corners_1h=$cornerData[0].total}
$checks|ConvertTo-Json|Set-Content (Join-Path $root 'journal_update_verification.json') -Encoding UTF8
$checks|ConvertTo-Json
