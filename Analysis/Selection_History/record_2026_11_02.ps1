$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$analysis=Split-Path $root -Parent
$goals=@(Import-Csv (Join-Path $analysis 'Goal70_2026-11-02/Europe_Combined/comparison_with_first_goal.csv') -Encoding UTF8)
$corners=@(Import-Csv (Join-Path $analysis 'Corners1H_2026-11-02/all_models.csv') -Encoding UTF8)
$selectedGoal=@($goals|Where-Object id -eq '401883147')[0]
$selectedCorner=@($corners|Where-Object id -eq '5749738')[0]
if($goals.Count -ne 27 -or -not $selectedGoal -or -not $selectedCorner){throw 'Missing source observations'}
$recorded='2026-11-02T14:10:16+02:00'
$records=@(
 [pscustomobject]@{selection_id='2026-11-02_GOAL_BY_70_401883147';analysis_date='2026-11-02';selected_at_ro=$recorded;selected_by='user';
   market='GOAL_BY_70';event_definition='At least one valid goal by either team at displayed minute <=70';home=$selectedGoal.home;away=$selectedGoal.away;
   kickoff_ro='2026-11-02T22:00:00+02:00';match_id='401883147';provider='ESPN';raw_rank=$selectedGoal.rank;model_snapshot_at_ro='2026-11-02T11:36:58+02:00';
   p_raw=$selectedGoal.p_raw;p_stress=$selectedGoal.p_stress;fair_odds=$selectedGoal.fair_odds;calibrated=$false;market_price='';stake='';
   actual_first_goal_minute='';actual_corners_1h='';result='';result_source='';result_verified_at_ro='';net_profit='';
   status='user_intended_selection_result_pending';analysis_path='Analysis/Goal70_2026-11-02/MALLORCA_CADIZ.md';source=$selectedGoal.source},
 [pscustomobject]@{selection_id='2026-11-02_CORNERS_1H_UNDER_7_5_5749738';analysis_date='2026-11-02';selected_at_ro=$recorded;selected_by='user';
   market='CORNERS_1H_UNDER_7_5';event_definition='Both teams combined first-half corners <=7, including added time';home=$selectedCorner.home;away=$selectedCorner.away;
   kickoff_ro='2026-11-02T19:30:00+02:00';match_id='5749738';provider='FotMob';raw_rank='2';model_snapshot_at_ro='2026-11-02T13:31:35+02:00';
   p_raw=$selectedCorner.p_raw;p_stress=$selectedCorner.p_stress;fair_odds=$selectedCorner.fair_odds;calibrated=$false;market_price='';stake='';
   actual_first_goal_minute='';actual_corners_1h='';result='';result_source='';result_verified_at_ro='';net_profit='';
   status='user_intended_selection_result_pending';analysis_path='Analysis/Corners1H_2026-11-02/ALTERNATIVE_SASSUOLO.md';source=$selectedCorner.source}
)
$journal=Join-Path $root 'user_selections.csv'
if(Test-Path -LiteralPath $journal){
 $old=@(Import-Csv -LiteralPath $journal -Encoding UTF8)
 $new=@($records|Where-Object {$_.selection_id -notin $old.selection_id})
 if($new.Count){$new|Export-Csv -LiteralPath $journal -NoTypeInformation -Encoding UTF8 -Append}
}else{$records|Export-Csv -LiteralPath $journal -NoTypeInformation -Encoding UTF8}
$snapshot=Join-Path $root '2026-11-02_goal70_candidates.csv'
if(-not (Test-Path -LiteralPath $snapshot)){
 $goals|ForEach-Object {
  $r=$_|Select-Object *
  $r|Add-Member -NotePropertyName analysis_date -NotePropertyValue '2026-11-02'
  $r|Add-Member -NotePropertyName model_snapshot_at_ro -NotePropertyValue '2026-11-02T11:36:58+02:00'
  $r|Add-Member -NotePropertyName archived_at_ro -NotePropertyValue $recorded
  $r|Add-Member -NotePropertyName user_selected -NotePropertyValue ($r.id -eq '401883147')
  $r|Add-Member -NotePropertyName analyst_finalist -NotePropertyValue ($r.id -in @('401874963','401874494'))
  $r|Add-Member -NotePropertyName actual_first_goal_minute -NotePropertyValue ''
  $r|Add-Member -NotePropertyName result -NotePropertyValue ''
  $r|Add-Member -NotePropertyName result_source -NotePropertyValue ''
  $r|Add-Member -NotePropertyName result_verified_at_ro -NotePropertyValue ''
  $r
 }|Export-Csv -LiteralPath $snapshot -NoTypeInformation -Encoding UTF8
}
$culture=[Globalization.CultureInfo]::InvariantCulture
function Pct($v){(100*[double]::Parse([string]$v,$culture)).ToString('F2',$culture).Replace('.',',')+'%'}
$lines=New-Object System.Collections.Generic.List[string]
function Add([string]$s=''){$lines.Add($s)}
Add '# Istoric - 2 noiembrie 2026'
Add
Add ('Inregistrat la '+$recorded+', Europe/Bucharest. Rezultate in asteptare. Orele si probabilitatile de mai jos provin din analizele salvate, nu dintr-o noua verificare live sau din cote.')
Add
Add '## Alegerile declarate de utilizator'
Add
Add '| Partida | Ora RO | Eveniment | P model brut | Stres | Rezultat |'
Add '|---|---|---|---:|---:|---|'
Add ('| Mallorca - Cadiz | 22:00 | Minimum un gol pana la 70 inclusiv, oricine marcheaza | '+(Pct $selectedGoal.p_raw)+' | '+(Pct $selectedGoal.p_stress)+' | In asteptare |')
Add ('| Sassuolo - Fiorentina | 19:30 | Sub 7,5 cornere cumulate prima repriza, inclusiv 45+ | '+(Pct $selectedCorner.p_raw)+' | '+(Pct $selectedCorner.p_stress)+' | In asteptare |')
Add
Add 'Utilizatorul a declarat ca le alege astazi. Nu exista confirmare ca pariurile au fost plasate; mizele, preturile disponibile si profitul raman necompletate. Aceste doua alegeri reprezinta instructiunea explicita a zilei, nu schimbarea automata a preferintei generale de o selectie pe zi. Nu presupun un bilet combinat.'
Add
Add '## Selectia utilizatorului si recomandarile analistului'
Add
Add 'Pentru gol, prima recomandare a analistului dupa comparatia completa a fost Sassuolo-Fiorentina; a doua analiza a fost Sonderjyske-Copenhaga, cu rezerva privind diferenta dintre Poisson si istoricul primului gol. Benfica-Alverca a ramas alternativa a treia. Mallorca-Cadiz, ales de utilizator, ocupa locul 21 al formulei, cu profil mai favorabil numai pe teren (87,13%) decat pe sezon. Nu il relabelam retrospectiv drept prima recomandare.'
Add
Add 'Pentru cornere, prima selectie a analistului a fost Radnicki Nis-Zeleznicar Pancevo. La solicitarea unei alternative, a fost ales Sassuolo-Fiorentina, confirmat apoi de utilizator. Clasamentul modelului, judecata analistului si alegerea utilizatorului sunt pastrate separat.'
Add
Add '## Lista completa pentru gol pana la 70 inclusiv'
Add
Add 'Toate cele 27 de partide analizate, ordonate dupa modelul fix. Procente brute necalibrate, nu rate reale demonstrate. Cota echitabila din CSV este matematica, nu pret disponibil. Rezultatele raman necompletate, inclusiv pentru partidele nealese.'
Add
Add '| Loc model | Partida | Ora RO | P brut | Stres -20% | Rol in discutie |'
Add '|---:|---|---|---:|---:|---|'
foreach($g in $goals){
 $role=if($g.id -eq '401883147'){'ALES DE UTILIZATOR'}elseif($g.id -eq '401874963'){'Prima recomandare analist'}elseif($g.id -eq '401874494'){'A doua analiza, cu rezerve'}elseif($g.id -eq '401885403'){'A treia alternativa'}elseif($g.id -eq '401876419'){'Alegere initiala; loc 8 dupa extindere'}else{'Analizat'}
 Add ('| '+$g.rank+' | '+$g.home+' - '+$g.away+' | '+$g.kickoff_ro+' | '+(Pct $g.p_raw)+' | '+(Pct $g.p_stress)+' | '+$role+' |')
}
Add
Add 'U Cluj-FCSB: ora 20:30 este cea din furnizor/SuperLiga; pagina clubului afisa 21:00. Conflictul este pastrat in raportul original. Tenerife-Andorra este partida reprogramata la 15:00 RO, nu o selectie eliminata automat ca amanata.'
Add
Add '## Dovezi si reluare'
Add
Add '- [Comparatia completa Gol70](../Goal70_2026-11-02/EUROPE_ALL_MATCHES.md).'
Add '- [Analiza Mallorca-Cadiz](../Goal70_2026-11-02/MALLORCA_CADIZ.md).'
Add '- [Comparatia completa cornere 1H](../Corners1H_2026-11-02/ANALYSIS.md).'
Add '- [Alternativa Sassuolo-Fiorentina](../Corners1H_2026-11-02/ALTERNATIVE_SASSUOLO.md).'
Add '- [Jurnalul alegerilor utilizatorului](user_selections.csv).'
Add '- [Instantaneul celor 27 de candidate Gol70](2026-11-02_goal70_candidates.csv).'
Add
Add 'La revenire, se citesc AGENTS.md, promptul master si acest istoric. Se verifica ulterior rezultatul exact si sursa, fara a rescrie estimarile initiale. Nu exista actualizare automata sau programare creata prin aceasta salvare.'
$report=Join-Path $root '2026-11-02.md'
if(-not (Test-Path -LiteralPath $report)){$lines|Set-Content -LiteralPath $report -Encoding UTF8}
$saved=@(Import-Csv -LiteralPath $journal -Encoding UTF8|Where-Object analysis_date -eq '2026-11-02')
$savedCandidates=@(Import-Csv -LiteralPath $snapshot -Encoding UTF8)
if($saved.Count -ne 2 -or $savedCandidates.Count -ne 27 -or @($savedCandidates|Where-Object user_selected -eq 'True').Count -ne 1){throw 'Journal validation failed'}
[pscustomobject]@{status='saved_and_verified';recorded_at_ro=$recorded;user_selections=$saved.Count;goal_candidates=$savedCandidates.Count;results='existing_results_preserved';report=$report}|ConvertTo-Json
