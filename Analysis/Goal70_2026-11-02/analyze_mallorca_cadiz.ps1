$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$dataDir=Join-Path $root 'Europe_Combined'
$history=@(Import-Csv (Join-Path $dataDir 'historical_validated.csv') -Encoding UTF8|Where-Object {
    $_.league -eq 'esp.2' -and ($_.home_id -in @('84','3842') -or $_.away_id -in @('84','3842'))
})
$model=@(Import-Csv (Join-Path $dataDir 'all_models.csv') -Encoding UTF8|Where-Object id -eq '401883147')[0]
$live=Get-Content (Join-Path $dataDir 'mallorca_cadiz_live_status.json') -Raw -Encoding UTF8|ConvertFrom-Json
if($history.Count -ne 22 -or @($history|Group-Object id|Where-Object Count -gt 1).Count -gt 0){throw 'Incomplete or duplicate season observations'}
foreach($r in $history){
    $hg=@($r.home_goals -split ';'|Where-Object {$_});$ag=@($r.away_goals -split ';'|Where-Object {$_})
    if($hg.Count -ne [int]$r.home_score -or $ag.Count -ne [int]$r.away_score){throw ('Score mismatch '+$r.id)}
    foreach($s in @(@($hg,$r.home70),@($ag,$r.away70))){
        $count=@($s[0]|Where-Object {if($_ -notmatch '^(\d+)'){throw 'Invalid minute'};[int]$Matches[1] -le 70}).Count
        if($count -ne [int]$s[1]){throw ('Goal70 mismatch '+$r.id)}
    }
}
function Stats([string]$id,[string]$venue='all'){
    $rows=@($history|Where-Object {($venue -eq 'all' -and ($_.home_id -eq $id -or $_.away_id -eq $id)) -or
        ($venue -eq 'home' -and $_.home_id -eq $id) -or ($venue -eq 'away' -and $_.away_id -eq $id)})
    $gf=0;$ga=0;$win=0
    foreach($r in $rows){if($r.home_id -eq $id){$gf += [int]$r.home70;$ga += [int]$r.away70}else{$gf += [int]$r.away70;$ga += [int]$r.home70};if($r.success70 -eq 'True'){$win++}}
    [pscustomobject]@{n=$rows.Count;gf70=$gf;ga70=$ga;success=$win}
}
$a=Stats '84';$b=Stats '3842';$av=Stats '84' 'home';$bv=Stats '3842' 'away'
$ls=($a.gf70/$a.n+$b.ga70/$b.n+$b.gf70/$b.n+$a.ga70/$a.n)/2
$lv=($av.gf70/$av.n+$bv.ga70/$bv.n+$bv.gf70/$bv.n+$av.ga70/$av.n)/2
$lc=($ls+$lv)/2;$p=1-[math]::Exp(-$lc)
if([math]::Abs($p-[double]::Parse($model.p_raw,[Globalization.CultureInfo]::InvariantCulture)) -gt 1e-12){throw 'Model mismatch'}
$venue=@($history|Where-Object {$_.home_id -eq '84' -or $_.away_id -eq '3842'})
if($venue.Count -ne 9 -or @($venue|Where-Object success70 -eq 'True').Count -ne 9){throw 'Venue frequency mismatch'}
if($live.state -ne 'pre' -or $live.time_valid -ne $true -or [DateTimeOffset]::Parse($live.date) -le [DateTimeOffset]::Parse($live.verified_at_utc)){throw 'Invalid saved pre-match status'}
$out=[pscustomobject]@{match_id=$model.id;market='GOAL_BY_70';kickoff_ro='2026-11-02T22:00:00+02:00';verified_at_utc=$live.verified_at_utc;
    mallorca_season=$a;cadiz_season=$b;mallorca_home=$av;cadiz_away=$bv;lambda_season=$ls;lambda_venue=$lv;lambda_combined=$lc;
    p_season=1-[math]::Exp(-$ls);p_venue=1-[math]::Exp(-$lv);p_raw=$p;p_stress=1-[math]::Exp(-.8*$lc);fair_odds=1/$p;
    historical_unique_n=22;historical_success=17;venue_unique_n=9;venue_success=9;calibrated=$false;market_price_verified=$false;
    checks='PASS: 22 unique fixtures, goal-score reconciliation, <=70 counts, unchanged model, 9/9 venue, saved pre-match status'}
$out|ConvertTo-Json -Depth 6|Set-Content (Join-Path $dataDir 'mallorca_cadiz_inputs.json') -Encoding UTF8
$history|Sort-Object date -Descending|Export-Csv (Join-Path $dataDir 'mallorca_cadiz_season.csv') -NoTypeInformation -Encoding UTF8
$cup=[pscustomobject]@{league='Copa del Rey';id='xerez-cadiz-20261029';date='2026-10-29T20:00Z';home='Xerez CD';away='Cadiz';home_id='xerez';away_id='3842';home_score=1;away_score=3;first_goal="2'";success70='True';home_goals="2'";away_goals="25';59';83'";source='https://www.cadizcf.com/noticias/el-cadiz-cf-remonta-en-chapin-y-avanza-en-la-copa-del-rey-1-3'}
$lines=New-Object System.Collections.Generic.List[string]
function Add([string]$s=''){$lines.Add($s)}
Add '# Mallorca-Cadiz: de ce este pe locul 21'
Add
Add '**2 noiembrie 2026, 22:00 RO / 21:00 CET, Son Moix, LaLiga Hypermotion, etapa 12.** Status pre-meci reverificat la 11:52:34 RO. Nu au fost folosite cote. Este un studiu al unui candidat, nu o noua selectie zilnica sau o schimbare a modelului.'
Add
Add '## Concluzia'
Add
Add '**82,66% brut, 87,13% numai pe teren, 75,39% in stres, cota echitabila 1,2097.** Locul 21 vine din formula fixa, nu dintr-un veto contextual. Are un semnal favorabil care merita aratat: Mallorca acasa 5/5 si Cadiz afara 4/4 cu gol <=70. Totusi, pe sezon sunt numai 17/22 (77,27%), iar ponderarea 50/50 trage rezultatul in jos. Nici 9/9, nici 82,66% nu reprezinta probabilitati reale validate.'
Add
Add '## Inputuri si calcul'
Add
Add '| Esantion | N | Goluri marcate <=70 | Goluri primite <=70 | Partide cu primul gol <=70 |'
Add '|---|---:|---:|---:|---:|'
Add '| Mallorca sezon | 11 | 9 | 3 | 8/11 |'
Add '| Cadiz sezon | 11 | 7 | 13 | 9/11 |'
Add '| Mallorca acasa | 5 | 7 | 1 | 5/5 |'
Add '| Cadiz deplasare | 4 | 2 | 8 | 4/4 |'
Add
Add '```text'
Add 'lambda_sezon = ((9/11 + 13/11) + (7/11 + 3/11))/2 = 1,454545'
Add 'P_sezon = 1 - exp(-1,454545) = 76,65%'
Add 'lambda_teren = ((7/5 + 8/4) + (2/4 + 1/5))/2 = 2,05'
Add 'P_teren = 1 - exp(-2,05) = 87,13%'
Add 'lambda_combinat = (1,454545 + 2,05)/2 = 1,752273'
Add 'P_model = 1 - exp(-1,752273) = 82,66%'
Add 'P_stres = 1 - exp(-0,8 * 1,752273) = 75,39%'
Add 'Cota echitabila = 1 / 0,82662055 = 1,2097'
Add '```'
Add
Add 'Ponderarea 50/50 este euristica. Esantioanele sezon/teren se suprapun. Diferenta 76,65-87,13% este sensibilitate la alegerea esantionului, nu interval de incredere. Nu schimb ponderile pentru a obtine peste 85%. Modelul nu are ajustare dupa adversari sau schimbarea antrenorului, iar golurile ulterioare deschiderii scorului pot distorsiona interpretarea pentru PRIMUL gol.'
Add
Add '## Toate cele 9 observatii relevante pentru teren'
Add
Add '| Data | Meci si scor final | Primul gol |'
Add '|---|---|---:|'
foreach($r in ($venue|Sort-Object date)) {Add ('| '+$r.date.Substring(0,10)+' | ['+$r.home+' - '+$r.away+' '+$r.home_score+'-'+$r.away_score+']('+$r.source+') | '+$r.first_goal+' |')}
Add
Add 'Toate au primul gol pana la 62 inclusiv. Mallorca a marcat pana la 70 in 4/5 acasa; al cincilea succes a fost golul adversarului Almeria. Cadiz a primit gol pana la 70 in toate cele 4 deplasari. Faptul ca Mallorca este defensiv solida poate proteja scorul mic, dar reduce probabilitatea ca oaspetii sa salveze selectia daca gazdele nu marcheaza.'
Add
Add '## Ultimele 10 meciuri oficiale ale fiecarei echipe'
Add
Add 'Numai meciuri jucate, nu amicalele; la Cadiz includ cupa din 29 octombrie separat de formula de campionat. Mallorca-Alcoyano nu s-a jucat la data initiala; nu inventez un rezultat de cupa.'
foreach($id in @('84','3842')){
    Add
    Add $(if($id -eq '84'){'### Mallorca'}else{'### Cadiz'})
    Add
    Add '| Data | Competitie | Meci si scor final | Primul gol | Gol <=70 |'
    Add '|---|---|---|---:|---|'
    $rows=@($history|Where-Object {$_.home_id -eq $id -or $_.away_id -eq $id})
    if($id -eq '3842'){$rows+=@($cup)}
    foreach($r in ($rows|Sort-Object date -Descending|Select-Object -First 10)){
        $comp=if($r.id -eq $cup.id){'Copa del Rey'}else{'LaLiga Hypermotion'}
        $fg=if($r.first_goal){$r.first_goal}else{'fara gol (0-0)'}
        Add ('| '+$r.date.Substring(0,10)+' | '+$comp+' | ['+$r.home+' - '+$r.away+' '+$r.home_score+'-'+$r.away_score+']('+$r.source+') | '+$fg+' | '+$(if($r.success70 -eq 'True'){'DA'}else{'NU'})+' |')
    }
}
Add
Add 'Esecurile din intreg sezonul: Eldense-Mallorca 0-0, Real Sociedad II-Mallorca 0-1 (primul gol 81), Girona-Mallorca 0-0; Cadiz-Celta Fortuna 0-0 si Cadiz-Sporting 0-1 (87). Niciunul nu apartine configuratiei de teren de azi. Ultimele 10 oficiale: Mallorca 7/10; Cadiz 9/10, cu adversar inferior in cupa. Nu tratez victoria din cupa ca echivalenta cu o deplasare la Mallorca.'
Add
Add '## Context, antrenori si loturi'
Add
Add 'Programul oficial Mallorca confirma etapa 12, luni la 21:00 CET. Sunt puncte in campionat, fara scor agregat sau prelungiri. Obiectivul sezonului este promovarea pentru Mallorca; o victorie ar duce-o de la 19 la 22 puncte, egalul la 20. Clasamentul din match center-ul oficial Cadiz arata Mallorca 8 cu 19 puncte si Cadiz 19 cu 10, dupa 11 meciuri fiecare. Cadiz se afla in zona retrogradarii; un punct in deplasare la o candidata la promovare poate fi util. Aceasta ultima interpretare este analiza, nu o declaratie ca echipa joaca pentru 0-0. [Program Mallorca](https://www.rcdmallorca.es/noticias/entradas-a-la-venta-para-el-partido-contra-el-cadiz-cf), [clasament oficial pre-meci](https://www.cadizcf.com/partidos/temporada-2026-2027-laliga-hypermotion-12-rcd-mallorca-vs-cadiz-cf-3428).'
Add
Add 'Luis Garcia cere mai multa regularitate si agresivitate, fara ca asta sa insemne un ritm ofensiv nesustenabil. Raportul clubului de la Leganes descrie constructie prin posesie, Torre intre linii, Fuentes cu spatele la poarta, Luvumbo pe spatii si la recuperarea mingilor respinse. Prima repriza a avut putine ocazii, desi primul gol a fost la 34. Interpretarea pentru azi: Mallorca poate controla teritorial fara sa genereze foarte multe ocazii clare; posesia singura nu produce garantia golului. [Raport oficial Leganes-Mallorca](https://www.rcdmallorca.es/noticias/empate-1-1-ante-el-cd-leganes).'
Add
Add 'Configuratie tactica de lucru: Mallorca 4-2-3-1 contra Cadiz 4-4-2, asa cum apar in pagina AS, tratate numai ca probabile. Mallorca poate cauta superioritate intre liniile de patru, schimbarile de directie si duelurile extremei cu fundasul lateral. Cadiz poate inchide centrul si cauta Izeta/Garcia Pascual la tranzitie. Daca blocul oaspetilor rezista, poate exista dominare fara gol; daca gazdele pierd mingea cu lateralii sus, contraatacul ofera a doua cale spre gol. Acestea sunt scenarii, nu bonusuri procentuale. Nu copiez integral un 11 precompletat de agregator: banca include si David Lopez, pe care antrenorul il declara indisponibil. [AS, module pre-meci](https://as.com/resultados/futbol/segunda/2026_2027/directo/regular_a_12_6a44993ebedbb16/alineaciones/).'
Add
Add 'Borja Jimenez este la al TREILEA meci pe banca Cadiz: victoriile 3-1 cu Ceuta si 3-1 la Xerez. Declaratia oficiala pune accent pe competitivitate, pastrarea sanselor in meci si putinele antrenamente avute. Este o schimbare de regim: cea mai mare parte din cele 11 observatii de campionat este de dinaintea sa. Nu avem suficiente meciuri sub noul antrenor pentru un model separat; nici doua victorii nu autorizeaza o crestere automata a probabilitatii. [Avancronica oficiala Borja](https://www.cadizcf.com/noticias/borja-jimenez-debemos-estar-vivos-en-los-partidos).'
Add
Add 'Loturi Mallorca: John Chetauya si David Lopez nu sunt disponibili; Arnau Puigmal si Josep Cerda sunt pregatiti, conform conferintei din 31 octombrie. Nu preiau liste vechi de indisponibili de la alte avancronici. [Conferinta oficiala Luis Garcia](https://www.rcdmallorca.es/noticias/luis-garcia-tenemos-que-encontrar-esa-regularidad-mirarnos-a-nosotros-mismos-y-mejorar).'
Add
Add 'Cadiz: Jandro este explicit indisponibil luni (aductor); Vladys/Kopotun are leziune miofasciala, disponibilitate dependenta de evolutie; Petit era asteptat sa revina la grup saptamana urmatoare. Ontiveros a lipsit din cupa din motive familiale, nu din cauza unei accidentari comunicate. Buletinul nu este un 11 confirmat si nu presupun automat revenirea pe teren a vreunuia. Match center-ul oficial precizeaza ca formatiile vor aparea cu o ora inainte de start. [Buletin medical oficial](https://www.cadizcf.com/noticias/parte-medico-jandro-y-vladys).'
Add
Add '## Recuperare, deplasare, vreme'
Add
Add 'Mallorca nu a jucat in cupa la Alcoy; antrenorul confirma ca echipa a calatorit, a revenit si s-a pregatit pentru luni. Ultimul meci efectiv a fost pe 24 octombrie: 9 zile calendaristice pana azi. Cadiz a jucat la Xerez JOI 29, nu miercuri 28, dupa amanarea cu 24 ore: 4 zile pana luni. Cadiz are deplasare la Palma; nu am verificat ora zborului si nu ii atribui un efect numeric. Mallorca are avantaj de odihna, dar golul timpuriu nu rezulta automat din aceasta. Surse: conferinta Luis Garcia si [raport oficial cupa Cadiz](https://www.cadizcf.com/noticias/el-cadiz-cf-remonta-en-chapin-y-avanza-en-la-copa-del-rey-1-3).'
Add
Add 'Nu am o prognoza orara verificata sau un raport actual despre gazon. Nu afirm ca vremea este perfecta si nu modific estimarea pe baza unei presupuneri meteorologice.'
Add
Add '## Decizie explicata'
Add
Add 'Cel mai puternic argument PRO: toate cele 4 deplasari Cadiz au inclus gol primit <=70, iar toate cele 5 meciuri Mallorca acasa au inclus primul gol <=70. Principalul risc: un singur producator ofensiv credibil intr-un meci controlat de Mallorca, cu Cadiz compact si un antrenor nou. Palmaresul defensiv Mallorca (numai 3 goluri primite <=70 in 11 meciuri, unul acasa) reduce contributia probabila a oaspetilor.'
Add
Add 'Locul 21 este corect pentru formula fixata, dar nu demonstreaza ca are a 21-a probabilitate reala. Aici formula este sensibila la includerea deplasarilor slab productive ale Mallorca si a meciurilor Cadiz acasa, care nu sunt configuratia de azi. Este o limita de evaluat ulterior pe un backtest, nu o justificare sa schimbam astazi ponderile. Il pastrez ca alternativa interesanta pe teren, fara a-l promova peste finalisti si fara a-l eticheta 90%.'
Add
Add 'Date si reproducere: Europe_Combined/mallorca_cadiz_inputs.json, mallorca_cadiz_season.csv, mallorca_cadiz_live_status.json; powershell -NoProfile -File Analysis/Goal70_2026-11-02/analyze_mallorca_cadiz.ps1. Calculul a fost refacut si verificat fata de modelul din comparatia de 27 de meciuri. Nu a fost modificat topul, jurnalul selectiilor, modelul de productie sau vreun pariu.'
$lines|Set-Content (Join-Path $root 'MALLORCA_CADIZ.md') -Encoding UTF8
$out|ConvertTo-Json -Depth 5
