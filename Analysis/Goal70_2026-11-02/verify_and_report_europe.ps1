param([string]$RecordedAtUtc = '2026-11-02T09:34:42Z')
$ErrorActionPreference = 'Stop'
$dir = Join-Path $PSScriptRoot 'Europe_Combined'
$models = @(Import-Csv (Join-Path $dir 'all_models.csv') -Encoding UTF8)
$history = @(Import-Csv (Join-Path $dir 'historical_validated.csv') -Encoding UTF8)
$fixtures = @(Get-Content (Join-Path $dir 'today_fixtures.json') -Raw -Encoding UTF8 | ConvertFrom-Json)
$coverage = @(Import-Csv (Join-Path $dir 'league_coverage.csv') -Encoding UTF8)
$invariant = [Globalization.CultureInfo]::InvariantCulture
function Num($v) { [double]::Parse([string]$v,$invariant) }
function Fmt($v,[int]$digits=2) { ([double]$v).ToString(('F'+$digits),$invariant).Replace('.',',') }
function Percent($v) { (Fmt (100*[double]$v))+'%' }
function Check($ok,[string]$message) { if(-not $ok){throw $message} }
function Minute([string]$v) { Check ($v -match '^(\d+)') ('Invalid minute '+$v); [int]$Matches[1] }
function Sample($x,[string]$team,[string]$venue='all') {
    @($history | Where-Object {$_.league -eq $x.league -and
        (($venue -eq 'all' -and ($_.home_id -eq $team -or $_.away_id -eq $team)) -or
         ($venue -eq 'home' -and $_.home_id -eq $team) -or ($venue -eq 'away' -and $_.away_id -eq $team))})
}
function Stat($rows,[string]$team) {
    $gf=0;$ga=0;$wins=0
    foreach($r in $rows){if($r.home_id -eq $team){$gf += [int]$r.home70;$ga += [int]$r.away70}else{$gf += [int]$r.away70;$ga += [int]$r.home70}; if($r.success70 -eq 'True'){$wins++}}
    [pscustomobject]@{n=@($rows).Count;gf=$gf;ga=$ga;wins=$wins}
}
Check ($models.Count -eq 27) 'Expected 27 models'
Check (@($models|Group-Object league,id|Where-Object Count -gt 1).Count -eq 0) 'Duplicate fixtures'
Check (@($history|Group-Object league,id|Where-Object Count -gt 1).Count -eq 0) 'Duplicate history'
foreach($r in $history){
    Check ([DateTimeOffset]::Parse($r.date) -lt [DateTimeOffset]::Parse('2026-11-02T00:00:00+02:00')) ('Future history '+$r.id)
    $hg=@($r.home_goals -split ';'|Where-Object {$_ -ne ''});$ag=@($r.away_goals -split ';'|Where-Object {$_ -ne ''})
    Check ($hg.Count -eq [int]$r.home_score -and $ag.Count -eq [int]$r.away_score) ('Score mismatch '+$r.id)
    $h70=@($hg|Where-Object {(Minute $_) -le 70}).Count;$a70=@($ag|Where-Object {(Minute $_) -le 70}).Count
    Check ($h70 -eq [int]$r.home70 -and $a70 -eq [int]$r.away70) ('Minute count mismatch '+$r.id)
    Check (([string]($h70+$a70 -gt 0)) -eq $r.success70) ('Event label mismatch '+$r.id)
    $first=@(@($hg)+@($ag)|Sort-Object {Minute $_}|Select-Object -First 1)
    Check (($first.Count -eq 0 -and $r.first_goal -eq '') -or ($first.Count -eq 1 -and (Minute $first[0]) -eq (Minute $r.first_goal))) ('First goal mismatch '+$r.id)
}
$tz=[TimeZoneInfo]::FindSystemTimeZoneById('GTB Standard Time')
$asof=[DateTimeOffset]::Parse($RecordedAtUtc)
$recordedRo=[TimeZoneInfo]::ConvertTime($asof,$tz).ToString('yyyy-MM-ddTHH:mm:sszzz')
$enriched=New-Object System.Collections.Generic.List[object]
foreach($x in $models){
    $date=[DateTimeOffset]::Parse($x.date)
    Check ($date -gt $asof) ('Already started fixture '+$x.id)
    $f=@($fixtures|Where-Object {$_.id -eq $x.id -and $_.league -eq $x.league})
    Check ($f.Count -eq 1 -and $f[0].state -eq 'pre' -and $f[0].status -eq 'STATUS_SCHEDULED' -and $f[0].time_valid) ('Status mismatch '+$x.id)
    $allA=@(Sample $x $x.home_id);$allB=@(Sample $x $x.away_id)
    $venueA=@(Sample $x $x.home_id 'home');$venueB=@(Sample $x $x.away_id 'away')
    $a=Stat $allA $x.home_id;$b=Stat $allB $x.away_id;$av=Stat $venueA $x.home_id;$bv=Stat $venueB $x.away_id
    foreach($pair in @(@('home',$a),@('away',$b),@('home_venue',$av),@('away_venue',$bv))){
        $prefix=$pair[0];$s=$pair[1]
        foreach($z in @(@('n','n'),@('gf70','gf'),@('ga70','ga'),@('success','wins'))){
            Check ([int]$x.($prefix+'_'+$z[0]) -eq [int]$s.($z[1])) ('Input mismatch '+$x.id+' '+$prefix+' '+$z[0])
        }
    }
    # Equivalent expanded form, recomputed from the CSV goal observations.
    $ls=($a.gf/$a.n+$a.ga/$a.n+$b.gf/$b.n+$b.ga/$b.n)/2
    $lv=($av.gf/$av.n+$av.ga/$av.n+$bv.gf/$bv.n+$bv.ga/$bv.n)/2
    $lc=($ls+$lv)/2;$p=1-[math]::Exp(-$lc);$stress=1-[math]::Exp(-.8*$lc)
    Check ([math]::Abs($p-(Num $x.p_raw)) -lt 1e-12 -and [math]::Abs($stress-(Num $x.p_stress)) -lt 1e-12) ('Probability mismatch '+$x.id)
    Check ([math]::Abs(1/$p-(Num $x.fair_odds)) -lt 1e-12) ('Fair odds mismatch '+$x.id)
    $unique=@(@($allA)+@($allB)|Sort-Object id -Unique)
    $uniqueVenue=@(@($venueA)+@($venueB)|Sort-Object id -Unique)
    $success=@($unique|Where-Object success70 -eq 'True').Count
    $successVenue=@($uniqueVenue|Where-Object success70 -eq 'True').Count
    $c=@($coverage|Where-Object league -eq $x.league)[0]
    $enriched.Add([pscustomobject]@{rank=$enriched.Count+1;league=$x.league;uefa_rank=[int]$c.uefa_rank;home=$x.home;away=$x.away;id=$x.id;
        kickoff_ro=[TimeZoneInfo]::ConvertTime($date,$tz).ToString('HH:mm');p_raw=$p;p_stress=$stress;fair_odds=1/$p;
        season_only=1-[math]::Exp(-$ls);venue_only=1-[math]::Exp(-$lv);
        first_goal_unique_success=$success;first_goal_unique_n=$unique.Count;first_goal_frequency=$success/$unique.Count;
        venue_unique_success=$successVenue;venue_unique_n=$uniqueVenue.Count;
        n_home=$a.n;n_away=$b.n;n_home_venue=$av.n;n_away_venue=$bv.n;source=$x.source})
}
Check (@($coverage|Where-Object {[int]$_.uefa_rank -gt 0}).Count -eq 20) 'Expected 20 UEFA national leagues'
Check (@($enriched|Where-Object uefa_rank -gt 0).Count -eq 16) 'Expected 16 top-20 fixtures'
Check (@($enriched|Where-Object uefa_rank -eq 0).Count -eq 11) 'Expected 11 additional fixtures'
$enriched|Export-Csv (Join-Path $dir 'comparison_with_first_goal.csv') -NoTypeInformation -Encoding UTF8
$coverage|Where-Object {[int]$_.uefa_rank -gt 0}|Sort-Object {[int]$_.uefa_rank}|ForEach-Object {
    [pscustomobject]@{uefa_rank=[int]$_.uefa_rank;country=$_.country;league=$_.league;fixtures=[int]$_.todays_fixtures;
        espn_supported=($_.league -notin @('pol.1','cro.1'));fotmob_daily_checked=$true;
        coverage_status=if([int]$_.todays_fixtures -gt 0){'all_identified_fixtures_calculated'}else{'none_found_in_checked_schedule'};
        history_status=if([int]$_.todays_fixtures -gt 0){'collected_for_all_candidates'}else{'not_needed_today'};
        recorded_at_ro=$recordedRo}
}|Export-Csv (Join-Path $dir 'universe_coverage.csv') -NoTypeInformation -Encoding UTF8
$leagueNames=@{'eng.1'='Premier League';'ita.1'='Serie A';'esp.1'='LaLiga';'fra.1'='Ligue 1';'por.1'='Liga Portugal';'tur.1'='Super Lig';'gre.1'='Super League';'pol.1'='Ekstraklasa';'den.1'='Superliga';'swe.1'='Allsvenskan';'rou.1'='SuperLiga Romania';'srb.1'='SuperLiga Serbia';'esp.2'='LaLiga 2';'fra.2'='Ligue 2';'tur.2'='1. Lig'}
$lines=New-Object System.Collections.Generic.List[string]
function Add([string]$line=''){ $lines.Add($line) }
Add '# Gol pana la 70: comparatia completa Europa, 2 noiembrie 2026'
Add
Add ('Calcul si verificari salvate la **'+$recordedRo+'**, Europe/Bucharest. Extragerile de program sunt anterioare acestei ore; toate cele 27 de ore de start sunt in viitor si statusurile salvate sunt SCHEDULED/pre. Acesta este un instantaneu, nu monitorizare live.')
Add 'Statusul celor doua finaliste a fost reverificat separat la 11:36:57-58 RO: ambele pre-meci, cu orele 19:30 si 20:00 RO. Dovada: Europe_Combined/finalists_live_status.json.'
Add
Add '**Rezultat:** topul strict al formulei este Sassuolo-Fiorentina, Sonderjyske-Copenhaga, Benfica-Alverca. Primele doua raman cele doua analize individuale cerute. Nu sunt un bilet combinat. Lorient-Brest coboara pe locul 8; alegerea initiala din doua ligi era incompleta.'
Add
Add 'Procentele sunt rezultate Poisson brute, necalibrate; nu sunt rate de reusita demonstrate. Nicio cota de bookmaker sau predictie de tipster nu a intrat in selectie sau calcule. Cota echitabila nu confirma o oferta disponibila de minimum 1,10. Nu a fost plasat niciun pariu.'
Add
Add '## Universul verificat'
Add
Add 'Primele 20 de asociatii din [clasamentul UEFA 2026, tabelul Kassies](https://kassiesa.net/uefa/data/method5/crank2026.html), numai prima liga masculina. Separat: Romania, Serbia si diviziile secunde urmarite deja in proiect, plus Ligue 2. Clasamentul fix 2026 nu este clasamentul provizoriu 2027. Arabia Saudita din vechea lista a proiectului nu este Europa. Nu pretindem ca am cercetat toate diviziile inferioare ale acestor tari.'
Add
Add '| Loc UEFA 2026 | Tara | Meciuri azi, prima liga | Verificarea programului |'
Add '|---:|---|---:|---|'
foreach($c in ($coverage|Where-Object {[int]$_.uefa_rank -gt 0}|Sort-Object {[int]$_.uefa_rank})){
    $status=if($c.league -eq 'cro.1'){'FotMob: niciun meci gasit; endpoint ESPN nesuportat'}elseif($c.league -eq 'pol.1'){'FotMob + program oficial Ekstraklasa'}elseif([int]$c.todays_fixtures -eq 0){'Niciun meci gasit in programele ESPN/FotMob'}else{'Program ESPN/FotMob; calcul pe minutele golurilor'}
    Add ('| '+$c.uefa_rank+' | '+$c.country+' | '+$c.todays_fixtures+' | '+$status+' |')
}
Add
Add 'Total: **16 meciuri din primele ligi ale topului 20 + 11 suplimentare = 27 evaluate**. In 10 dintre primele ligi nu am gasit meciuri azi. Absenta dintr-un agregator nu este o garantie oficiala impotriva unei modificari ulterioare de program.'
Add
Add '[Program global FotMob, 2 noiembrie](https://www.fotmob.com/api/data/matches?date=20261102). [Program ESPN, exemplu Serie A](https://site.api.espn.com/apis/site/v2/sports/soccer/ita.1/scoreboard?dates=20261102&limit=1000). [Program oficial polonez, etapa 13](https://ekstraklasa.org/en/news/schedule-of-the-202627-matchweek-13/). Pentru ligile fara meciuri nu a fost necesara descarcarea istoricului complet.'
Add
Add '## Toate partidele, aceeasi formula'
Add
Add 'Frecventa este numarul partidelor din sezonul curent ale ORICAREIA dintre cele doua echipe cu primul gol <=70, fara dublarea unui meci direct. Este descriptiva si nu este o a doua probabilitate validata. N sezon = numarul meciurilor gazdei/oaspetilor; N teren = gazda acasa/oaspete in deplasare.'
Add
Add '| Rang brut | Liga | Meci | Ora RO | P brut | Stres -20% | Echitabila | Istoric primul gol <=70 | N sezon; N teren |'
Add '|---:|---|---|---|---:|---:|---:|---:|---|'
foreach($x in $enriched){
    Add ('| '+$x.rank+' | '+$leagueNames[$x.league]+' | ['+$x.home+' - '+$x.away+']('+$x.source+') | '+$x.kickoff_ro+' | '+(Percent $x.p_raw)+' | '+(Percent $x.p_stress)+' | '+(Fmt $x.fair_odds 4)+' | '+$x.first_goal_unique_success+'/'+$x.first_goal_unique_n+' ('+(Percent $x.first_goal_frequency)+') | '+$x.n_home+'/'+$x.n_away+'; '+$x.n_home_venue+'/'+$x.n_away_venue+' |')
}
Add
Add 'Tenerife-Andorra este inclus la 15:00 RO: partida amanata duminica din cauza ploii a fost reprogramata luni. Acest lucru corecteaza excluderea provizorie din raportul initial. [RTVC, stirea reprogramarii](https://rtvc.es/entradas/). Universitatea Cluj-FCSB: providerul si SuperLiga indica 18:30 UTC / 20:30 RO; pagina clubului arata 21:00. Marchez conflictul de ora, fara a afecta numaratorile istorice. [SuperLiga](https://www.superliga.ro/meci/superliga-26-27-regular-ucj-fcsb?sectiune=live-blog), [club](https://www.fcucluj.ro/meciuri).'
Add
Add '## Formula si ipoteze'
Add
Add 'A = gazda, B = oaspete; GF/GA sunt numai goluri valide cu minutul de baza afisat <=70, inclusiv 45+. Nu se inmulteste din nou cu 70/90.'
Add
Add '```text'
Add 'lambda_sezon = ((GF_A/n_A + GA_B/n_B) + (GF_B/n_B + GA_A/n_A))/2'
Add 'lambda_teren = aceeasi formula, numai A acasa si B in deplasare'
Add 'lambda = 0,5*lambda_sezon + 0,5*lambda_teren'
Add 'p = 1 - exp(-lambda); p_stres = 1 - exp(-0,8*lambda); echitabila = 1/p'
Add '```'
Add
Add 'Ponderarea 50/50 este euristica; esantioanele se suprapun. Minimul tehnic pentru calcul, 5 meciuri de sezon si 3 pe teren pentru fiecare echipa, nu este o garantie de precizie. Nu exista calibrare pe un esantion independent. Multe goluri ulterioare deschiderii scorului si meciurile cu eliminari pot creste artificial lambda pentru evenimentul PRIMULUI gol. Stresul nu este un interval de incredere. Nu aplic bonusuri sau penalizari procentuale pentru tactica.'
Add
Add '## Inputuri si calcule pentru primele trei'
foreach($x in ($models|Select-Object -First 3)){
    Add
    Add ('### '+$x.home+' - '+$x.away)
    Add
    Add '| Esantion | N | GF pana la 70 | GA pana la 70 | Partide cu gol <=70 |'
    Add '|---|---:|---:|---:|---:|'
    foreach($q in @(@('Gazda, sezon','home'),@('Oaspete, sezon','away'),@('Gazda acasa','home_venue'),@('Oaspete deplasare','away_venue'))){$pfx=$q[1];Add ('| '+$q[0]+' | '+$x.($pfx+'_n')+' | '+$x.($pfx+'_gf70')+' | '+$x.($pfx+'_ga70')+' | '+$x.($pfx+'_success')+' |')}
    Add
    Add ('lambda_sezon = (('+$x.home_gf70+'/'+$x.home_n+' + '+$x.away_ga70+'/'+$x.away_n+') + ('+$x.away_gf70+'/'+$x.away_n+' + '+$x.home_ga70+'/'+$x.home_n+'))/2 = '+(Fmt (Num $x.lambda_season) 6)+'.')
    Add ('lambda_teren = (('+$x.home_venue_gf70+'/'+$x.home_venue_n+' + '+$x.away_venue_ga70+'/'+$x.away_venue_n+') + ('+$x.away_venue_gf70+'/'+$x.away_venue_n+' + '+$x.home_venue_ga70+'/'+$x.home_venue_n+'))/2 = '+(Fmt (Num $x.lambda_venue) 6)+'.')
    Add ('lambda_combinat = '+(Fmt (Num $x.lambda_combined) 6)+'; numai sezon = '+(Percent (1-[math]::Exp(-(Num $x.lambda_season))))+'; numai teren = '+(Percent (1-[math]::Exp(-(Num $x.lambda_venue))))+'; combinat = '+(Percent (Num $x.p_raw))+'; stres = '+(Percent (Num $x.p_stress))+'; echitabila = '+(Fmt (Num $x.fair_odds) 4)+'.')
}
Add
Add '## Istoricul verificat al finalistilor'
Add
Add 'Mai jos sunt ultimele maximum 10 meciuri DE CAMPIONAT ale fiecarui finalist, din sezonul actual. Sassuolo si Fiorentina au cate 9; cele 10 oficiale incluzand cupa sunt deja documentate in raportul initial. Pentru danezi si portughezi, aceste tabele NU sunt prezentate ca ultimele 10 din toate competitiile: cupa si meciurile europene sunt excluse din modelul de campionat si nu avem aici o serie completa a minutelor lor. Limitarea este explicita.'
foreach($x in ($models|Select-Object -First 3)){
    foreach($team in @($x.home_id,$x.away_id)){
        $name=if($team -eq $x.home_id){$x.home}else{$x.away}
        Add
        Add ('### '+$name+' - campionat')
        Add
        Add '| Data | Meci si scor final | Primul gol | Gol <=70 |'
        Add '|---|---|---:|---|'
        foreach($r in (Sample $x $team|Sort-Object date -Descending|Select-Object -First 10)){
            $first=if($r.first_goal -eq ''){'fara gol (0-0)'}else{$r.first_goal}
            Add ('| '+$r.date.Substring(0,10)+' | ['+$r.home+' - '+$r.away+' '+$r.home_score+'-'+$r.away_score+']('+$r.source+') | '+$first+' | '+$(if($r.success70 -eq 'True'){'DA'}else{'NU'})+' |')
        }
    }
}
Add
Add '## Analiza contextuala si decizia'
Add
Add '### 1. Sassuolo-Fiorentina, 19:30 RO'
Add
Add 'Serie A, etapa 10, campionat regulat, fara calificare pe un scor agregat. Ambele au nevoie de puncte: Sassuolo avea 8, Fiorentina 5 dupa 9 meciuri. Nu rezulta ca trebuie sa atace nechibzuit de la start; un egal poate fi acceptabil intr-un meci intre echipe aflate in dificultate. Prima deschidere a scorului <=70 apare in toate cele 18 partide distincte ale sezonului celor doua, cu esantion pe teren de numai 4+4 partide. Acesta este principalul argument pentru pastrarea candidatului dupa cautarea completa.'
Add
Add 'Aquilani vs Vanoli; formatii PROBABILE Sky 4-3-3 vs 4-3-2-1. Berardi si Lauriente pe laterale pot crea dueluri directe, iar intermediarii Fiorentinei pot ataca spatiul din fata fundasilor: acestea sunt scenarii tactice, nu bonusuri numerice. Sassuolo nu ii are pe Thorstvedt (suspendat), Kone, Leysen si Boloca; la Fiorentina sunt listati indisponibili Parisi, Atta, Ranieri si Pongracic. Absentele defensive pot ajuta producerea ocaziilor, dar niciun efect cauzal nu a fost cuantificat. [Sky, loturi si module probabile](https://sport.sky.it/calcio/serie-a/partite/2026/giornata-10/sassuolo-fiorentina/formazioni).'
Add
Add '**Corectie de sursa:** Sassuolo-Lazio 2-6, 27 octombrie: al doilea gol Noslin a fost la 65, nu 75. Toti cei sase marcati de Lazio sunt <=70. Sursa oficiala Lazio si ESPN se aliniaza pentru aceasta clasificare; vechiul raport se baza pe o transcriere eronata a minutului. De aceea P brut trece de la 93,87% la 94,40%. Diferentele de 1-2 minute dintre furnizori pentru alte goluri din acel meci nu trec pragul de 70. [Tablou oficial Lazio](https://sslazio.hiwaymedia.io/it/news/match-report/serie-a-enilive-or-sassuolo-lazio-le-formazioni-ufficiali-a).'
Add
$s=$models[0]
$sasAll=@(Sample $s $s.home_id);$sasHome=@(Sample $s $s.home_id 'home')
$outlier=@($sasAll|Where-Object {$_.home -eq 'Sassuolo' -and $_.away -eq 'Lazio' -and $_.home_score -eq '2' -and $_.away_score -eq '6'})
Check ($outlier.Count -eq 1) 'Expected the Sassuolo-Lazio outlier'
$a=Stat @($sasAll|Where-Object id -ne $outlier[0].id) $s.home_id
$av=Stat @($sasHome|Where-Object id -ne $outlier[0].id) $s.home_id
$b=Stat @(Sample $s $s.away_id) $s.away_id;$bv=Stat @(Sample $s $s.away_id 'away') $s.away_id
$sensitivityLambda=(($a.gf/$a.n+$a.ga/$a.n+$b.gf/$b.n+$b.ga/$b.n)/2+($av.gf/$av.n+$av.ga/$av.n+$bv.gf/$bv.n+$bv.ga/$bv.n)/2)/2
Add ('**Riscul decisiv:** acel 2-6 a avut eliminarea lui Thorstvedt la 31, iar sase goluri primite pana la 70 umfla intensitatea. Eliminand INTREG meciul numai ca diagnostic, N Sassuolo devine 8 sezon/3 acasa si P scade la **'+(Percent (1-[math]::Exp(-$sensitivityLambda)))+'**. Nu inlocuiesc estimarea cu acest rezultat ales dupa observarea unui outlier. Arata fragilitatea celor 94,40%, nu o probabilitate reala alternativa validata. Acesta este un contraargument mai relevant decat diferenta de 0,04 puncte procentuale fata de meciul danez.')
Add
Add '### 2. Sonderjyske-Copenhaga, 20:00 RO'
Add
Add 'Superliga daneza, etapa 13, faza regulata de 22 etape; puncte in campionat, fara departajare la finalul serii. Sonderjyske este ultima, cu 8 puncte dupa 12 jocuri; Copenhaga a doua, cu 26. La actualizarea clasamentului dupa duminica, Midtjylland are 31 din 13; nu mai folosesc diferenta de doua puncte din avancronica publicata sambata. FCK cauta revenirea dupa trei etape fara victorie. Un egal ar ajuta relativ mai mult gazdele, astfel ca blocul jos este un scenariu plauzibil. [Avancronica oficiala FCK](https://www.fck.dk/nyhed/fokus-paa-soenderjyske-fc-koebenhavn), [clasament actualizat in avancronica Sportportalen](https://sportportalen.dk/nyheder/fodbold/soenderjyske-fck-optakt-superligaen-runde-13).'
Add
Add 'Gazdele lui Fatah Abdirahman au primit cele mai multe goluri din liga, dar ultimele doua acasa au fost victorii 4-2 si 4-0: argumentul este ca pot contribui si ele la gol, nu doar ca FCK trebuie sa marcheze. Au urmat 1-5 la Horsens si eliminare in cupa dupa prelungiri, miercuri. FCK a jucat joi in cupa si mai are Iberia pe 5 noiembrie si derby cu Brondby pe 8: program incarcat si risc de rotatie, nu dovada unei oboseli inevitabile. Copenhaga trebuie sa desfaca eventualul bloc jos prin latime, combinatii intre linii si prezenta in careu; daca incepe lent, volumul total istoric de goluri poate sa nu se traduca intr-un gol devreme.'
Add
Add 'Bo Svensson a convocat aceiasi 20 de jucatori ca in cupa. Sunt confirmati ABSENTI Magnus Mattsson, Thapelo Maseko, Marvin Nasnas si Robert Silva. Elyounoussi, Cornelius si Mads Emil Madsen sunt convocati. Acesta este lotul convocat, NU un 11 de start confirmat. Nu am verificat o formatie probabila actuala suficient de solida pentru a o publica drept fapt si nu am reconciliat integral lotul gazdelor. [Lot oficial FCK, 1 noiembrie 18:25](https://www.fck.dk/nyhed/truppen-mod-soenderjyske-50).'
Add
Add '**Contradictia importanta:** 94,36% din Poisson fata de 20/23 = 86,96% in istoricul sezonului fara dublarea meciului direct. Numai configuratia de teren are 10/12 = 83,33%. Esecurile sunt OB-Sonderjyske 1-0 (primul gol 81), Sonderjyske-Viborg 0-0 si Brondby-FCK 0-1 (79). Inclusiv o favorita care castiga poate rata evenimentul <=70. Modelul foloseste goluri dupa primul; nu avem justificare pentru a considera cei 94% bine calibrati. Pastrez partida drept a doua in topul FORMULEI, cu aceasta rezerva substantiala.'
Add
Add '### Benfica-Alverca, a treia alternativa, 22:15 RO'
Add
Add 'Model 92,45%, stres 87,34%, istoric actual 18/18 partide distincte cu primul gol <=70. Benfica are 20 goluri marcate pana la 70 in 9 jocuri, Alverca 11 primite. Sunt numai cate 4 meciuri relevante pe teren, deci 18/18 nu inseamna certitudine. Marco Silva si Sergio Ferreira sunt antrenorii mentionati in avancronica, iar antrenorul gazdelor avertizeaza asupra dificultatii, fara a se baza pe diferenta din clasament. [Declaratie pre-meci, SPORT TV](https://www.sporttv.pt/noticias/857511/marco-silva-alverca-vai-ser-dificil-nao-olhamos-para-tabela).'
Add
Add 'Benfica este alternativa cu frecventa istorica mai curata decat meciul danez, dar raw si stres sunt mai mici. Nu am verificat integral loturile actuale; listele publice gasite erau contradictorii sau aveau date vechi. Nu mut automat meciul pe locul 2 doar ca istoricul are 100%, nici nu pretind ca are o cota oferita de minimum 1,10. Poate intra in comparatia finala la confirmarea loturilor; nu este o a treia selectie de jucat.'
Add
Add '### De ce nu celelalte'
Add
Add 'Deportivo-Osasuna este urmatoarea dupa model (91,37%, 19/20 istoric), dar fara verificare integrala a loturilor nu pretind o analiza contextuala identica cu cea a finalistilor. Atalanta-Parma are 90,92%, insa Parma a marcat ZERO goluri <=70 in cele patru deplasari si istoricul sezonului celor doua este 15/18. Radomiak-Slask are 89,89%, U Cluj-FCSB 89,76%, Lorient-Brest 89,46%: sunt alternative reale, dar nu conduc formula aplicata uniform. U Cluj-FCSB are 23/26 in istoric, cu doua esecuri in deplasarile FCSB (0-0 cu Sepsi si primul gol 77 la CFR) si primul gol 73 in Voluntari-U Cluj. Reims-Sochaux are numai 15/20; liga sau numele favoritei nu elimina aceste esecuri. Suedia ofera un sezon mai lung, dar ambele meciuri au raw sub 85%. Lista completa de mai sus pastreaza restul comparatiei, fara selectie timpurie pe tari.'
Add
Add 'Nu am validat prognoze meteo orare, starea gazonului sau toate deplasarile pentru cele 27; acestea nu intra ca date inventate si nu justific o precizie numerica suplimentara. Cercetarea este completa ca PROGRAM si CALCUL pentru universul definit; verificarea contextuala amanuntita se concentreaza pe finalisti si nu este echivalenta pentru toate cele 27. Nu declar un avantaj financiar sau o rata reala de 90%.'
Add
Add '## Controlul datelor si reproducere'
Add
Add ('Sunt '+$history.Count+' partide istorice distincte validate: toate golurile reconciliate cu scorul final pe fiecare echipa, minutele si etichetele <=70 recalculate, autogolurile atribuite beneficiarului. Un meci suedez fara lista completa a golurilor a fost respins (Orgryte-Elfsborg, 29 mai); nu implica nicio echipa suedeza care joaca azi. 27/27 modele trec verificarea inputurilor, formulei, pretului echitabil, statusului salvat si orei de start. Nu sunt rezultate de backtest.')
Add
Add 'Datele ESPN pentru sezonul calendaristic 2026 au fost filtrate dupa season.year=2026, status final si data anterioara zilei analizei. Pentru Suedia: sezon 2026; pentru celelalte: 2026/27. Nu se amesteca sezonul 2025/26. Pentru Polonia, Romania, Serbia si Turcia 2 s-a folosit FotMob, toate partidele sezonului echipelor de azi. Alte ligi ESPN contin istoric si pentru echipe care nu joaca azi; nu intra in numaratorile candidatilor.'
Add
Add 'Fisiere: Europe_Combined/all_models.csv (toate inputurile), comparison_with_first_goal.csv (comparatia finala), historical_validated.csv (1306 partide), historical_rejected.csv, league_coverage.csv, today_fixtures.json. JSON-urile ESPN/FotMob sunt selectii de campuri sportive publice; cotele au fost omise. Datele initiale si jurnalul initial nu sunt suprascrise.'
Add
Add 'Reproducere offline: powershell -NoProfile -File Analysis/Goal70_2026-11-02/evaluate_europe_espn.ps1 -DataDirectory Analysis/Goal70_2026-11-02/Europe_Combined; apoi verify_and_report_europe.ps1 -RecordedAtUtc cu ORA ANALIZEI SALVATE, nu cu o data ulterioara meciurilor. Scriptul refuza un kickoff deja trecut la ora introdusa.'
$lines | Set-Content (Join-Path $PSScriptRoot 'EUROPE_ALL_MATCHES.md') -Encoding UTF8
$checks=[pscustomobject]@{recorded_at_utc=$asof.ToString('o');recorded_at_ro=$recordedRo;models=27;top20_fixtures=16;additional_fixtures=11;historical_valid=$history.Count;historical_rejected=1;checks='PASS: unique identities, past-only history, score reconciliation, <=70 labels, independent count/formula, saved scheduled status, future kickoff, 20-league universe';calibrated=$false}
$checks|ConvertTo-Json|Set-Content (Join-Path $dir 'verification.json') -Encoding UTF8
$journal=@($enriched|Select-Object -First 2|ForEach-Object {[pscustomobject]@{analysis_date='2026-11-02';recorded_at_ro=$recordedRo;version='full_europe_20_v2';home=$_.home;away=$_.away;kickoff_ro=$_.kickoff_ro;market='GOAL_BY_70';raw_rank=$_.rank;p_raw=$_.p_raw;p_stress=$_.p_stress;fair_odds=$_.fair_odds;historical_success=$_.first_goal_unique_success;historical_n=$_.first_goal_unique_n;calibrated=$false;market_price_verified=$false;result='';status='prospective_analysis_only'}})
$journal|Export-Csv (Join-Path $PSScriptRoot 'prospective_selections_europe_v2.csv') -NoTypeInformation -Encoding UTF8
$checks|ConvertTo-Json
