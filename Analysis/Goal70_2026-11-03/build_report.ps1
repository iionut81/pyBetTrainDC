param([string]$RecordedAtUtc)
$ErrorActionPreference='Stop'
$root=$PSScriptRoot;$ci=[Globalization.CultureInfo]::InvariantCulture
function Num($v){[double]::Parse([string]$v,$ci)}
function Fmt($v,[int]$n=2){([double]$v).ToString(('F'+$n),$ci).Replace('.',',')}
function Pct($v){(Fmt (100*(Num $v)))+'%'}
$tz=[TimeZoneInfo]::FindSystemTimeZoneById('GTB Standard Time')
$stamp=[TimeZoneInfo]::ConvertTime([DateTimeOffset]::Parse($RecordedAtUtc),$tz).ToString('yyyy-MM-ddTHH:mm:sszzz')
$models=@(Import-Csv (Join-Path $root 'all_models.csv') -Encoding UTF8)
$comparison=@(Import-Csv (Join-Path $root 'comparison_verified.csv') -Encoding UTF8)
$coverage=@(Import-Csv (Join-Path $root 'Data/league_coverage.csv') -Encoding UTF8|Where-Object {$_.league -notin @('uefa.champions','ukr.1','svk.1')})
if($coverage.Count -ne 29){throw 'Expected 29 baseline leagues'}
$inventory=@($coverage|ForEach-Object {
 $n=@($models|Where-Object competition -eq $_.league).Count
 [pscustomobject]@{uefa_2026_rank=$_.uefa_rank;league=$_.league;country=$_.country;identified_fixtures=$n;calculated_fixtures=$n;espn_schedule_supported=$_.today_ok;fotmob_global_checked=$true;coverage=if($n){'all_identified_fixtures_calculated'}else{'no_fixture_in_checked_schedules'};limit=if($_.today_ok -ne 'True'){'ESPN unsupported; absence based on FotMob daily inventory'}else{''};recorded_at_ro=$stamp}
})
$inventory|Export-Csv (Join-Path $root 'universe_coverage.csv') -NoTypeInformation -Encoding UTF8
$global=Get-Content (Join-Path $root 'fixtures_global.json') -Raw -Encoding UTF8|ConvertFrom-Json
$europe=@('ENG','SCO','ESP','CYP','DEN','POL','CRO','MNE','INT')
$exclusions=@($global.leagues|Where-Object {$_.country -in $europe -and $_.name -notin @('Champions League','Championship','Allsvenskan','DBU Pokalen') -and $_.name -notmatch '^AFC'}|ForEach-Object {
 [pscustomobject]@{country=$_.country;competition=$_.name;fixtures=@($_.matches).Count;reason=if($_.name -match 'U18|Youth|Reina|A-Liga'){'Women or youth outside men senior scope'}elseif($_.name -match 'Cup|Copa|Trophy'){'Outside baseline league scope; lower division or mixed opposition without collected comparable history'}else{'Outside top20 first divisions and previously tracked second divisions'};source=$global.source}
})
$exclusions|Export-Csv (Join-Path $root 'exclusions.csv') -NoTypeInformation -Encoding UTF8
$l=New-Object System.Collections.Generic.List[string]
function Add([string]$s=''){$l.Add($s)}
Add '# Gol pana la 70 inclusiv: 3 noiembrie 2026'
Add
Add ('**Selectia individuala a analistului: QPR - Burnley, 21:45 Romania, Championship, minimum un gol valid al oricarei echipe pana la minutul afisat 70 inclusiv.** Instantaneu prospectiv salvat la '+$stamp+'. Utilizatorul nu a declarat inca o alegere pentru azi; niciun pariu, pret sau rezultat nu este presupus.')
Add
Add 'Model brut 93,63%; scenariu intensitate -20%: 88,95%; cota echitabila strict matematica 1,0681. Nu este o probabilitate calibrata sau o cota oferita. La un pret ipotetic 1,10, pragul de rentabilitate este 90,91%; scenariul de stres este sub acest prag. Nu declar un avantaj demonstrat sau profitabilitate.'
Add
Add '## Acoperire, limite si excluderi'
Add
Add 'Universul de baza este fixat dupa [clasamentul UEFA 2026](https://kassiesa.net/uefa/data/method5/crank2026.html), nu clasamentul provizoriu 2027: primele ligi masculine din 20 asociatii, plus Romania, Serbia si 7 divizii secunde deja urmarite. Sunt 29 ligi inventariate. Am adaugat separat toate cele 9 partide Champions League si Lyngby-Randers din Cupa Danemarcei. Nu numesc acest inventar toate diviziile Europei.'
Add
Add '| Rang UEFA 2026 | Tara / liga | Meciuri identificate | Calculate |'
Add '|---|---|---:|---:|'
foreach($r in $inventory|Sort-Object @{e={if([int]$_.uefa_2026_rank -gt 0){[int]$_.uefa_2026_rank}else{99}}},league){Add ('| '+$(if([int]$r.uefa_2026_rank -gt 0){$r.uefa_2026_rank}else{'Supliment'})+' | '+$r.country+' / '+$r.league+' | '+$r.identified_fixtures+' | '+$r.calculated_fixtures+' |')}
Add
Add 'Rezultat: 1 Allsvenskan + 9 Championship in universul de baza; 9 UCL + 1 Cupa Danemarcei suplimentar = **20 partide identificate si 20 calcule**. Nicio partida de prima liga in Franta sau Romania nu a fost identificata azi in programele verificate. Inventarul programului si calculul tuturor candidatilor sunt separate de verificarile contextuale aprofundate de mai jos.'
Add
Add '[Program FotMob](https://www.fotmob.com/api/data/matches?date=20261103), [Championship ESPN](https://site.api.espn.com/apis/site/v2/sports/soccer/eng.2/scoreboard?dates=20261103&limit=1000), [program Allsvenskan oficial](https://allsvenskan.se/nyheter/sa-spelas-omgang-24-30-av-allsvenskan/), [program UEFA](https://www.uefa.com/uefachampionsleague/news/02a8-2174c9e9019d-f909a77bd77a-1000--2026-27-champions-league-all-the-league-phase-fix/). Ultima reverificare pentru toate cele 20 de partide: 10:36:56 RO, toate programate/neincepute, ore neschimbate; `live_status.json`.'
Add
Add 'Limita inventarului: ESPN nu suporta aici pol.1, cro.1, srb.1 si bel.2 (HTTP 400). Pentru absenta partidelor in aceste patru ligi ma bazez pe inventarul global FotMob, nu pe raspunsul ESPN. Nu prezint aceasta absenta drept confirmare independenta din doua surse. Istoricul integral nu a fost colectat pentru ligile fara candidati azi.'
Add
Add 'Excluderile observate sunt in `exclusions.csv`: inclusiv 2 meciuri din Polonia liga a doua (neinclusa anterior), 2 din Muntenegru (in afara top20), 16 Challenge Cup Scotia, 2 Copa del Rey, 2 Cupa Ciprului, 2 Cupa Croatiei, League One/Two si cupe inferioare, juniori si fotbal feminin. Nu am calculat aceste meciuri si nu afirm ca sunt mai slabe sportiv; nu au istoricul comparabil colectat in acest audit.'
Add
Add 'Date: 1.520 partide istorice validate, 4 respinse prin neconcordanta dintre scor si lista golurilor; niciuna dintre cele patru nu implica echipele candidate. Sezoanele sunt 2026 pentru Suedia/Norvegia si 2026/27 pentru restul. Au fost verificate independent in ESPN/FotMob toate cele 78 de observatii pe echipa QPR/Burnley/Sirius/Halmstad; acestea reprezinta 77 partide distincte, deoarece Halmstad-Sirius apare in ambele istorice. Scorurile si numaratorile GF70/GA70 concorda. Suplimentar sunt salvate ultimele 10 oficiale Derby/Norwich.'
Add
Add '## Clasamentul complet al modelului brut'
Add
Add 'Acesta este un clasament al valorilor aceleiasi formule, nu al unor probabilitati comparabile validate. UCL si cupa folosesc proiectii din campionatele interne. Aceasta schimbare de populatie este motivul pentru care nu aleg automat cea mai mare valoare UCL. Cele 20 randuri nu sunt 20 recomandari.'
Add
Add '| Rang brut | Meci | Competitie | Ora RO | Model | Stres -20% | Cota echitabila | Prim gol <=70, istoric distinct | Acasa/deplasare |'
Add '|---:|---|---|---|---:|---:|---:|---:|---:|'
foreach($x in $comparison){Add ('| '+$x.rank+' | '+$x.home+' - '+$x.away+' | '+$x.competition+' | '+$x.kickoff_ro+' | '+(Pct $x.p_raw)+' | '+(Pct $x.p_stress)+' | '+(Fmt (Num $x.fair_odds) 4)+' | '+$x.first_goal_success+'/'+$x.first_goal_n+' | '+$x.venue_success+'/'+$x.venue_n+' |')}
Add
Add '## Alegerea: QPR - Burnley'
Add
Add 'Este primul model dintre meciurile cu istoric din aceeasi competitie, dupa compararea tuturor celor 20. Are contributii ofensive de ambele parti si rezista numeric excluderii rezultatelor extreme/eliminarilor. Acesta este motivul alegerii; nu pretind ca tactica elimina riscul schimbarii antrenorului Burnley.'
Add
Add 'Championship 2026/27, etapa 14, sezon regulat la Loftus Road; nu exista retur eliminatoriu, gol in deplasare ori prelungiri. QPR este pe locul 9 (20p), Burnley pe 22 (9p). Gazdele au interes pentru apropierea de playoff, oaspetii pentru iesirea din zona retrogradarii. [Clasament/program EFL](https://www.efl.com/competitions/efl-championship/), [context local](https://www.playmakerstats.com/news/preview-qpr-vs-burnley-team-news-lineups-predictions-efl-championship-03-11/1249940). Ordinea de baza a clasamentului foloseste puncte, golaveraj, apoi goluri marcate; nu este o situatie de calificare decisa azi. [Regulament EFL, sectiunea 9](https://images.gc.eflservices.co.uk/6a32d710-5b0b-11f0-ad0f-97502a343283.pdf) este editia 2025/26; intreaga succesiune a departajarii 2026/27 nu a fost verificata, deci nu o presupun neschimbata.'
Add
Add 'Un egal nu rezolva obiectivele niciuneia, dar poate fi acceptabil pentru Burnley in deplasare: nu transform nevoia de puncte intr-o promisiune de atac din primul minut. Ambele au jucat pe 31 octombrie, deci aproximativ trei zile intre meciuri. QPR revine acasa; Burnley calatoreste in Anglia, fara diferenta de fus orar.'
Add
Add 'Forma si antrenori: Julien Stephan conduce QPR, care vine dupa 5-0 acasa cu Swansea si 1-4 la Southampton. Kone a marcat de patru ori in aceste doua meciuri. Celestini a preluat Burnley pe 28 octombrie; debutul din 31 octombrie a fost 0-0 cu Portsmouth. [Preview oficial Burnley](https://www.burnleyfootballclub.com/media-article/match-preview-qpr-v-burnley). Contradictie relevanta: afirmatia din acest preview privind numai doua infrangeri QPR in ultimele noua nu concorda cu istoricul salvat, care include Birmingham, Southampton si Middlesbrough; nu o folosesc in evaluare.'
Add
Add 'Tactic, ultimele formule verificate au fost QPR 4-4-2 si Burnley 4-2-3-1. Sunt formule din ultimul meci, nu echipe confirmate azi. Smyth-Kone, sprijiniti de Poku/Vale, ofera o cale prin joc lateral si prezenta in careu; cu doi atacanti, QPR poate ocupa fundasii centrali. Burnley are variante de tranzitie prin Ramazani/Edwards si Vardy. Acestea sunt scenarii tactice, nu bonusuri numerice. Un bloc mediu/jos Burnley, protejat de doi mijlocasi, ar putea reduce ritmul si accesul central. Nu exista suficiente meciuri cu Celestini la Burnley pentru a atribui echipei un stil nou deja stabilizat. Ultimele formule sunt verificate in istoricul FotMob; proiectiile de titulari sunt [neconfirmate](https://www.sportsmole.co.uk/football/qpr/preview/qpr-vs-burnley-prediction-team-news-lineups_608199.html). Nu preiau predictia acelui site.'
Add
Add 'Loturi: Burrell, Lamptey si Madsen sunt raportati indisponibili la QPR; Amdouni, accidentat la glezna, este raportat absent la Burnley. [Burnley Express, 2 noiembrie](https://www.burnleyexpress.net/sport/football/burnley-fc/qpr-v-burnley-early-team-news-9323189). Madsen/Burrell reduc optiunile creative/ofensive. Celestini confirma revenirea lui Cullen, Banel si Ramsey, dar si restrictii de minute; Broja nu este inca aproape de revenire, Satpayev se recupereaza. Antrenorul spune ca programul permite mai ales recuperare, putin antrenament. [Conferinta oficiala](https://www.burnleyfootballclub.com/media-article/celestinis-press-points-qpr). Nicio formula de start nu este confirmata la ora analizei. Lista secundara care afirma ca QPR nu are indisponibili este in conflict cu relatarea locala; nu o consider dovada de lot complet.'
Add
Add 'Conditii: [Met Office, actualizare 3 noiembrie 04:00 UTC](https://weather.metoffice.gov.uk/forecast/gcpvj0u07) indica nori, averse izolate seara si tendinta de uscare, vant mai accentuat spre coasta. Nu apare de aici un motiv verificat pentru a invalida meciul; este prognoza regionala, nu inspectie a gazonului. Starea exacta a suprafetei si planul efectiv de calatorie nu sunt confirmate. Nu am folosit xG: nu am un set contemporan cu aceeasi definitie si acoperire pentru toate cele 20 partide.'
Add
Add '### Numaratori si formula neschimbata'
Add
Add 'Golurile de mai jos sunt numai cele marcate/primite pana la minutul afisat 70 inclusiv, nu goluri la pauza sau goluri totale la 90. Se includ 45+, se exclud golurile anulate, prelungirile eliminatorii si departajarea. Nu inmultesc din nou cu 70/90.'
Add
foreach($name in @('Queens Park Rangers','Birmingham City','Derby County','IK Sirius')){
 $x=@($models|Where-Object home -eq $name)[0]
 Add ('#### '+$x.home+' - '+$x.away)
 Add
 Add '| Esantion | N | GF70 | GA70 | Gol <=70 | Deschidere dupa 70 | 0-0 |'
 Add '|---|---:|---:|---:|---:|---:|---:|'
 foreach($prefix in @('home','away','home_venue','away_venue')){
  $label=switch($prefix){'home'{$x.home+' sezon'}'away'{$x.away+' sezon'}'home_venue'{$x.home+' acasa'}'away_venue'{$x.away+' deplasare'}}
  Add ('| '+$label+' | '+$x.($prefix+'_n')+' | '+$x.($prefix+'_gf70')+' | '+$x.($prefix+'_ga70')+' | '+$x.($prefix+'_success')+' | '+$x.($prefix+'_late')+' | '+$x.($prefix+'_0_0')+' |')
 }
 Add
 Add ('GF70/meci pe teren: gazda '+(Fmt ((Num $x.home_venue_gf70)/(Num $x.home_venue_n)))+'; oaspete '+(Fmt ((Num $x.away_venue_gf70)/(Num $x.away_venue_n)))+'.')
 Add
 Add ('lambda_sezon = (('+$x.home_gf70+'/'+$x.home_n+' + '+$x.away_ga70+'/'+$x.away_n+') + ('+$x.away_gf70+'/'+$x.away_n+' + '+$x.home_ga70+'/'+$x.home_n+')) / 2 = '+(Fmt (Num $x.lambda_season) 6)+'.')
 Add
 Add ('lambda_teren = (('+$x.home_venue_gf70+'/'+$x.home_venue_n+' + '+$x.away_venue_ga70+'/'+$x.away_venue_n+') + ('+$x.away_venue_gf70+'/'+$x.away_venue_n+' + '+$x.home_venue_ga70+'/'+$x.home_venue_n+')) / 2 = '+(Fmt (Num $x.lambda_venue) 6)+'.')
 Add
 Add ('lambda_combinat = 0,5 x lambda_sezon + 0,5 x lambda_teren = '+(Fmt (Num $x.lambda_combined) 6)+'; p = 1-exp(-lambda_combinat) = '+(Pct $x.p_raw)+'. Numai sezon: '+(Pct (1-[math]::Exp(-(Num $x.lambda_season))))+'; numai teren: '+(Pct (1-[math]::Exp(-(Num $x.lambda_venue))))+'; intensitate -20%: '+(Pct $x.p_stress)+'; cota echitabila = 1/p = '+(Fmt (Num $x.fair_odds) 4)+'.')
 Add
}
Add 'Ponderarea 50/50 este euristica; esantioanele sezon/teren se suprapun si nu sunt dovezi independente. Modelul Poisson nu este calibrat pentru timpul primului gol. Golurile suplimentare dupa deschiderea scorului, inclusiv in meciuri devenite foarte deschise, pot umfla intensitatea si supraestima sansa de a evita 0-0 pana la 70.'
Add
Add 'La QPR-Burnley, modelul 93,63% depaseste frecventa istorica distincta 23/26 = 88,46%; pe teren frecventa este 11/12 = 91,67%. Ultimele 10 oficiale sunt 9/10 pentru fiecare, 18/20 distincte. Cele 3 esecuri de sezon sunt QPR-Bolton 0-0 (22 august), Charlton-QPR 0-0 (9 septembrie), Burnley-Portsmouth 0-0 (31 octombrie); nicio deschidere tarzie in acest esantion. Nu inlocuiesc aceste esecuri cu succes si nu prezint valoarea modelului drept rata istorica.'
Add
Add '### Sensibilitate la eliminari si rezultate extreme'
Add
Add '| Partida | Diagnostic | p | N sezon A/B | N teren A/B |'
Add '|---|---|---:|---:|---:|'
foreach($s in Import-Csv (Join-Path $root 'sensitivity.csv')){Add ('| '+$s.match+' | '+$s.variant+' | '+(Pct $s.p)+' | '+$s.home_n+'/'+$s.away_n+' | '+$s.home_venue_n+'/'+$s.away_venue_n+' |')}
Add
Add 'La QPR, eliminarea din QPR-Preston a venit la 15, dupa primul gol din minutul 1; in Norwich-Burnley eliminarea a venit la 77, dupa golul din minutul 2. Aceste eliminari nu au cauzat cronologic primul gol. Excluderea integrala este doar un test conservator de sensibilitate, nu o corectie cauzala; eliminarea de la 77 nici nu poate schimba golurile deja inscrise pana la 70. Rezultatul fara QPR-Swansea 5-0 este 92,83%, deci acel rezultat ridica valoarea, dar nu explica singur semnalul. Nu rescriu predictia initiala cu cea mai convenabila varianta.'
Add
Add '## Alternative si contraargumente'
Add
Add '**Feyenoord-Inter, 22:00:** 94,28% brut / 89,86% stres, maximul numeric. Dar intrarile sunt Eredivisie/Serie A, adversari neuniformi, numai 5+5 meciuri pe teren; in UCL au impreuna doar 6 observatii, cu 5 succese. Nu il aleg peste QPR pe baza acestui transfer nevalidat. Van Bronckhorst confirma revenirea lui Shaqueel van Persie si Reiss Nelson, St Juste ramane absent. [Feyenoord, 2 noiembrie](https://www.feyenoord.com/nl/news/van-persie-en-nelson-bij-wedstrijdselectie-tegen-inter-021126). Chivu confirma Provedel recuperat si Pavard revenit la grup. [Inter, conferinta](https://www.inter.it/it/notizie/chivu-conferenza-stampa-vigilia-feyenoord-inter-champions-league-2026-27). Listele secundare care ii dau pe cei reveniti drept absenti sunt depasite. Barcelona-Villa si Galatasaray-Stuttgart au aceeasi limita a transferului intre ligi; nu pretind ca le-am verificat contextual la aceeasi adancime.'
Add
Add '**Birmingham-Millwall, 22:00:** al doilea model din aceeasi competitie, 91,69% / 86,33%; 23/26 istoric si 11/12 pe teren. Profil ofensiv bilateral: 11/6 si 7/6 GF70 pe teren. Ingrijorarile privind Beadle/Priske si suspendarea Metcalfe apar in [preview-ul secundar](https://www.sportsmole.co.uk/football/birmingham-city/preview/birmingham-vs-millwall-prediction-team-news-lineups_608183.html); listele de absenti publicate de agregatoare nu concorda. Nu tratez ca verificate toate numele si nici nu pretind verificare contextuala completa pentru acest candidat. QPR are scorul brut si stresul superior; istoricul succeselor nu le departajeaza.'
Add
Add '**Derby-Norwich, 21:45:** 91,63% / 86,26%; istoric 25/26, pe teren 12/12, mai bun descriptiv decat QPR. Totusi Derby are numai 4 GF70 in 6 acasa (0,67/meci), fata de 1,83 la QPR; nu aleg mecanic 12/12 dintr-un esantion mic peste model. Clement descrie Derby ca direct, puternic in dueluri, cere control cu mingea, recunoaste vulnerabilitatea propriei aparari si posibila rotatie dupa Watford. [Conferinta Norwich, 2 noiembrie](https://www.canaries.co.uk/news/press-conference-clement-im-looking-forward-game-its-going-be-hard-battle). Ozoh/Ward/Meister sunt incerti, Clarke/Agyemang au reluat antrenamentul; sursele nu concorda asupra absentei certe Ozoh, deci o las incertitudine. [Relatarea actualizata](https://staging-aws.sports.yahoo.com/articles/derby-county-duo-closing-returns-170400754.html). Formatiile anterioare, nu confirmate azi, sunt in ultimele 10 de mai jos.'
Add
Add '**Sirius-Halmstad, 20:00:** 90,52% / 84,81%; 46/51 istoric distinct si 25/26 pe teren, esantion mai mare. Sirius poate reveni pe primul loc, iar Halmstad este ultima, cu 17 puncte dupa 26 jocuri. [Clasament contemporan](https://en.wikipedia.org/wiki/2026_Allsvenskan), [preview Sirius](https://www.siriusfotboll.se/nyheter/infor-ik-sirius-halmstads-bk-5/). Antrenorii sunt Andreas Engelmark/Stuart Baxter; Soumah este suspendat, Halmstad nu are suspendati. Ultimele dispozitive publicate sunt 4-2-3-1/4-4-2, nu formule confirmate. Ultimul duel a avut primul gol la 14, dar cel din august 2025 abia la 76; H2H nu este garantie. [Ghid oficial HBK](https://www.hbk.se/nyheter/2026/november/matchguide-ik-sirius-hbk). Liimatta/Uneken au situatii medicale insuficient clarificate; nu afirm ca sunt recuperati. [Relatare, 24 octombrie](https://bollsvenskan.se/engelmark-oppnar-for-rutingenomlysning-i-sirius-uneken-missar-aik-och-hammarby-siktar-pa-halmstad-3-november/). Sirius are 5 zile de la ultimul joc, Halmstad 6; teren artificial, [confirmare AIK](https://www.aikfotboll.se/artiklar-och-nyheter/261024-bortareseinformation-ik-sirius-fk-aik-herr). [Prognoza publicata pe 2 noiembrie](https://www.weather2travel.com/sweden/uppsala/forecast/) nu indica ploaie semnificativa. Principalul motiv pentru a ramane alternativa este atacul Halmstad: 5 GF70/13 deplasari = 0,38, cu dependenta crescuta de Sirius. Cota echitabila 1,1048 nu este dovada ca piata ofera 1,10 sau ca exista avantaj la acel pret.'
Add
Add 'Alte limite: Lyngby-Randers este proiectie de campionat intr-o cupa si revine dupa duelul direct din 1 noiembrie, deci rotatia poate fi substantiala; nu am validat contextual formatul exact al turului, nu il recomand. Shakhtar-Sporting se joaca la Stamford Bridge, Londra, nu in mediul obisnuit al gazdelor: statisticile domestice acasa se transfera si mai slab. [Confirmare Shakhtar, 2 noiembrie](https://shakhtar.com/en/news/2026/november/2_news/2_fc-shakhtar-training-session/).'
Add
Add 'La Sirius-Halmstad exista si un contraargument recent: ultimele 10 oficiale dau 9/10 pentru Sirius, dar numai 7/10 pentru Halmstad. Cele trei esecuri recente Halmstad sunt deschideri la 76, 90+1 si 90+3, nu partide 0-0. Sunt incluse integral mai jos; frecventa pe tot sezonul ascunde aceasta concentrare recenta. Nu adaug o penalizare procentuala improvizata, dar este un motiv concret pentru a nu promova aceasta alternativa in fata QPR.'
Add
Add '## Ultimele 10 partide oficiale verificate ale finalistelor'
Add
Add 'Am folosit inventarul tuturor competitiilor din API-ul echipelor FotMob, am exclus amicalele, apoi am verificat evenimentele si scorul fiecarei partide. Sezonul de model ramane numai campionatul intern. Competitia Cup din sursa este cupa nationala suedeza. Toate partidele de mai jos sunt anterioare zilei analizei. Un 0-0 are minut NULL, nu minut 0.'
Add
foreach($team in @('qpr','burnley','derby','norwich','sirius','halmstad')){
 $data=@(Get-Content (Join-Path $root ('Finalists/'+$team+'_last10.json')) -Raw -Encoding UTF8|ConvertFrom-Json|ForEach-Object {$_})
 if($data.Count -ne 10){throw ('Not 10 matches '+$team)}
 Add ('### '+$team)
 Add
 Add '| Data | Competitie | Meci, scor gazda-oaspete | Primul gol | Gol70 |'
 Add '|---|---|---|---:|---|'
 foreach($r in $data){Add ('| '+([DateTimeOffset]::Parse($r.date)).ToString('yyyy-MM-dd')+' | '+$r.competition+' | ['+$r.home+' - '+$r.away+' '+$r.score+']('+$r.source+') | '+$(if($null -eq $r.first_goal -or $r.first_goal -eq ''){'fara gol'}else{$r.first_goal+"'"})+' | '+$(if($r.success70){'DA'}else{'NU'})+' |')}
 Add
 Add ('Bilant: '+@($data|Where-Object success70 -eq $true).Count+'/10. Formatiile, evenimentele, eliminarile si sursele individuale sunt pastrate in Finalists/'+$team+'_last10.json.')
 Add
}
Add '## Verdict, risc si evidenta prospectiva'
Add
Add 'Aleg o singura selectie: **QPR-Burnley, GOAL_BY_70**. Argumentul decisiv este maximul modelului intre partidele cu istoric din aceeasi competitie, sustinut de productia ofensiva pe teren a ambelor echipe si de sensibilitatile fara eliminari/fara 5-0. Principalul risc este ca Burnley sub Celestini sa fie structural mai prudenta decat esantionul vechi, concomitent cu absentele creative QPR. 0-0 cu Portsmouth arata deja ca acest scenariu exista. Nu scad sau adaug procente arbitrare pentru el.'
Add
Add 'Modelul brut nu demonstreaza ca QPR-Burnley are o probabilitate reala mai mare decat Derby-Norwich; frecventele favorizeaza Derby si sunt o contradictie de raportat. Selectia urmeaza regula de clasare declarata, nu un scor contextual inventat. Daca s-ar cere o probabilitate validata de 90% sau o oferta de peste 1,10 confirmata, datele de aici nu o demonstreaza. Este o predictie prospectiva pentru validare, fara recomandare de miza.'
Add
Add 'Dovezi: `all_models.csv`, `comparison_verified.csv`, `universe_coverage.csv`, `exclusions.csv`, `Data/historical_validated.csv`, `Data/historical_rejected.csv`, `independent_provider_checks.csv`, `sensitivity.csv`, `live_status.json`, `verification.json`. Scripturi locale: `rank_today.ps1`, `verify_today.ps1`, `build_report.ps1`; colectoarele acceseaza doar endpointuri publice si salveaza campurile necesare, fara cote. Modelele de productie, calibrarile si selection_engine nu sunt modificate.'
$l|Set-Content (Join-Path $root 'EUROPE_ALL_MATCHES.md') -Encoding UTF8
$comparison|Export-Csv (Join-Path $root 'prospective_selections.csv') -NoTypeInformation -Encoding UTF8
$historyRoot=Join-Path (Split-Path $root -Parent) 'Selection_History'
$comparison|Export-Csv (Join-Path $historyRoot '2026-11-03_goal70_candidates.csv') -NoTypeInformation -Encoding UTF8
$journal=@(
 '# Istoric prospectiv: 3 noiembrie 2026',
 '',
 ('Salvat: '+$stamp+'. Piata GOAL_BY_70, orice echipa, minut afisat <=70 inclusiv.'),
 '',
 'Recomandarea individuala a analistului: QPR-Burnley, 21:45 RO; p brut 93,6276708%, stres 88,9480588%, cota echitabila 1,0680603. Rang brut 2/20, primul intre meciurile cu istoric in aceeasi competitie.',
 '',
 'Utilizatorul nu a declarat inca alegerea pentru azi. Nu s-a inregistrat pariu plasat, cota disponibila, miza, rezultat sau profit. Protocolul ramane maximum o selectie individuala pe zi intre cele doua piete; NO BET este permis.',
 '',
 'Lista completa: [20 candidati](2026-11-03_goal70_candidates.csv). Acestea sunt comparatii, nu 20 recomandari. [Analiza, minute, calcule, excluderi si riscuri](../Goal70_2026-11-03/EUROPE_ALL_MATCHES.md).',
 '',
 'Shortlist cercetat: QPR-Burnley, Birmingham-Millwall, Derby-Norwich, Sirius-Halmstad; verificare suplimentara Feyenoord-Inter. Diferentele de profunzime sunt explicit marcate in analiza.',
 '',
 'Rezultatele raman necompletate pana la verificarea ulterioara. Nu rescrie estimarile originale dupa meci.'
)
$journal|Set-Content (Join-Path $historyRoot '2026-11-03.md') -Encoding UTF8
Write-Output ('Saved full report, coverage, exclusions and prospective journal at '+$stamp)
