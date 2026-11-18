# CORNERS_1H_UNDER_7_5 — 18 noiembrie 2026

**NO BET, fără candidat.** Niciun model eligibil în cele 22 de partide identificate. În cele 273 de istorice sunt 138 de observații FirstHalf lipsă sau nereconciliate. La meciurile din Cupa Spaniei lipsesc în special datele adversarilor din ligile inferioare. UWCL are 4 general / 2 teren în ediția curentă, sub minime; cele nouă naționale sunt deja încheiate. Nu deducem cornerele FirstHalf din FT sau din scorul pauzei.

[Numărători, dispersie și toate excluderile](CALCULATIONS.md), [inventar](../Goal70_2026-11-18/INVENTORY.md), [surse](../Goal70_2026-11-18/sources.json).

## Lectii aplicate din review

Predarea review-ului și lecțiilor: **2026-11-18T09:10:57.650+02:00**. Începutul scanării: **2026-11-18T09:11:14.341+02:00**. Ordinea este verificată.

|Lecție|Verificare și dovadă nouă|Statut / efect asupra preferinței|
|---|---|---|
|Model versus frecvența evenimentului|Portugalete 84,61% versus 17/23 general și 8/12 teren; toate cele șase modele comparate|Aplicată: rangul 1 rămâne screening, fără recomandare fermă|
|Atac propriu, 0–0 și prime goluri tardive|GF70 pe teren și toate tardivele în calcule; Badajoz are patru 0–0/tardive și numai 6/10 reușite recente|Aplicată: nu promovăm alternativa numai pentru 11/12 pe teren|
|Sensibilitate la scoruri ample și AR20|Badajoz–Tenerife scade de la 83,45% la 81,53% în diagnosticul fără diferențe FT≥4; toate scenariile păstrate|Aplicată: diagnostic separat, baseline neschimbat|
|Campionat versus cupă|Șase proiecții între campionate și prima rundă a cupei|Aplicată: incertitudine de transfer, fără ajustare procentuală arbitrară|
|FirstHalf, exact 7 / ≥8 și dispersie|138 FirstHalf lipsă; numărătorile și cozile observate sunt păstrate în CSV|Aplicată: nu deducem din FT; dispersia bilaterală incompletă susține NO BET la cornere|
|90 minute versus prelungiri|Irlanda 2–1 la 90 / 2–2 AET; primul 62, golul din prelungiri exclus|Aplicată în review; istoricele nereconciliate rămân lipsă|
|Loturi, antrenori, program și vreme|Bilbao, Cervera și Vélez confirmați în surse oficiale; 4/4 sau 3/3 zile calendaristice; rafale prognozate 60,1 km/h la Badajoz|Parțial aplicată: XI, loturi bilaterale și tactica recentă rămân neconfirmate; fără preferință fermă|
|Categorii, ediție, teren și status|UWCL 4/2 sub minime; nouă naționale FT; șase modele de cupă reconfirmate neîncepute|Aplicată: fără pooling și fără selecții retrospective; juniori/feminin mondial incomplet|

Auditul și protocolul challengerilor din 17 noiembrie rămân cercetare izolată. Nu au schimbat formula, ponderile, clasamentul brut sau modelele de producție.

## Acoperire și limite

**Acoperire parțială**, actualizare 2026-11-18T09:25:16.308+02:00. Sunt 22 de partide identificate: nouă între naționale masculine de seniori, deja încheiate în noaptea României; șase în Cupa Spaniei; trei în Cupa Sloveniei; patru în Liga Campionilor la cluburi feminine. Nu este un top global complet.

Am verificat cele 29 de ligi convenite: primele divizii masculine din top 20 asociații UEFA conform listei finale 2026 utilizate în proiect, România și Serbia, plus secundele din Anglia, Germania, Spania, Italia, Franța, Belgia și Turcia. Nu am identificat partide pentru 18 noiembrie în aceste ligi. FotMob a furnizat calendare complete de sezon; ESPN a coroborat 25 de ligi. Pentru Polonia, Croația, Serbia și Belgia 2 lipsește confirmarea independentă. Pagina UEFA accesată dinamic nu a reprodus tabelul clasamentului; lista explicită și anul sunt păstrate în universe_coverage.csv. Un endpoint indisponibil nu dovedește absența meciurilor.

Pentru cupe am căutat suplimentar campionatele curente ale echipelor: Segunda Federación, Primera Federación și Prva Liga. Sunt 273 de observații istorice distincte, cu patru liste de goluri și 138 de observații FirstHalf lipsă sau nereconciliate. Cele șase modele Gol70 folosesc istorice complete bilateral și respectă minimele 5 general / 3 teren; au fost calculate toate înainte de decizie. Niciun model la cornere nu este eligibil. Proiecțiile din campionat în Cupa Spaniei sunt declarate separat; nu constituie validarea transferului între competiții.

Naționalele folosesc ediția curentă, iar amicalele numai anul 2026, separat. Seniori, juniori, masculin și feminin nu sunt combinați. Cele nouă naționale deja încheiate nu primesc predicții retrospective. Cele patru partide UWCL au câte 4 meciuri general / 2 teren în ediția curentă, sub minime; nu am completat cu date de la echipe masculine sau naționale. Terenul din feed este desemnat; pentru partidele fără candidat ferm neutralitatea nu a fost confirmată exhaustiv.

Programul a fost cercetat prin FotMob global pentru 17/18/19 UTC, filtrat numeric la data României, calendarele ligilor, ESPN, Footynomad, FootballMania și căutări federative. Footynomad a furnizat 181 de rânduri mondiale, inclusiv cluburi și partide din afara domeniului. Paginile UEFA WU17/WU19 nu au furnizat liste dinamice utilizabile: nu demonstrează absența naționalelor feminine. Nu există o partidă feminină sau de juniori între naționale confirmată pentru 18 RO în setul verificat; acoperirea mondială rămâne incompletă.

[FIFA](https://www.fifa.com/en/tournaments/mens/u17worldcup/qatar-2026/articles/score-fixtures-match-schedule) fixează începutul CM U17 pe 19 noiembrie. Semnalarea Olanda U19–Irlanda U19 are un conflict 2025/2026 și adversari diferiți în ediția curentă: nu este un meci confirmat azi. Brazilia U20–Peru U20 apare la 16 noiembrie 23 UTC / 17 RO în coroborare, în conflict cu eticheta SNA; US U20–Paraguay este confirmat de calendarul Eintracht pentru 17 noiembrie 17 UTC. Western Springs–Tonga este club contra națională, în afara domeniului națională contra națională. Aceste semnalări nu sunt transformate în partide eligibile pentru azi.

Reviziile inițiale sunt păstrate în Revisions/before_extension și before_identity_reconciliation. Șapte meciuri ale cluburilor U19 ucrainene au fost eliminate din suplimentul de naționale, cu urmă de audit. Endpointul 9137 s-a dovedit China League One și a fost eliminat din domeniul cupelor; identitățile spaniole 9138/8968 au fost confirmate prin API-urile echipelor. Nicio probabilitate publicată anterior nu a fost înlocuită. URL-urile și momentele extragerii sunt păstrate lângă datele brute și în sources.json.
