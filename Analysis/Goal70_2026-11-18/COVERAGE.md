# Acoperire — 18 noiembrie 2026

**Acoperire parțială**, actualizare 2026-11-18T09:25:16.308+02:00. Sunt 22 de partide identificate: nouă între naționale masculine de seniori, deja încheiate în noaptea României; șase în Cupa Spaniei; trei în Cupa Sloveniei; patru în Liga Campionilor la cluburi feminine. Nu este un top global complet.

Am verificat cele 29 de ligi convenite: primele divizii masculine din top 20 asociații UEFA conform listei finale 2026 utilizate în proiect, România și Serbia, plus secundele din Anglia, Germania, Spania, Italia, Franța, Belgia și Turcia. Nu am identificat partide pentru 18 noiembrie în aceste ligi. FotMob a furnizat calendare complete de sezon; ESPN a coroborat 25 de ligi. Pentru Polonia, Croația, Serbia și Belgia 2 lipsește confirmarea independentă. Pagina UEFA accesată dinamic nu a reprodus tabelul clasamentului; lista explicită și anul sunt păstrate în universe_coverage.csv. Un endpoint indisponibil nu dovedește absența meciurilor.

Pentru cupe am căutat suplimentar campionatele curente ale echipelor: Segunda Federación, Primera Federación și Prva Liga. Sunt 273 de observații istorice distincte, cu patru liste de goluri și 138 de observații FirstHalf lipsă sau nereconciliate. Cele șase modele Gol70 folosesc istorice complete bilateral și respectă minimele 5 general / 3 teren; au fost calculate toate înainte de decizie. Niciun model la cornere nu este eligibil. Proiecțiile din campionat în Cupa Spaniei sunt declarate separat; nu constituie validarea transferului între competiții.

Naționalele folosesc ediția curentă, iar amicalele numai anul 2026, separat. Seniori, juniori, masculin și feminin nu sunt combinați. Cele nouă naționale deja încheiate nu primesc predicții retrospective. Cele patru partide UWCL au câte 4 meciuri general / 2 teren în ediția curentă, sub minime; nu am completat cu date de la echipe masculine sau naționale. Terenul din feed este desemnat; pentru partidele fără candidat ferm neutralitatea nu a fost confirmată exhaustiv.

Programul a fost cercetat prin FotMob global pentru 17/18/19 UTC, filtrat numeric la data României, calendarele ligilor, ESPN, Footynomad, FootballMania și căutări federative. Footynomad a furnizat 181 de rânduri mondiale, inclusiv cluburi și partide din afara domeniului. Paginile UEFA WU17/WU19 nu au furnizat liste dinamice utilizabile: nu demonstrează absența naționalelor feminine. Nu există o partidă feminină sau de juniori între naționale confirmată pentru 18 RO în setul verificat; acoperirea mondială rămâne incompletă.

[FIFA](https://www.fifa.com/en/tournaments/mens/u17worldcup/qatar-2026/articles/score-fixtures-match-schedule) fixează începutul CM U17 pe 19 noiembrie. Semnalarea Olanda U19–Irlanda U19 are un conflict 2025/2026 și adversari diferiți în ediția curentă: nu este un meci confirmat azi. Brazilia U20–Peru U20 apare la 16 noiembrie 23 UTC / 17 RO în coroborare, în conflict cu eticheta SNA; US U20–Paraguay este confirmat de calendarul Eintracht pentru 17 noiembrie 17 UTC. Western Springs–Tonga este club contra națională, în afara domeniului națională contra națională. Aceste semnalări nu sunt transformate în partide eligibile pentru azi.

Reviziile inițiale sunt păstrate în Revisions/before_extension și before_identity_reconciliation. Șapte meciuri ale cluburilor U19 ucrainene au fost eliminate din suplimentul de naționale, cu urmă de audit. Endpointul 9137 s-a dovedit China League One și a fost eliminat din domeniul cupelor; identitățile spaniole 9138/8968 au fost confirmate prin API-urile echipelor. Nicio probabilitate publicată anterior nu a fost înlocuită. URL-urile și momentele extragerii sunt păstrate lângă datele brute și în sources.json.

|Ligă / țară|RangUEFA2026|Global|Sezon|ESPN|Statut|
|---|---|---|---|---|---|
|eng.1/England|1|0|0|0|FotMob_global+season+ESPN|
|ita.1/Italy|2|0|0|0|FotMob_global+season+ESPN|
|esp.1/Spain|3|0|0|0|FotMob_global+season+ESPN|
|ger.1/Germany|4|0|0|0|FotMob_global+season+ESPN|
|fra.1/France|5|0|0|0|FotMob_global+season+ESPN|
|por.1/Portugal|6|0|0|0|FotMob_global+season+ESPN|
|ned.1/Netherlands|7|0|0|0|FotMob_global+season+ESPN|
|bel.1/Belgium|8|0|0|0|FotMob_global+season+ESPN|
|tur.1/Turkey|9|0|0|0|FotMob_global+season+ESPN|
|cze.1/Czechia|10|0|0|0|FotMob_global+season+ESPN|
|gre.1/Greece|11|0|0|0|FotMob_global+season+ESPN|
|pol.1/Poland|12|0|0|indisponibil|FotMob_global+season;independent_missing|
|den.1/Denmark|13|0|0|0|FotMob_global+season+ESPN|
|nor.1/Norway|14|0|0|0|FotMob_global+season+ESPN|
|cyp.1/Cyprus|15|0|0|0|FotMob_global+season+ESPN|
|sui.1/Switzerland|16|0|0|0|FotMob_global+season+ESPN|
|aut.1/Austria|17|0|0|0|FotMob_global+season+ESPN|
|sco.1/Scotland|18|0|0|0|FotMob_global+season+ESPN|
|swe.1/Sweden|19|0|0|0|FotMob_global+season+ESPN|
|cro.1/Croatia|20|0|0|indisponibil|FotMob_global+season;independent_missing|
|rou.1/Romania|supplement|0|0|0|FotMob_global+season+ESPN|
|srb.1/Serbia|supplement|0|0|indisponibil|FotMob_global+season;independent_missing|
|eng.2/England 2|supplement|0|0|0|FotMob_global+season+ESPN|
|ger.2/Germany 2|supplement|0|0|0|FotMob_global+season+ESPN|
|esp.2/Spain 2|supplement|0|0|0|FotMob_global+season+ESPN|
|ita.2/Italy 2|supplement|0|0|0|FotMob_global+season+ESPN|
|fra.2/France 2|supplement|0|0|0|FotMob_global+season+ESPN|
|bel.2/Belgium 2|supplement|0|0|indisponibil|FotMob_global+season;independent_missing|
|tur.2/Turkey 2|supplement|0|0|0|FotMob_global+season+ESPN|