# CORNERS_1H_UNDER_7_5 — 17 noiembrie 2026

**NO BET.** Niciun model eligibil în cele 133 de înregistrări evaluate. Lipsesc eșantioane FirstHalf complete bilateral, minimum 5 general și 3 în configurația terenului. Sunt 126 de observații FirstHalf lipsă/neconciliate în cele 361 de istorice, iar 75 de suplimente nu au istoric exact bilateral validat.

Scorul la pauză și cornerele FT nu substituie FirstHalf. Nu publicăm probabilitate sau cotă matematică fără date. Candidatul Gol70 Irlanda–Finlanda nu devine candidat la cornere.

[Calcule și lipsuri](CALCULATIONS.md), [toate excluderile](../Goal70_2026-11-17/INVENTORY.md), [context](../Goal70_2026-11-17/CONTEXT.md).

Cercetare **parțială**: 133 de înregistrări, dintre care 58 au identitate numerică FotMob și 75 sunt suplimentare. Zece suplimente sunt semnalări cu dată/oră/status neconfirmate. Numărul de rânduri nu este un număr de partide independent confirmate. Toate cele cinci partide eligibile Gol70 au fost calculate; la cornere nu există model eligibil. Au fost colectate 361 de partide istorice: 16 liste de goluri și 126 observații FirstHalf lipsesc sau nu se reconciliază. Lipsa nu este zero.

Au fost verificate 29 de ligi: primele ligi din top 20 al [clasamentului final UEFA 2026](https://kassiesa.net/uefa/data/method5/crank2026.html), România, Serbia și diviziile secunde urmărite din Anglia, Germania, Spania, Italia, Franța, Belgia și Turcia. Feedul global și calendarele de sezon nu identifică partide azi în aceste ligi. Pentru 25 există și răspuns ESPN utilizabil; Polonia, Croația, Serbia și Belgia 2 păstrează limita confirmării independente. Cupele europene, FA Cup, Coupe de France, Copa del Rey și League Cup urmărite nu au partide identificate azi. EFL Trophy și celelalte competiții din feedul mondial sunt păstrate în fișierele brute, în afara setului de cupe urmărite de acest baseline.

Naționalele de seniori și juniori, masculin și feminin, rămân distincte. Amicalele folosesc numai anul 2026, separat de oficiale; calificările folosesc ediția curentă. Cele opt meciuri UEFA Nations League au 5 observații generale, dar numai 2 în configurația terenului pentru fiecare echipă: sub minimul 3. Nu se adaugă ediții vechi. Pentru suplimentele U17/U19 nu este validat istoricul exact bilateral; cele două meciuri ale mini-turneului nu sunt declarate automat întreaga ediție.

La feminin au fost făcute căutări separate și consultate paginile UEFA WU17/WU19. Paginile dinamice fără rezultate nu demonstrează absența partidelor. Nicio partidă feminină de naționale nu este confirmată independent în această etapă; acoperirea mondială rămâne incompletă. Conflictele de oră și gazdă desemnată sunt păstrate în [crosswalk](Supplemental/crosswalk.json). Japonia–Paraguay este pe teren neutru la Singapore; alte neutralități neconfirmate sunt declarate ca atare.

Inventarul și istoricul suplimentelor sunt parțiale. Nu prezentăm un top global complet. [Inventar și excluderi](INVENTORY.md), [calendare și limite pe ligă](universe_coverage.csv), [lipsuri suplimentare](Supplemental/coverage_gaps.json), [istoric validat](historical_observations.csv).

## Lectii aplicate din review

- **Aplicată — model față de frecvență.** Grecia U21 a ratat ieri cu primul gol la 72, după un model de 91,60% și istoric 18/21. Pentru Irlanda afișăm separat 95,02% brut, 90,93% stres, 19/21 general, 10/10 teren și 9/10 în ultimele oficiale pentru fiecare echipă. Diferența nu este ascunsă și nu produce o corecție arbitrară.
- **Aplicată — goluri proprii și deschideri tardive.** Irlanda are GF70 propriu 10/5 acasă, Finlanda 15/5 în deplasare. Kazakhstan–Irlanda, primul 89, și Finlanda–Kosovo 0–0 rămân în istoric. Fără diferențe finale de cel puțin patru goluri, diagnosticul scade la 91,06%; baseline-ul rămâne 95,02%.
- **Aplicată — contextul barajului și lotul actual.** Turul 0–1 permite Finlandei să se califice cu egal. Sunt verificate antrenorii, înlocuirile Orazi/Cummins și Talvitie/Laine, programul și ploaia prognozată la Cork. XI, minutele individuale și gazonul efectiv rămân neverificate.
- **Aplicată — cornere exacte și dispersie.** Succesul Albacete cu 4 cornere ieri nu validează distribuția Poisson și nu anulează riscul prospectiv. Azi nu există eșantion bilateral complet 5/3; alegerea este NO BET. Totalurile exact 7 și >=8 disponibile rămân în istoricul și diagnosticele salvate.
- **Aplicată — ediții și cohorte.** Nations League 5/2 rămâne neeligibilă. Amicalele 2026 și calificările curente nu sunt amestecate; terenul neutru este declarat. **Neverificate integral:** acoperirea mondială la feminin, istoricul suplimentelor și auditul eliminărilor. Nu inventăm rezultate, cauze tactice sau probabilități.

[Lecțiile predate înaintea scanării](../Review_2026-11-16/LESSONS_FOR_NEXT_SCAN.md). Timestampurile canonice din handoff și research_start păstrează ordinea. Nu s-au schimbat formulele, ponderile, pragurile sau modelele de producție.
