# Acoperire — 14 noiembrie 2026

## Acoperire și limite

Inventarul are **164 de înregistrări**, dintre care 38 cu identitate FotMob și 126 de suplimente: 51 de partide din Coupe de France, 59 de calificări U17/U19 și 16 alte semnalări. Acesta nu este un total de 164 de partide confirmate independent: există un conflict England U18–Norway U19 versus England U18–Sweden U18. Istoricul exact și minimele 5/3 nu au fost validate pentru suplimente. Nu substituim alte generații, cluburi sau ediții vechi.

Sunt verificate 29 de ligi: primele 20 de asociații din [clasamentul final UEFA 2026](https://kassiesa.net/uefa/data/method5/crank2026.html), calculat pe 2021/22–2025/26, plus România, Serbia și cele șapte secunde urmărite. În 28 de ligi nu au fost identificate partide; în Segunda sunt cinci. Am confruntat programul global FotMob, calendarele edițiilor actuale și ESPN. Patru ligi nu au confirmare ESPN: Polonia, Croația, Serbia și secunda belgiană. Un răspuns nesuportat nu dovedește absența meciurilor.

Separat sunt trei partide de cupă din Slovacia/Croația și cele 51 din Coupe de France. Nu au istoric bilateral validat pentru proiecția de campionat. Calendarele UCL/UEL/UECL și Copa del Rey consultate nu indică partide azi. Lista completă, sursele și excluderile sunt în [INVENTORY.md](../Goal70_2026-11-14/INVENTORY.md), universe_coverage.csv și candidate_fixtures.json.

Am colectat **304 istorice unice** în edițiile actuale, cu zece liste de goluri și 36 de observații FirstHalf lipsă sau nereconciliate. Suplimentele fără istoric nu intră în acest numitor. Sunt calculate și comparate toate cele **șase modele Gol70 și cinci modele de cornere eligibile în datele validate**. Modelarea întregului inventar și confirmarea independentă mondială rămân parțiale; nu prezentăm un top global complet.

Cele opt partide UEFA Nations League au numai 4 generale/2 pe teren pentru fiecare echipă, sub 5/3. Edițiile AFCON/CONCACAF sunt și ele limitate. Amicalele folosesc numai anul 2026, separat de competițiile oficiale. Identitățile seniori/juniori/masculin/feminin sunt distincte. Meciurile începute sau terminate în dimineața RO sunt inventariate fără predicții retrospective și fără rezultate fabricate azi.

Două listări duble Friendly/King’s Cup, Oman–Tajikistan și Thailand–Brunei, sunt deduplicate și excluse până la clarificarea competiției. Basque Country–Venezuela este o selecție regională/non-FIFA, distinctă și fără model. Haiti–USA necesită confirmarea gazdei desemnate și a terenului; nu este finalist.

Japan–Brazil este **neutru, la Singapore**, cu Japonia gazdă desemnată. Baseline-ul 87,13/80,60% folosește părțile desemnate ale amicalelor 2026, fără a pretinde avantaj propriu real sau calibrare pe neutru. Frecvențele sunt 11/14 general și 6/8 pe părțile desemnate. Nu este candidat preferat.

[AUF confirmă Uruguay U17 feminin–Peru U17 feminin](https://www.auf.org.uy/la-sub-17-femenina-disputara-dos-amistosos-ante-peru/) la 10:00 local = 15:00 RO, pe teren propriu în Uruguay; istoricul exact al amicalelor 2026 nu este validat. Nu a fost identificată o partidă de naționale feminine de seniori, fără a demonstra absența mondială. Panama–El Salvador și Costa Rica–Guatemala U19 feminin, afișate 14 nov 04:00 în UTC+8, aparțin zilei 13 nov, 22:00 RO, conform surselor secundare UTC.

Snapshotul numeric inițial și reviziile sunt arhivate în Revisions/. Procentele, rangurile originale și momentele inițiale ale predicțiilor numerice sunt păstrate. Nicio alegere nouă a utilizatorului nu a fost creată.


## Corecție de cutoff și control final

La 2026-11-14T09:08:51.541+02:00, controlul fusului a exclus din istoricul anterior zilei Argentina–Uruguay și Saint Martin–El Salvador, ambele13nov23UTC=14nov01RO. Rămân în inventarul de azi, deja începute și fără predicții. Sunt304istorice utilizate din306cereri raw; niciunul dintre cele11modele nu folosea aceste două observații. Comparația se face pe timestampuri interpretate, nu textual între formate ISO cu offset diferit. [Dovadă și copii](cutoff_correction.json). Modelele, procentele și timestampurile prospective sunt neschimbate.

Toate înregistrările fără observații/minime/status validate sunt **NO BET formal**, cu motiv individual. Lipsa nu este zero.
