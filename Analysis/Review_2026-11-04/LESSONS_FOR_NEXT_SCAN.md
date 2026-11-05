# Lectii pentru scanarea urmatoare

Predare salvata 2026-11-05T08:02:53.746+02:00, inainte de orice inventar5 noiembrie. Review: [REVIEW.md](REVIEW.md), goal70_outcomes.csv, corners1h_outcomes.csv, FotMob/. Surse prospective: Goal70_2026-11-04/EUROPE_ALL_MATCHES.md, Corners1H_2026-11-04/ANALYSIS.md si Selection_History/2026-11-04_*_candidates.csv. Lectii cumulative: Review_2026-11-02/REVIEW.md si Review_2026-11-03/REVIEW.md.

| Statut | Lectie/evidenta | Verificare concreta pentru azi | Limita/test ramas |
| --- | --- | --- | --- |
| constatare | Modelul Poisson si frecventa observata sunt marimi distincte; pe3 noiembrie Watford-Lincoln avea87,03% versus20/26 istoric, apoi10 cornere | Afiseaza brut, AR20%, frecventa si numitor sezon/teren pentru fiecare finalist | Calibrarea necesita mai multe zile cronologice |
| constatare | Proiectiile nationale nu reproduc automat Europa/cupa; Leipzig-PSV75 era deja contraexemplu prospectiv | Separa intern/Europa/cupa, ultimele10 oficiale si campionatul folosit in formula | Forta adversarilor si transferul intre competitii netestate |
| constatare | Primul gol tardiv/0-0 pierde Gol70 chiar daca FT are goluri; Lyngby90+3 pe3 noiembrie | Raporteaza GF70 propriu pe teren, toate deschiderile >70/0-0, sensibilitati numai-sezon/teren si eliminari | Nu atribui cauze tactice din scor |
| constatare | Exact7 este succes aproape de limita; >=8 esec. Dispersia poate contrazice Poisson | Afiseaza seria FirstHalf, exact7/>=8, medie/varianta, lipsuri distincte si reconciliere FirstHalf+SecondHalf=All | Alternativa de distributie necesita backtest separat autorizat |
| constatare | Schimbarile de antrenor/lot invalideaza descrierile vechi: West Ham, Kortrijk, Mechelen | Cauta surse actuale pentru antrenori, absente, loturi probabile/confirmate, miza, odihna si vreme; pastreaza contradictiile | Fara ajustari procentuale arbitrare; informatia neverificata ramane explicita |
| ipoteza | Frecventa primului gol si adversarii pot descrie alte riscuri decat suma golurilor | Compara modelul cu evenimentul binar si contraexemplele recente | De testat cronologic, fara tuning dupa o zi |
| lipsa date | Cauzele efective ale esecurilor si unele rezultate restante nu sunt documentate complet | Pastreaza pending/void/amanat si reincearca ulterior | Lipsa nu este zero si nu opreste scanarea dupa predarea partiala |

Nu se promit eliminarea esecurilor, profit sau rata90%. Snapshoturile originale, ponderile50/50 si productia raman nemodificate.

Constatari suplimentare4 noiembrie: Rennes-Olympiacos0-0, singurul failGol70 (rang13, brut84,71%, AR20%77,74%); Fenerbahce-Liverpool8 cornere1H, singurul failcornere (rang3, brut89,11%). Ambele proiectii europene. Intern si Europa trebuie separate, fara a declara o cauza sau regula noua din aceste doua observatii. Kortrijk-Mechelen are FirstHalf0-0 cornere reale, SecondHalf2-4 si All2-4; zero este observatie reconciliata, nu completare a unei lipse.
Tenerife-Andorra: furnizorul s-a schimbat acum laFT2-2, corner1H2; exista relatare4 noiembrie despre decizia comisiei de a incheia partida. Arhiva ramane momentan void pana la reconcilierea explicita; nu mai afirmam ca abandonul este singura informatie actuala.


## Reconciliere5 noiembrie

2026-11-05T08:03:59.214+02:00: LaLiga afiseaza Finalizado2-2; raportulAS4 noiembrie17:36CET descrie hotararea comisiei de a incheia partida, fara reluare. FotMob confirma acumfinished=true, cancelled=false, primul gol3 si FirstHalf1-1=2, reconciliat cuSecondHalf siAll. Rezultatele Tenerife-Andorra devin success pe ambele piete, observatii sportive; nu este decontare de bookmaker. Vechile stari amanat/abandonat sunt arhivate inRevisions/2026-11-05_before_official_conclusion, probabilitatile prospective raman intacte. [LaLiga](https://www.laliga.com/es-NG/partido/temporada-2026-2027-laliga-hypermotion-cd-tenerife-fc-andorra-12), [decizie relatataAS](https://as.com/futbol/segunda/el-tenerife-andorra-se-da-por-finalizado-no-se-tendra-que-reanudar-f202611-n/). Bilant actualizat2 noiembrie: Gol7022/26, cornere22/26; Radnicki-Zeleznicar ramane reprogramat25 noiembrie, oraTBD, exclus.
