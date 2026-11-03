# Banca 100 lei, prima miza 100 lei: scenarii matematice

Data: 2026-11-01. Analiza foloseste numai rata retrospectiva CORNERS_1H_UNDER_7_5,
367/413 = 88,861985%, nu o probabilitate viitoare validata si nici rata GOAL_BY_70.
Cotele de mai jos sunt ipoteze numerice, nu cote de piata consultate.
Sumele sunt brute, fara taxe, comisioane sau restrictii de retragere.

## Garantia ceruta nu este posibila

Capital initial 100 lei, prima miza 100 lei: daca primul pariu pierde,
soldul devine zero. O rata istorica apropiata de 90% nu exclude acest rezultat.
Nicio schema ulterioara a mizelor nu poate impiedica pierderea deja produsa.
Reinvestirea intregului sold aduce banca la zero la prima pierdere.

Noua victorii si o pierdere la mize constante de 100 lei ar produce:

| Cota ipotetica | Profit din 9 victorii | Pierdere din 1 esec | Rezultat net |
|---|---:|---:|---:|
| 1,10 | 90 lei | 100 lei | -10 lei |
| 1,12 | 108 lei | 100 lei | +8 lei |
| 1,15 | 135 lei | 100 lei | +35 lei |

Acest tabel presupune ca toate cele zece mize pot fi finantate. O banca initiala
de 100 lei nu permite asta daca pierderea apare prima. Ordinea rezultatelor
conteaza pentru posibilitatea de a continua.

Cu p=0,90, pragul matematic fara costuri este 1/p = 1,1111.
Cu frecventa observata p=367/413, pragul ar fi 413/367 = 1,12534, numai daca
aceasta frecventa ar reprezenta probabilitatea reala viitoare.
La cota 1,10, valoarea asteptata pe 100 lei este
100 * ((367/413)*1,10 - 1) = -2,2518 lei.

La miza integrala reinvestita si presupunand independent aceeasi probabilitate
p=367/413 la fiecare pariu, sansa de a traversa zece pariuri fara pierdere
este p^10 = 30,7015%. Complementul, 69,2985%, este sansa de cel putin o
pierdere si, deci, de epuizare a bancii in acest scenariu simplificat.
Independenta si p constant sunt ipoteze, nu proprietati demonstrate ale datelor.

## Ce arata ordinea rezultatelor existente

Sirul celor 413 prime alegeri contine doua pierderi consecutive. In cele
404 ferestre suprapuse de cate zece selectii:

| Pierderi in fereastra | Numar de ferestre |
|---|---:|
| 0 | 118 |
| 1 | 157 |
| 2 | 95 |
| 3 | 29 |
| 4 | 5 |

Ferestrele se suprapun si nu sunt observatii independente. Aceste numere
descriu istoricul, nu estimeaza separat probabilitatile viitoare.
Ele infirma ideea ca o rata de aproximativ 90% inseamna exact o pierdere
in fiecare grup de zece pariuri sau ca pierderea poate fi programata.

O rezerva initiala necesita o prima miza sub 100 lei. Retragerea unor castiguri
poate proteja sume deja castigate, dar nu garanteaza recuperarea capitalului
si nu impiedica esecul primului pariu. Cresterea mizei dupa pierderi nu
corecteaza un avantaj statistic absent si poate mari pierderea.
[GambleAware: sfaturi privind limitele si recuperarea pierderilor](https://www.gambleaware.org/advice/for-your-gambling/advice-to-consider-if-you-re-gambling/)

Nu s-au plasat pariuri, consultat cote de bookmaker sau calculat un ROI real.
