# PRED — pagină publică pentru review

Copie separată, numai pentru consultare, destinată vizitatorilor care verifică analizele și evaluarea modelelor. Publicarea a fost cerută explicit la18noiembrie2026.

**Publicat și verificat:** https://iionut81.github.io/pyBetTrainDC/review/

GitHub Pages a confirmat `built` pentru commitul `cac28a1189414ceb3603c464fdc093e2c2811944` de pe `gh-pages`. Pagina, ghidul, un raport individual, un raport de zi și vechiul site răspund HTTP200. Cele17 controale browser au trecut și pe HTTPS public, inclusiv verificarea pe mobil. Dovezi în `verification.json` și `publication-verification.json`.

Pagina existentă de la rădăcina GitHub Pages este păstrată. Sursa datelor este o revizie explicită deja publicată pe `origin/main`, nu starea locală de lucru. Buildul nu execută modele și nu modifică probabilități, alegeri sau rezultate.

Pagina se deschide în Evaluare, pentru ca vizitatorul să poată inspecta rezultatele și decalajul observat–estimat. Panoul general, meciurile și istoricul sunt disponibile din navigarea laterală.

```powershell
node Previews/Pred_Review_Page/export.mjs
node Previews/Pred_Review_Page/check.mjs
```

Opțional, primul argument al exporterului este un SHA sursă. Rezultatul se găsește în `dist/`, iar `publication.json` înregistrează revizia și acoperirea. Exportul păstrează datele modelului și rezultatele; un digest verifică identitatea lor cu sursa. Rapoartele și legăturile sunt adaptate pentru găzduire statică.

Versiunea publicată folosește sursa `275e89b13a7bbbf87b8268ba784863d9e553dd16`:17zile,905partide inventariate și1810rapoarte meci/piață. Nouă resurse auxiliare citate nu existau în revizia sursă; legăturile respective sunt omise și lista este păstrată în manifest. Aceasta nu creează observații sau probabilități suplimentare.

Exporterul extrage o arhivă Git a reviziei într-un cache ignorat din `.runtime/`, pentru a evita citiri repetate și erori de mapare Git pe Windows. Necesită Git, Node.js și tar. Argumentele tar sunt relative pentru compatibilitate cu directoarele Windows cu diacritice. Pentru verificarea versiunii live:

```powershell
node Previews/Pred_Review_Page/check.mjs https://iionut81.github.io/pyBetTrainDC/review/
```

Nu sunt exportate conversații, jurnale personale, configurații de execuție, autentificări sau API-ul local pentru alegeri. Vizitatorii pot filtra și exporta date, fără a modifica proiectul. Ghidul de review explică metoda, cohortele, verificarea și limitele.

Nu există sincronizare automată cu calculatorul local sau un nou job programat. O actualizare publică ulterioară necesită autorizare și publicare explicită. `dist/` și capturile locale sunt ignorate în main; versiunea publicată este păstrată separat pe ramura `gh-pages`, în directorul `review/`.
