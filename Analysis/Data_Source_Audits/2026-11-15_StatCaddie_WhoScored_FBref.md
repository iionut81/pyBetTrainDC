# Evaluare surse: StatCaddie, WhoScored, FBref

15 noiembrie 2026. Cerere: verificarea utilității celor trei site-uri pentru Gol70 și cornere1H≤7. Audit separat de rularea zilnică activă; fără integrare,schimbări de model,programare sau Git.

## StatCaddie: util complementar, acces direct reușit

[Pagina principală](https://statcaddie.co.uk/) arată azi12partide în2competiții,nu acoperirea completă a scopului PRED. [Tenerife–Ceuta](https://statcaddie.co.uk/matches/cd-tenerife-vs-ad-ceuta-15-nov-2026) prezintă filtre sezon/ultime5/10,acasă/deplasare și numitori5/6 în unele tabele, deși eticheta ferestrei spune ultimele10. [Franța–Turcia](https://statcaddie.co.uk/matches/france-vs-turkiye-15-nov-2026) are tabele cu2/2pe fiecare parte. Procentele sunt frecvențe ale eșantionului afișat,nu probabilități validate.

Pagina de meci arată totaluri FTde cornere și are filtruHalves.Nu am confirmat din conținutul accesibil nici listele golurilor pe minute necesareGF70/GA70,nici seria exactă de cornereFirstHalf. Nu afirm că site-ul nu le oferă în alte vizualizări. Furnizorul statistic,API/exportul autorizat,completitudinea arhivei și stabilitatea nu au fost validate. Util pentru triere/confruntarea unor medii,insuficient validat ca sursă principală.

## WhoScored: context sportiv promițător, acces automat blocat

Homepage,/Statistics și/Explanations au returnatHTTP403prin instrumentul web. Acest rezultat se referă la accesul nostru,nu dovedește că pagina nu funcționează în Chrome. [Descrierea oficială a aplicației](https://apps.apple.com/us/app/whoscored-football-live-scores/id940048063) menționează dateOpta,statistici,jucători,analize tactice șiXIprobabile. [Pagina grupului proprietar](https://www.fairplaysportsmedia.com/network/whoscored) identifică produsul.

Utilizarea propusă este verificarea finalistelor și profilului echipelor/jucătorilor;ratingul nu reprezintă probabilitateGol70. Nu am validat pe un meci curent cronologia completă și cornerele1H,acoperirea detaliată a ligilor noastre sau export/APIpermis. Nu ocolim blocajele și nu adăugăm un colector în acest audit.

## FBref: istoric de bază, avertisment important privind datele avansate

Homepage/en și/en/about/au returnat403. ComunicatulSportsReference20ianuarie2026spune că furnizorul datelor avansate a încetat accesul și a cerut ștergerea acestora,iarFBrefcontinuă cu istoric de bază pentru peste100competiții. Articolul individual a returnat403,dar textul oficial este indexat în [arhiva categoriei General](https://www.sports-reference.com/blog/category/general/) și [Advanced Stats](https://www.sports-reference.com/blog/category/advanced-stats/).

Nu îl tratăm implicit ca sursă actuală de xG ori metrici avansate pe baza reputației istorice. O eventuală ofertă nouă,acoperirea sezonului2026/27 și minutele/cornerele1Hexacte trebuie validate înainte de utilizare. Posibil complementar pentru rezultate și loturi istorice,când accesul și ediția pot fi confirmate.

## Recomandare

WhoScored pentru context dacă accesibil,StatCaddie pentru confruntare secundară a agregatelor,FBref pentru istoric de bază verificat.Păstrăm FotMob și sursele oficiale deja documentate ca bază actuală,fără să declarăm aceste trei integrate. Fiecare import viitor necesită identitate/ediție/categorie,perioadă exactă,minute/FirstHalf și numitori verificabili.Lipsa rămâneNULL,nu zero.Douăsite-uri nu garantează doi furnizori independenți.Nicio adăugare de sursă nu demonstrează automat creșterea ratei de succes sau calibrarea.
