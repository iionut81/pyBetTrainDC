# PRED — copie web de consultare

Stare: PUBLICAT pe GitHub Pages dupa aprobarea explicita a utilizatorului.

Link: https://iionut81.github.io/pyBetTrainDC/

Ramura de publicare: gh-pages. Commit publicare:
2e01ad5120a97aafc687f12535af071b2ee21b03. GitHub Pages confirma built;
rootul, dashboardul, raportul individual, review-ul si publication.json au
raspuns HTTP200 fara autentificare. Este copia aprobata2-4 noiembrie2026.
O actualizare locala nu actualizeaza automat site-ul: este necesara o noua
publicare autorizata. Fluxul local de salvare a alegerilor ramane separat.

Exportul foloseste NUMAI snapshotul si rapoartele deja publicate pe origin/main,
commit 253a39ec1acd466c80e42fd67114de70c182da11. Nu copia datele locale recente,
transcripturi, jurnale de conversatie, configuratii, procese sau serviciul Node.

- 3 zile (2–4 noiembrie), 64 partide, rapoarte individuale si rezultate.
- Cele doua alegeri explicite publicate pe4 noiembrie sunt vizibile.
- Selectarea/anularea si toate legaturile spre lansatorul local sunt eliminate.
- Rapoartele Markdown sunt prezentate ca HTML pentru citire pe mobil.
- Fisierele sursa CSV necesare rapoartelor sunt incluse; nu exista API de scriere.

```powershell
node Previews/Pred_Public_Dashboard/export.mjs
node Previews/Pred_Public_Dashboard/check-public.mjs
```

Sites a creat anterior proiectul privat indicat in .openai/hosting.json,
dar nu permite public in acest cont; nu au fost incarcate surse si nu exista
deployment Sites. Nu crea alt proiect si nu prezenta expected_url ca activ.
Utilizatorul a autorizat explicit GitHub Pages; acela este furnizorul utilizat.
