# WTA U12.5 Set 2 — CoVe Manual
## Leolia Jeanjean vs Francesca Jones
**Turneu:** Grand Est Open 88 — WTA 125, Contrexeville (Vosges), France  
**Suprafață:** Lut (outdoor) | **Tur:** R2 (R16) | **Data:** 09.07.2026, ~15:30 CEST  
**Seedings:** Jones #4 seed, Jeanjean #10 seed  
**Surse:** CoreTennis.net, TennisStat.com, Robinhood Markets, Meteoblue, WTA draws  

---

## STATUS MODEL

**Jeanjean vs Jones nu apare în `1.5_WTA_Under12_5.csv`** — manual CoVe necesar.  
Procedura: CoVe manual cu date empirice.

---

## TRIPLE FILTER — PASUL 1 (Market Check)

| Criteriu | Valoare | Semnal |
|---|---|---|
| tb_p_cal | N/A (model absent) | — |
| p_elo / p_markov gap | N/A | — |
| unstable_reason | N/A | — |
| Robinhood P(favorita) | **Jeanjean 51% / Jones 49%** | **SKIP** |

**Robinhood check:**
- P(favorita) = **51%** — sub pragul minim de 60%
- Regula: P(favorita) < 60% → **SKIP — meci echilibrat, S2 poate fi lung**
- Piața vede acest meci ca un coin-flip, în ciuda faptului că Jones este seed #4 (WTA 106) vs Jeanjean seed #10 (WTA 132)
- Explicație divergență: Jones are 40.7% win rate în 2026 (8/22 meciuri), 3 retrageri medicale din cauza accidentărilor (coapsă Auckland, gluteal Australian Open, boală Miami). Piața nu o mai evaluează ca jucătoare de top-110 în forma actuală.

**VERDICT PASUL 1: SKIP** ❌

---

## DE CE SKIP — ANALIZĂ COMPLETĂ

### Profilul TennisStat (risc ridicat pentru U12.5 S2)

| Metric | Jeanjean | Jones | Match Average |
|---|---|---|---|
| TB per match | 0.33 (30% meciuri) | 0.37 (32% meciuri) | 0.35 |
| Over 12.5 games/set | **22%** | 5% | **14%** |
| Average games/set | **9.74** | 8.82 | 9.28 |
| Breaks per match | 3 | 3.34 | **6.34** |

**Comparație față de meciurile RECOMANDATE:**
- Zantedeschi-Pigato (RECOMMEND 8/10): 13.38 breaks/match, TB rate 10-20%
- Jeanjean-Jones: 6.34 breaks/match, TB rate **30-32%**, avg games/set 9.74

6.34 breaks combined înseamnă că seturile merg des la 5-5, 6-5, 6-6. Jeanjean servește relativ bine (2.87 ași/meci) și ține serviciul decent. Jones la fel. Rezultat: seturi close cu risc real de TB.

---

## DATE EMPIRICE S2 TB (pentru audit)

### Leolia Jeanjean — Clay S2 TB Rate (CoreTennis ID 9983)

**Sample: 7 meciuri în 2 seturi pe lut (2024-2026) — sample MIC**

| Data | Turneu | Scor | S1 | S2 |
|---|---|---|---|---|
| Jul 2026 | Contrexeville R1 | W 6-4 6-3 | no TB | no TB |
| May 2025 | Roma Q (vs Parry) | W 6-1 7-6(0) | no TB | **TB** |
| May 2025 | Roma (vs Haddad Maia) | W 7-6(6) 6-4 | **TB** | no TB |
| Jun 2024 | Makarska (vs Lukas) | W 6-2 6-2 | no TB | no TB |
| Jun 2024 | Makarska (vs Bulgaru) | W 6-4 6-2 | no TB | no TB |
| May 2025 | Strasbourg (vs Fernandez) | W 6-1 7-5 | no TB | no TB |
| Jun 2024 | Roland Garros (vs Swiatek) | L 6-1 6-2 | no TB | no TB |

**S2 TB rate: 1/7 = 14.3%** — borderline (sub 15%, dar sample CRITIC DE MIC)

**S1 TB → S2 cascade (lut):**

| Meci | S1 | S2 |
|---|---|---|
| vs Haddad Maia (RG 2025) | 7-6(6) | 6-4 (no TB) |
| vs Tomljanovic (Roma 2025) | 7-6(5) | 5-7 (no TB) |
| vs Paolini (Roma 2025) | 6-7(4) | 6-2 (no TB) |
| vs Gibson (St. Malo 2025) | 6-7(5) | 6-2 (no TB) |

**Cascade S1→S2: 0/4 = 0%** ✅ (semnal pozitiv)

**Atenție:** Sample de 7 meciuri în 2 seturi este insuficient pentru o concluzie robustă. Intervalul de încredere pentru 14.3% pe 7 meciuri este extrem de larg (2%-50%). Jeanjean joacă frecvent în 3 seturi (battle-style), ceea ce reduce sample-ul de 2 seturi pe lut.

---

### Francesca Jones — Clay S2 TB Rate (CoreTennis ID 74984)

**Sample: 33 meciuri în 2 seturi pe lut (2024-2026) — sample solid**

**S2 TBs confirmate:**

| Data | Turneu | Scor | S1 | S2 | Context oponent |
|---|---|---|---|---|---|
| May 2025 | Roland Garros R32 | W 6-0 7-6(3) | no TB | **TB** | Bouzkova M. — WTA ~50 (top-50) |
| Sep 2025 | Sao Paulo R16 | W 6-4 7-6(6) | no TB | **TB** | Osuigwe W. — WTA ~200 |
| Jul 2025 | Contrexeville F | W 6-4 7-6(2) | no TB | **TB** | Jacquemot E. — WTA ~90 |

**S2 TB rate: 3/33 = 9.1%** — sub 15% → semnal pozitiv

**Context S2 TBs Jones:**
- vs Bouzkova (RG, WTA ~50): adversar de top-50, nivel WTA 250 final → TB explicabilă la nivel înalt
- vs Jacquemot (Contrexeville final 2025): finala turneului, adversar local, presiune maximă → TB în finală ≠ pattern standard
- vs Osuigwe (Sao Paulo WTA 250): adversar WTA ~200, s-ar putea să fi survenit un drop de concentrare

**S1 TB → S2 cascade (lut, 2-set):**
- Timofeeva: 7-6(2) 6-3 → S2=6-3 ✓
- Tjen: 7-6(0) 6-3 → S2=6-3 ✓
- Marcinko: 7-6(4) 6-3 → S2=6-3 ✓

**Cascade S1→S2: 0/3 = 0%** ✅ (excellent)

---

## H2H DETALIAT

| Data | Turneu | Suprafață | Scor | S1 | S2 | S3 |
|---|---|---|---|---|---|---|
| Mar 2024 | W75 Vacaria, Final | Lut | Jones W 1-6 6-4 6-1 | no TB | no TB | no TB |

**Singurul meci:** Jones a întors din 0-1 seturi, câștigând 6-4 6-1 seturile 2 și 3. Niciun TB în cele 3 seturi. **Dar:** Contextul era diferit (Vacaria W75, ambele jucătoare la nivel ITF/W75). Jeanjean era mai sus în ranking la acea vreme.

---

## CONTEXT DETAILAT

### Condiție fizică și form

| Factor | Jeanjean | Jones |
|---|---|---|
| R1 scor | W 6-4 6-3 vs Monnot (FRA) | W 6-2 7-5 vs Rame (FRA) |
| Durată R1 | ~60 min | ~70 min |
| 2026 win rate | 51.9% (14/27) | **36.4% (8/22)** |
| Retrageri 2026 | 0 documentate | **3 retrageri** (coapsă, gluteal, boală) |
| Ultimul meci | Wimbledon R2 pierdut (02.07) | — |
| Accidentare curentă | Nu | Nu (dar istoric fragil 2026) |

### Motivație și miză

- **Jones:** Defending champion la Contrexeville (a câștigat titlul în 2025 exact pe acest teren, a bătut Jacquemot în finală 6-4 7-6). Motivație psihologică masivă de a-și apăra titlul. Familiaritate maximă cu curtea, condiții, organizare.
- **Jeanjean:** Franța (homecourt advantage emoțional), suport local. Dar ranking în scădere (career high 91, acum 132). Sezon inconsistent.

### Stil de joc și compatibilitate U12.5 S2

- **Jeanjean:** Topspin forehand, clay baseline player. Servitoare bună (2.87 ași/meci). **9.74 games/set medie** = cel mai ridicat din toate meciurile analizate azi. Tinde spre seturi lungi, close.
- **Jones:** Ectrodactyly (7 degete, 7 degete la picioare) — adaptare completă a serviciului. 4.42 duble greșeli/meci = serviciu nesigur. Joacă agresiv, atac precoce.
- **Compatibilitate:** 6.34 breaks/meci combined (mic), 9.28 avg games/set — profil tipic pentru meciuri close. Risc S2 TB real.

### Condiții meteo — Contrexeville, 09.07.2026

| Parametru | Valoare |
|---|---|
| Temperatură | **33°C** (very hot) |
| Condiții | Soare, cer senin |
| Vânt | 16 km/h NE |
| Precipitații | 0% |

33°C pe lut = rally-urile sunt scurte (caldura reduce capacitatea de a sustine rally-uri lungi), **dar** și serviciile devin mai dificile (sudoare, minge caldă), ceea ce poate favoriza break-urile în primul joc al setului.

---

## ESTIMARE CÂȘTIGĂTOARE

Robinhood 51%/49% reflectă realitatea: meci echilibrat.

| Factor | Jeanjean | Jones |
|---|---|---|
| Ranking | 132 | 106 (dar form 36.4%) |
| Form 2026 | 51.9% | 36.4% |
| Clay titluri 2025-26 | 0 | 2 (Palermo + Contrexeville) |
| Defending champion | Nu | **Da (Contrexeville 2025)** |
| Homecourt | **Da (FRA)** | Nu |
| Fitness | Solidă | Fragil (3 retrageri 2026) |
| H2H | 0-1 | 1-0 |

**Estimare câștigătoare: Jeanjean ușor favorită (51-55% în scenariul nostru)** — Jones are avantajul psihologic de defending champion dar fizicul questionable. Jeanjean joacă mai consistent în 2026.

**Scor estimat dacă Jeanjean câștigă:** 6-3 7-5 sau 7-5 6-4 (match strâns, fără TB dar scurt nu va fi)  
**Scor estimat dacă Jones câștigă:** 6-4 6-3 sau 1-6 6-3 6-1 (comeback pattern ca în H2H Vacaria)

---

## VERDICT FINAL

```
MARKET:   WTA U12.5 Set 2
MECI:     Jeanjean vs Jones — Contrexeville WTA 125, Clay, R2
DATA:     09.07.2026
SCOR:     SKIP — Nu recomandăm
MOTIV:    Robinhood P(favorita) = 51% < 60% (prag minim Triple Filter Pasul 1)
```

**Concluzie analyst:** Meci echilibrat fără class gap confirmat de piață. Profilul TennisStat (30-32% TB rate per match, avg 9.74 games/set pentru Jeanjean) confirmă că acesta este tipul de meci care generează seturi lungi și TB. Chiar dacă datele empirice S2 TB sunt parțial favorabile (Jones 9.1% pe 33 meciuri), lipsa unui favorit clar face pierderea de valoare la U12.5 S2. SKIP corect conform workflow Triple Filter.

**Dacă se joacă:** Observă dacă setul 1 se termină rapid (6-2, 6-3) — asta ar putea indica form mismatch real și S2 mai scurt. Dar nu paria înainte.
