# WTA U12.5 Set 2 — CoVe Analysis
## Lucia Bronzetti vs Nuria Brancaccio
### ATV Bancomat Tennis Open, Rome | WTA 125 | Clay | SF (derby italian) | 18 iulie 2026 | ~16:00-17:00 CEST

---

## 1. DATE MODEL

| Câmp | Valoare |
|---|---|
| Turneu | ATV Bancomat Tennis Open, Rome (WTA 125) |
| Suprafață | Clay |
| Round | SF |
| Player A (model) | Lucia Bronzetti |
| Player B (model) | Nuria Brancaccio |
| p_hold_a — Bronzetti | **0.6717** (67.17%) |
| p_hold_b — Brancaccio | **0.4876** (48.76%) |
| hold_asym | 0.1841 |
| min_hold | **0.4876** (Brancaccio) |
| bci | 0.0943 |
| blowout_score | 7 |
| fatigue_flag_a / fatigue_flag_b | True / True (ambele au jucat 3 seturi în QF ieri) |
| tb_p_raw | 0.0403 |
| tb_p_cal (calibrated clay) | **0.0927** |
| p_u125 | 0.9073 |
| premium_elite | no |
| premium_u125 | **no** (ratat — vezi notă) |
| danger_zone | no |
| p_markov (Bronzetti câștigă, raw) | **84.59%** |
| p_elo (Bronzetti câștigă) | 55.54% |
| p_cal blended (Winner model) | **66.69%** |
| GAP \|p_elo − p_markov\| × 100 | **29.05pp** |
| expected_games (match total) | 22.23 |
| days_rest_a / days_rest_b | 1 / 1 |

**Notă tehnică:** Match-ul satisface aparent criteriile premium_u125 (min_hold 0.4876<0.50 ✓, hold_asym 0.1841>0.15 ✓, tb_p_cal 0.0927<0.10 ✓), dar CSV-ul îl marchează premium_u125=no — probabil un criteriu suplimentar (bci sau sample threshold) nedocumentat exact. Tratăm pick-ul ca **non-premium standard** (recommended=True simplu), nu premium.

---

## 2. PASUL 1 — CSV Model + Market Check

### Filtre de bază:

| Filtru | Valoare | Status |
|---|---|---|
| tb_p_cal ≤ 0.10 | 0.0927 | ✅ |
| p_elo ≠ 0.0 | 0.5554 | ✅ |
| GAP Elo/Markov ≤ 35pp | **29.05pp** | ⚠️ elevat, sub prag |
| UNSTABLE flag | none | ✅ |
| danger_zone | no | ✅ |
| premium_u125 | no | standard pick |
| min_hold | 0.4876 | zonă medie (nu extrem de slab) |

**Analiza GAP 29.05pp:** p_markov=84.59% (din hold rates) vs p_elo=55.54% (istoric Sackmann). Diferență mare, similară cu tendința observată și la Badosa/Zidansek — modelul de hold poate supra-reprezenta un jucător cu hold rate mult mai bun (Bronzetti 67% vs Brancaccio 49%) fără să reflecte că Brancaccio a jucat constant meciuri strânse recent (a salvat match point în QF ieri, 6-3 1-6 7-6(5) vs Riera). Elo (55.54%) e mai conservator și mai aliniat cu H2H real (1-2 Brancaccio, meciuri competitive).

### Market check:

| Sursă | P(Bronzetti) | Status |
|---|---|---|
| Robinhood / bookmaker match winner | **indisponibil** — niciun market activ găsit pentru acest SF de Challenger WTA125 | ⚠️ gap în protocol |
| p_cal blended (model) | 66.69% | referință principală folosită |
| p_markov (model, raw) | 84.59% | outlier, tratat cu prudență |
| Sackmann Elo | 55.54% | conservator |

**Limitare recunoscută:** La nivel WTA125 Challenger, piețele de predicție (Robinhood) și bookmakerii mainstream nu oferă cotă directă pentru acest SF — am căutat activ și nu am găsit. Nu putem aplica triple-guard-ul complet (Elo/Markov/Market). Ne bazăm pe **p_cal blended (66.69%)** + H2H recent + context, cu prudență suplimentară în scor.

**Concluzie Pasul 1:** ✅ PASSED cu rezerve — non-premium, GAP mare (29pp) fără confirmare de piață. Nu SKIP (fără semnal contrar), dar scor redus față de un pick premium/confirmat de market.

---

## 3. PASUL 2 — Date Empirice S2 Tiebreak

### 3a. NURIA BRANCACCIO — Clay S2 TB Log (2024-2026, WTA125+ relevant tier)

*Sursă: [CoreTennis Brancaccio](https://www.coretennis.net/tennis-player/nuria-brancaccio/71197/results.html)*

**Meciuri WTA125+ (2025-2026, nivel relevant pentru SF Rome):**

| Turneu | Adversară | Scor complet | S2 | S2 TB? |
|---|---|---|---|---|
| Rome 2026 QF | Riera (~150) | 6-3 1-6 7-6(5) | 1-6 | No |
| Rome 2026 R2 | Vedder | 6-1 6-2 | 6-2 | No |
| Rome 2026 R1 | Pieri | 6-2 6-2 | 6-2 | No |
| Brescia 2025 R1 | Gorgodze | 4-6 6-2 7-6(3) | 6-2 | No |
| Modena 2025 R1 | Bosio | 6-4 7-5 | 7-5 | No |
| Modena 2025 R2 | Zidansek (~148) | 6-2 6-2 | 6-2 | No |
| **Modena 2025 QF** | **Bronzetti** | **6-4 6-2** | **6-2** | **No** |
| Foggia 2025 R1 | Prozorova | 6-4 6-0 | 6-0 | No |
| Oeiras 2025 R1 | Marcinko | 7-5 6-1 | 6-1 | No |
| Madrid 2025 R1 | Ponchet | 7-6(5) 6-1 | 6-1 | No |
| Dubrovnik 2025 R1 | Pigato | 6-2 7-5 | 7-5 | No |
| Dubrovnik 2025 R2 | Kalinina | 6-0 6-2 | 6-2 | No |
| Antalya(Mar10) 2025 R1 | **Bronzetti** | **6-1 7-5** | 7-5 | No |
| Antalya(Mar10) 2025 R2 | Grabher | 3-6 6-3 6-1 | 6-3 | No |
| Antalya(Mar10) 2025 QF | Erjavec | 6-4 6-1 | 6-1 | No |
| Antalya(Mar3) 2025 R1 | Oliynykova | 2-6 6-3 6-0 | 6-3 | No |
| Antalya(Feb24) 2025 R1 | Kumru | 6-2 6-2 | 6-2 | No |
| Antalya(Feb24) 2025 R2 | Udvardy | 6-7(9) 6-2 7-5 | 6-2 | No |
| Antalya(Feb24) 2025 QF | Erjavec | 7-6(4) 0-6 6-2 | 0-6 | No |
| La Bisbal 2024 R1 | Eva Lys (~junior talent, outside top150) | 6-3 7-6(7) | 7-6(7) | **YES** |

**S2 TB rate WTA125+ (2024-2026): 1 din 20 meciuri = 5%**

**Contextualizare TB — La Bisbal 2024 vs Eva Lys:**
Lys era o jucătoare tânwithout ranking solid la acea vreme (outside top 150), talentată dar inconsistentă. Nivel WTA125, dar 2 ani vechime și context diferit (Brancaccio la un alt nivel de formă). Relevanță moderată-scăzută pentru matchup-ul de azi.

**S1 TB → S2 pattern (Brancaccio, WTA125+): 0/7 = 0%** — toate cele 7 meciuri cu TB în S1 (Mallorca-nu e in tabel de mai sus dar in log complet; Rende R2, Antalya Feb24 R2 vs Udvardy, Antalya Feb24 QF vs Erjavec, Madrid R1 2025, ș.a.) au avut S2 fără TB. Semnal excelent, foarte curat.

### 3b. LUCIA BRONZETTI — Clay S2 TB Log (2025-2026)

*Sursă: [CoreTennis Bronzetti](https://www.coretennis.net/tennis-player/lucia-bronzetti/82226/results.html)*

| Turneu | Adversară | Scor complet | S2 | S2 TB? |
|---|---|---|---|---|
| Rome 2026 R1 | Mazzola | 7-5 6-1 | 6-1 | No |
| Rome 2026 R2 | Kostovic | 6-4 6-4 | 6-4 | No |
| Rome 2026 QF | Chiesa | 2-6 6-3 6-2 | 6-3 | No |
| Modena 2026 R1 | Gao | 4-6 6-3 6-1 | 6-3 | No |
| **Modena 2026 R2** | **Grabher (~top100)** | **7-6(6) 7-6(4)** | **7-6(4)** | **YES** |
| Modena 2026 QF | Brancaccio | 6-4 6-2 | 6-2 | No |
| Modena 2026 SF | Samson | 7-6(3) 6-1 | 6-1 | No |
| Modena 2026 F | loss to Kawa | 6-1 4-6 7-6(6) | 4-6 | No |
| Foggia 2026 R1 | Paganetti | 6-1 6-0 | 6-0 | No |
| Foggia 2026 R2 | Mair | 6-2 6-1 | 6-1 | No |
| Foggia 2026 QF | Pedone | 6-2 6-4 | 6-4 | No |
| Foggia 2026 SF | loss to Romero Gormaz | 6-4 6-1 | 6-1 | No |
| Contrexeville 2025 R1 | Waltert | 6-1 6-2 | 6-2 | No |
| Contrexeville 2025 R2 | Baindl | 7-5 5-7 6-3 | 5-7 | No |
| Contrexeville 2025 QF | Jacquemot | 4-6 6-3 6-4 | 6-3 | No |
| Contrexeville 2025 SF | Gao | 6-2 6-1 | 6-1 | No |
| **Contrexeville 2025 F** | **Sherif (top100)** | **6-4 6-7(4) 7-5** | **6-7(4)** | **YES** |

**S2 TB rate (2025-2026): 2 din 17 meciuri = 11.8%** — sub prag 15%, dar peste dublul ratei lui Brancaccio.

**Contextualizare TB-uri Bronzetti (CRITIC pentru acest matchup):**

| TB | Adversară | Nivel adversară | Relevanță |
|---|---|---|---|
| Modena 2026 R2 vs Grabher | ~top100, hold bun pe clay | WTA125, recent (o lună) | ⚠️ **relevant** — Grabher e mult mai solidă pe serviciu decât Brancaccio (49% hold) |
| Contrexeville 2025 F vs Sherif | top100, foarte solidă pe clay | WTA125 finală | ⚠️ **relevant** — Sherif e o adversară de calitate superioară lui Brancaccio |

**Concluzie critică:** Ambele TB-uri ale lui Bronzetti au venit contra unor jucătoare **mai solide pe serviciu** decât Brancaccio (min_hold 48.76%, sub media adversarelor din TB-uri). Structural, riscul de TB Bronzetti-indus e mai mic azi decât în acele 2 meciuri.

**S1 TB → S2 pattern (Bronzetti): 1/2 = 50%** ⚠️ **ATENȚIE — sample foarte mic (n=2)**

| Meci | S1 final | S2 final | S2 TB? |
|---|---|---|---|
| Modena 2026 R2 vs Grabher | Câștigat TB 7-6(6) | Câștigat TB 7-6(4) | **YES — cascade** |
| Modena 2026 SF vs Samson | Câștigat TB 7-6(3) | 6-1 | No |

Per protocol (S1TB→S2TB > 50% → scor maxim 6/10), acest semnal ar cere o capare severă. **Dar sample-ul e de doar 2 meciuri** — statistic nesemnificativ. Tratăm ca semnal de precauție moderată (-2pp), nu ca hard cap la 6/10, dat fiind că un singur caz din două nu constituie un pattern robust.

**Concluzie Pasul 2:**

| Condiție | Brancaccio | Bronzetti | Status |
|---|---|---|---|
| S2 TB rate ≤15% | 5% | 11.8% | ✅ ✅ |
| Sample ≥10 matches clay | 20 (WTA125+) | 17 | ✅ ✅ |
| S1→S2 cascade ≤20% | 0/7 = 0% | **1/2 = 50%** ⚠️ | ✅ ⚠️ (sample mic) |

---

## 4. PASUL 3 — Context

### Condiție fizică:

**Nuria Brancaccio:**
- Fără accidentări documentate.
- QF ieri: 3 seturi dramatice vs Riera (6-3 1-6 7-6(5)), a salvat match point — efort mare, mental și fizic.
- 1 zi odihnă. An 2026 dificil (38.7% win rate, "Bad Form"), dar acest turneu = revenire (a intrat aproape de top 200).

**Lucia Bronzetti:**
- Fără accidentări documentate.
- QF ieri: 3 seturi cu revenire (2-6 6-3 6-2 vs Chiesa) — efort mare similar.
- 1 zi odihnă. Formă 2026 medie (57.5% win rate), mai stabilă decât Brancaccio.

| Factor fizic | Brancaccio | Bronzetti |
|---|---|---|
| days_rest | 1 | 1 |
| had_3sets_7d | True | True |
| Meciuri recente | 3 Rome, toate câștigate (inclusiv comeback) | 3 Rome, toate câștigate (inclusiv comeback) |
| Fatigue level | Ridicat (echilibrat cu adversara) | Ridicat (echilibrat) |

*Surse: [TieBreakTennis — tripletta azzurra](https://www.tiebreaktennis.it/wta-125-roma-2026-tripletta-azzurra-in-semifinale-brancaccio-trevisan-e-bronzetti-sognano-il-titolo/)*

### Motivație:

**Brancaccio:** Citat direct: *"Am venit aici să mă găsesc pe mine"* — an dificil (38.7% win), acest turneu îi readuce ranking-ul spre top 200. Motivație emoțională mare, derby italian pe teren propriu (Rome).

**Bronzetti:** Formă mai stabilă, motivată de un traseu solid la Rome 2026 (a bătut deja o dată pe Brancaccio recent, la Modena QF). Joacă acasă (Italia), presiune moderată pozitivă.

*Sursă: [TieBreakTennis](https://www.tiebreaktennis.it/wta-125-roma-2026-tripletta-azzurra-in-semifinale-brancaccio-trevisan-e-bronzetti-sognano-il-titolo/)*

### Antrenori:

- **Bronzetti:** Francesco Piccari (academia lui din Anzio), cu Karin Knapp (fost top-50) ca advisor ocazional.
- **Brancaccio:** Nicio informație publică găsită despre antrenorul actual.

*Sursă: [EssentiallySports — Bronzetti coach](https://www.essentiallysports.com/wta-tennis-news-who-is-lucia-bronzettis-coach-at-the-cincinnati-open-all-you-need-to-know-about-him/)*

### Meteo Rome — 18 iulie 2026, ~16:00-17:00 CEST:

| Factor | Valoare | Impact U12.5 |
|---|---|---|
| Temperatură | ~28-30°C | Cald → mingi rapide → puncte scurte → ✅ |
| Umiditate | **~74%** (ridicată) | Aer greu → oboseală accelerată pentru ambele (deja fatigate) → ✅ pentru break-uri |
| Vânt | Ușor | Neglijabil |
| Precipitații | Puțin probabile (iulie = uscat la Roma) | Meci probabil garantat |

*Surse: [Climate-data Rome](https://en.climate-data.org/europe/italy/lazio/rome-1185/t/july-7/) | [Weather2Travel Rome](https://www.weather2travel.com/italy/rome/july/)*

### Stil de joc — Matchup dynamics:

**Nuria Brancaccio:** Joc de baseline solid pe clay (61.2% win rate carieră pe zgură), dar servici relativ slab (48.76% hold model, 65% BP won defensiv confirmă că se bazează pe retur/break-uri, nu pe serviciu). Foarte multe break-uri generate (7.35/meci per TennisStats).

**Lucia Bronzetti:** Serviciu mai solid (67.17% hold model, viteză 161km/h vs 144.5km/h Brancaccio), dar relativ puține break-uri generate (3.34/meci) — profil mai puțin agresiv la retur, se bazează pe propriul serviciu.

**Matchup structural:** Bronzetti ține serviciul semnificativ mai bine → Brancaccio va fi ruptă frecvent. Dar Brancaccio e MULT mai bună la retur (break rate mare) → poate crea presiune pe serviciul lui Bronzetti. Combinația: multe break-uri de ambele părți, dar Bronzetti câștigă mai multe game-uri nete (serviciu mai solid + break-uri proprii) → seturi tipic 6-3, 6-4, nu 7-5/7-6, CONFIRMAT de H2H recent (Modena QF 6-4 6-2).

*Sursă: [TennisStats H2H](https://tennisstats.com/h2h/nuria-brancaccio-lucia-bronzetti)*

### Context psihologic:

**Brancaccio:** An dificil mental (formă "Bad"), dar declarații publice pozitive despre suportul echipei ei ("thanked her coaching team for supporting her mental approach during high-pressure moments") — semnal de reziliență mentală în ciuda rezultatelor slabe per total.

**Bronzetti:** Fără semnale de criză mentală, formă stabilă medie, experiență de nivel superior (fost top-50 mondial, WTA rank 139 curent vs career-high mult mai bun).

### Head-to-Head:

**H2H: 2-1 Bronzetti** (istoric total), dar cronologic:
1. 2021 W25 Torino (ITF): Bronzetti 2-0
2. Mar 2026 Antalya 3: **Brancaccio** 2-0
3. **Jun 2026 Modena QF: Bronzetti 6-4 6-2** (cel mai recent, o lună în urmă)

Cel mai recent meci (Modena, acum o lună, exact aceleași condiții — clay, WTA125) confirmă direct: Bronzetti a câștigat clar 6-4 6-2, fără niciun TB, cu Set 2 sub 12.5 game-uri (7 game-uri). Precedent direct și foarte relevant.

*Surse: [TennisStats H2H Brancaccio-Bronzetti](https://tennisstats.com/h2h/nuria-brancaccio-lucia-bronzetti)*

---

## 5. SCOR FINAL

| Factor | Valoare | Ajustare |
|---|---|---|
| Baza tabel (S2 TB ≤15% ambele, sample ok) | 9/10 | — |
| Bronzetti S1→S2 cascade 50% (n=2, sample mic) | risc moderat, dar nesemnificativ statistic | **-2pp** |
| Non-premium (premium_u125=no) | pick standard, nu premium | 0pp (deja reflectat) |
| GAP Elo/Markov 29.05pp fără confirmare de piață | incertitudine mai mare decât normal | **-1pp** |
| Robinhood/bookmaker indisponibil | gap în triple-guard | 0pp (nu SKIP, dar prudență) |
| H2H recent (Modena, o lună în urmă): 6-4 6-2, S2 clean | confirmare directă puternică | 0pp (deja susține scorul de bază) |
| Ambele fatigate egal (1 zi odihnă, 3 seturi ieri) | simetric | 0pp |

### **SCOR FINAL: 6/10 — PASS / VALOARE MARGINALĂ**

**Motiv reducere sub minimul de 8/10 (clay):** Cascade risk (50%, Bronzetti, favorita) + absența confirmării de piață + GAP mare fac acest pick sub standardul minim clay (8/10 + RH). Semnalele individuale de S2 TB rate sunt bune (5% / 11.8%), dar combinația de incertitudini (fără market check, cascade thin-sample dar prezent) nu justifică un pick de încredere ridicată.

---

## 6. ATENȚIONARE

**NU RECOMANDĂM acest pick la scor sub prag.** Motive: (1) fără confirmare de piață independentă — nu putem aplica triple-guard-ul complet; (2) cascade S1TB→S2TB 50% pentru favorită (Bronzetti), chiar dacă sample e mic (n=2); (3) GAP Elo/Markov 29pp, al doilea cel mai mare din analizele recente. Dacă totuși pariezi, tratează ca pick speculativ, nu ca pick de bază (premium).

**Semnal pozitiv real:** H2H direct recent (Modena, o lună în urmă, exact acest matchup) a confirmat exact scenariul — 6-4 6-2, Set 2 clean sub 12.5. Acesta e cel mai bun predictor disponibil, dar nu compensează singur gap-urile de mai sus pentru un scor premium.

---

## 7. PREDICȚIE MECI

**Câștigătoare probabilă: Lucia Bronzetti** (~67% model blended)

**Motivare:** Hold rate superior (67.17% vs 48.76%), victorie recentă directă în H2H (Modena, o lună în urmă), formă mai stabilă (57.5% vs 38.7% în 2026), serviciu mult mai rapid (161 vs 144.5 km/h).

**Scenarii probabile:**

| Scor | Câștigătoare | Probabilitate |
|---|---|---|
| **6-4 6-2** | Bronzetti | 20% |
| **6-3 6-4** | Bronzetti | 16% |
| **6-4 6-3** | Bronzetti | 14% |
| **7-5 6-4** | Bronzetti | 8% |
| 3 seturi (diverse) | Bronzetti | 10% |
| Brancaccio câștigă (upset, cf. H2H istoric competitiv) | Brancaccio | ~32% |

**Estimare cea mai probabilă: 6-4 6-2 (Bronzetti)** — replică aproximativă a rezultatului direct din Modena (acum o lună).

**S2 estimat:** Probabil 6-2 sau 6-3 pentru Bronzetti, cu câteva break-uri reciproce dar fără presiune reală de TB — coerent cu istoricul direct și cu S2 TB rate scăzut al lui Brancaccio (5%). Risc rezidual moderat din cascade-ul Bronzetti (50%, n=2) dacă S1 devine neașteptat de strâns/TB.

---

## 8. SURSE

1. [TieBreakTennis — Tripletta azzurra Rome SF](https://www.tiebreaktennis.it/wta-125-roma-2026-tripletta-azzurra-in-semifinale-brancaccio-trevisan-e-bronzetti-sognano-il-titolo/) — context SF, motivație, citate
2. [CoreTennis — Brancaccio results](https://www.coretennis.net/tennis-player/nuria-brancaccio/71197/results.html) — log clay S2 TB
3. [CoreTennis — Bronzetti results](https://www.coretennis.net/tennis-player/lucia-bronzetti/82226/results.html) — log clay S2 TB
4. [TennisStats — H2H Brancaccio vs Bronzetti](https://tennisstats.com/h2h/nuria-brancaccio-lucia-bronzetti) — date model, H2H, stats sezon
5. [EssentiallySports — Bronzetti coach](https://www.essentiallysports.com/wta-tennis-news-who-is-lucia-bronzettis-coach-at-the-cincinnati-open-all-you-need-to-know-about-him/) — Francesco Piccari
6. [Matchstat — Brancaccio ranking/clay record](https://matchstat.com/tennis/player/Nuria%20Brancaccio/) — 64.8% clay Challenger/ITF ultimele 52 săpt
7. [Climate-data Rome iulie](https://en.climate-data.org/europe/italy/lazio/rome-1185/t/july-7/) — 29.9°C mediu 11-20 iulie
8. [Weather2Travel Rome iulie](https://www.weather2travel.com/italy/rome/july/) — umiditate 74%
9. [FanDuel — Brancaccio vs Riera QF odds](https://sportsbook.fanduel.com/tennis/wta-rome-ii-2026/nuria-brancaccio-v-eva-vedder-35824739) — referință context odds QF (fără date SF disponibile)
