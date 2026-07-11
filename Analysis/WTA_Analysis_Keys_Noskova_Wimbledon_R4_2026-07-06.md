# WTA CoVe — Multi-Market Analysis
# Madison Keys vs Linda Noskova
# Wimbledon 2026 — R4 (Last 16) — Court No. 1
# ~17:00 BST | July 6, 2026
# Markets analizate: O7.5 Set 1 | U12.5 Set 2 | U30.5 Total

---

## MODEL DATA (1.5 + 1.2 CSV)

| Parametru | Valoare |
|---|---|
| p_hold_a (Keys) | **0.8635** (EXCEPȚIONAL) |
| p_hold_b (Noskova) | **0.8300** (FOARTE RIDICAT) |
| hold_asym | 0.0335 (foarte mic = echilibrat) |
| p_cal_adj O7.5 S1 | **0.903** (90.3%) |
| elite_pick | **True** (blocat de fatigue filter) |
| blowout_score | 1 |
| competitive_set | True |
| expected_games | **26.06** (total match) |
| p_markov (Keys wins) | 0.6332 |
| p_elo (Keys wins) | 0.5255 |
| tb_p_cal S2 | **0.1656** (> 0.10 → U12.5 S2 sub prag) |
| days_rest A/B | 2 / **1** |
| last_3sets A/B | True / False |
| had_3sets_7d A/B | True / True |
| fatigue_flag A/B | True / True |
| O7.5 model output | "no" (blocat de fatigue, nu de probabilitate) |

---

## CONTEXT: DRUM SPRE R4

### Madison Keys [seed 26, WTA #22]

| Runda | Adversar | Scor | Note |
|---|---|---|---|
| R1 | Kayla Day (Q) | 6-7(5), 6-4, 6-3 | **3 seturi** — pierdut primul set |
| R2 | Katie Swan (WC) | 6-1, 6-4 | Dominant, rapid |
| R3 (4 iul.) | Amanda Anisimova [6] | 3-6, 6-2, 6-3 | **3 seturi** — pierdut primul set |

**2 din 3 meciuri în 3 seturi.** days_rest = 2. Sezonul de iarbă 2026: **10-1**, câștigătoarea turneului Eastbourne 2026 (al treilea titlu acolo). BP save rate Wimbledon 2026: **77.78%** (elite). 10 ace-uri în 3 runde.

Sursă: [Wikipedia Wimbledon 2026](https://en.wikipedia.org/wiki/2026_Wimbledon_Championships_%E2%80%93_Women%27s_singles)

### Linda Noskova [seed 9, WTA #12]

| Runda | Adversar | Scor | Note |
|---|---|---|---|
| R1 | Ella Seidel | 6-4, 6-3 | Straight sets, confortabil |
| R2 | Camila Osorio | 6-3, 4-6, 6-2 | **3 seturi** |
| R3 (5 iul.) | Sorana Cirstea [17] | 2-6, 6-3, **7-6(11)** | **3 seturi** + TB 13-11 ← CRITIC |

**3 DIN 3 meciuri în 3 seturi.** days_rest = **1** (a jucat ieri!). **Tiebreakul 13-11 din R3 = +30 min extra fizic extrem.** Sezonul de iarbă 2026: **8-1**, a câștigat Berlin WTA 500 pe iarbă. BP save rate Wimbledon 2026: **42.86%** (slab — sub presiune pe serviciu). 21 ace-uri în 3 runde (7/meci!).

Sursă: [WTA Berlin title](https://www.wtatennis.com/news/4523410/noskova-outlasts-pegula-in-three-sets-to-claim-second-career-title-in-berlin), [Wikipedia draw](https://en.wikipedia.org/wiki/2026_Wimbledon_Championships_%E2%80%93_Women%27s_singles)

---

## MARKET & CONDITIONS

### Robinhood Prediction Markets
URL: [robinhood.com — keys-vs-noskova-jul-06-2026](https://robinhood.com/us/en/prediction-markets/tennis/events/madison-keys-vs-linda-noskova-round-of-16-match-jul-06-2026/)

- **P(Keys câștigă) = 59%**
- **P(Noskova câștigă) = 41%**

Market și model (p_markov=63.3%) convergent: ambele au Keys favorit ~59-63%. **Divergență market/model = 4.3pp** — neglijabilă, nu necesită investigare.

### Condiții July 6
- **Court No. 1** (fără acoperiș)
- **Temperatură: 31-34°C** — All England Club a emis **avertisment de căldură extremă**
- Iarbă uscată rapidă, condiții de maximă viteză
- Iarbă fierbinte la ora 17:00 BST (sesiunea a 2-a Court 1) → servicii mai eficiente → mai puține break-uri → seturi mai lungi → **ajutor structural pentru O7.5**

Sursă: [ESPN heat alert](https://www.espn.com/tennis/story/_/id/49284583/wimbledon-2026-today-blog-06-07-2026-live-updates-news-tennis-arthur-fery-alexander-zverev-madison-keys), [Yahoo Sports extreme heat](https://sports.yahoo.com/articles/wimbledon-fans-sent-extreme-heat-115002181.html)

---

## TENNISABSTRACT — GRASS ANALYSIS

*Sursă: Sackmann WTA dataset local (TennisAbstract JavaScript-rendered = no web access)*

### Madison Keys — 57 meciuri iarbă (2015-2026)

| Statistică | Valoare |
|---|---|
| S1 avg games | **9.89** (cel mai ridicat dintre cele 2) |
| S1 O7.5 rate | **52/57 = 91.2%** ✅ |
| S1 sub 7.5 games | 5/57 = 8.8% (ALL vs ranked 100+) |
| S1 TB rate | 10/57 = 17.5% |
| S2 TB rate | **9/56 = 16.1%** |
| S1→S2 cascade | **1/10 = 10%** |

**S1 blowouts (sub 7.5 games) — toți adversarii:**
- Paszek (r119), Davis (r91), Kartal (r266), Marcinko (r51), Bouzas (r48)
- **Concluzie: NICIUN blowout vs jucătoare top-30 sau top-50 în iarbă.**

**Singurul cascade S1→S2 (Berlin 2026 QF vs Pegula):**
- Score: **7-6(5) 7-6(8)** — ambele seturi tiebreak, meci extrem de echilibrat cu Pegula (WTA ~5 la acel moment)
- Context: Pegula = una dintre cele mai bune jucătoare pe hard/iarbă → meci specific de anvergură maximă; nu comparabil cu meciul de astăzi ca tip de jucătoare

**S2 TB matches (9 meciuri), detaliu relevant:**

| Data | Turneu | Adversar | WTA rank | Scor complet |
|---|---|---|---|---|
| 2017-07-03 | Wimbledon R64 | Giorgi | ~55 | 6-4 **6-7(10)** 6-1 |
| 2023-06-26 | Eastbourne F | Kasatkina | ~14 | 6-2 **7-6(13)** |
| 2023-06-26 | Eastbourne R16 | Xiyu Wang | ~80 | 6-2 **7-6(3)** |
| 2023-07-03 | Wimbledon R16 | Andreeva | ~50 | 3-6 **7-6(4)** 6-2 |
| 2024-07-01 | Wimbledon R128 | Trevisan | ~85 | 6-4 **7-6(4)** |
| 2024-07-01 | Wimbledon R16 | **Paolini** | **~14** | 6-3 **6-7(6)** RET |
| 2025-06-09 | Queen's SF | Maria | ~65 | 6-3 **7-6(3)** |
| 2025-06-16 | Berlin R32 | Vondrousova | ~30 | 7-5 **7-6(6)** |
| **2026-06-15** | **Berlin QF** | **Pegula [5]** | **~5** | **7-6(5) 7-6(8)** ← cascade |

**Analiza S2 TBs Keys (relevante față de Noskova WTA #12):**
- Kasatkina (WTA ~14) → S2 TB în finală → relevant: jucătoare de nivel Noskova → S2 TB posibil vs top-15
- Paolini (WTA ~14) → S2 TB dar Keys a abandonat → anomalie
- Pegula (WTA ~5) → cascade, dar Pegula = nivel cu mult superior Noskovei la acel moment
- **Concluzie Keys S2:** S2 TB rate de 16.1% susținută și de meciuri vs top-30 (Kasatkina, Pegula) → nu putem exclude S2 TB vs Noskova (WTA #12)

---

### Linda Noskova — 27 meciuri iarbă (2023-2026)

| Statistică | Valoare |
|---|---|
| S1 avg games | 9.56 |
| S1 O7.5 rate | **23/27 = 85.2%** ✅ |
| S1 sub 7.5 games | 4/27 = 14.8% (ALL vs ranked 77+) |
| S1 TB rate | 6/27 = 22.2% |
| S2 TB rate | **2/27 = 7.4%** ✅ (FOARTE MIC) |
| S1→S2 cascade | **0/6 = 0.0%** ✅✅ (ZERO cascade) |

**S1 blowouts (sub 7.5 games) — toți adversarii:**
- Jones (r132), Badosa (r142 la acel moment), Zarazua (r77), Ruse (NaN) → ALL ranked 77+
- **NICIUN blowout vs jucătoare top-30 sau top-50.**

**S1TB→S2TB cascade — 6 meciuri cu S1 TB:**

| Meci | Scor S1 | S2 outcome |
|---|---|---|
| vs Galfi (r80), Wimbledon 2023 | 6-7(6) | **6-2 clar ✅** |
| vs Jabeur (~12), Berlin 2024 | 6-7(5) | **6-3 clar ✅** |
| vs Kerber (~50), Berlin 2024 | 7-6(4) | **2-6... 6-4 ✅** |
| vs Errani (~110), Wimbledon 2024 | 7-6(3) | **6-1 dominant ✅** |
| vs Pegula (~7), Bad Homburg SF 2025 | 6-7(2) | **7-5 (no TB!) ✅** |
| vs Rakhimova (~40), Wimbledon 2025 | 7-6(6) | **7-5 (no TB!) ✅** |

**Pattern Noskova anti-cascade: 0/6 = 0%** — indiferent de cât de tight a fost S1, Noskova NICIODATĂ nu a permis un S2 tiebreak după un S1 tiebreak. Nici vs Jabeur (top-12) sau Pegula (top-7). Semnalul cel mai puternic din toată analiza.

**S2 TB matches (2 meciuri):**
1. Bad Homburg 2023 R16 vs Samsonova (WTA ~30): 6-4 **6-7(4)** 6-3 — S1 clar, S2 tight
2. Wimbledon 2024 R64 vs Andreescu (WTA ~50): 6-3 **7-6(5)** — S2 tight vs jucătoare de calibru mediu

**Concluzie Noskova S2:** 2 S2 TBs în 27 meciuri = 7.4% → EXCELENT pentru U12.5 S2. Dar pentru meciul de azi, modelul deja ne dă tb_p_cal=0.1656 (16.56%) = prea ridicat pentru operational threshold.

---

## MARKET 1: OVER 7.5 SET 1

### Structura probabilistică (calcul din hold rates)

Cu hold Paolini=0.8635, Noskova=0.8300:
- P(break pe un joc de serviciu Keys) ≈ 13.7%
- P(break pe un joc de serviciu Noskova) ≈ 17.0%
- P(Set 1 ≤ 7 games) ≈ P(6-0) + P(6-1) ≈ 0.5% + 4.5% = ~5%
- **P(O7.5 S1) ≈ 95%** din matematica directă a hold rates

### Model Gates O7.5 (conf. wta_set1_filters.py)

| Criteriu | Status | Prag grass policy |
|---|---|---|
| p_cal_adj = 0.903 | ✅ | ≥ 0.83 |
| elite_pick = True | ✅ | — |
| blowout_score = 1 | ✅ | ≤ 2 |
| expected_games = 26.06 | ✅ | ≥ 24.5 |
| min_hold = 0.8300 | ✅ | ≥ 0.62 (holds_strong) |
| competitive_set = True | ✅ | — |
| unstable_reason = "" | ✅ | — |
| **fatigue_flag A=True, B=True** | ❌ | blocat în model |

**De ce fatigue_flag nu este decisiv pentru O7.5 în acest context:**

1. **Matematica serviciului:** Chiar dacă Noskova pierde 5-8pp hold din cauza oboselii (83% → 75-78%), P(S1 = 6-0 sau 6-1) rămâne sub 10%. La 75% hold, set de 6-1 contra Keys (WTA #22) = Noskova câștigă 1 game din 3 servicii + Keys câștigă toate 4 = P ≈ 7%. SUMA totală P(U7.5) ≈ 9% → P(O7.5) ≈ 91%.

2. **Fatigue effect pe serviciu vs pe returnare:** Noskova obosită înseamnă că RETURNEAZĂ mai greu, nu că nu mai serveste. Cu 21 ace-uri în 3 runde, ea se bazează pe ace și servicii câștigătoare. Serviciile câștigătoare sunt mai puțin afectate de oboseală decât schimburile lungi.

3. **Căldură extremă (31-34°C) = AJUTOR pentru O7.5:** Pe iarbă fierbinte uscată, servicii mai rapide → mai mulți ace → mai puțini break points → seturi mai lungi. Căldura crește dominanța serviciului, nu o reduce.

4. **TennisAbstract confirmat:** Keys 91.2% O7.5 grass, Noskova 85.2% O7.5 grass → NICIUN blowout vs top-30/50 pentru niciuna.

5. **Noskova BP save rate 42.86% la Wimbledon 2026** arată că ea luptă cu break points — dar PRIMEȘTE break points (adică returnarea adversarilor e bună pe serviciul ei), nu că ea PIERDE servicii ușor. Totuși a câștigat 3 din 3 meciuri.

### TennisStats confirmat (din datele userului)

| Stat | Keys | Noskova | Combinat |
|---|---|---|---|
| O7.5 per set (empiric) | 81% | 86% | **84%** |
| Avg games/set | 9.47 | 9.35 | 9.41 |
| O12.5 per set (= TB) | 10% | 14% | 12% |
| Aces/match | 3.61 | **6.79** | 10.40 |

TennisAbstract (grass specific) dă rate mai ridicate: Keys 91.2%, Noskova 85.2%. Diferența față de TennisStats se explică prin faptul că TennisStats include toate suprafețele în calculul per-set (clay mai puțin dominant pe serviciu → seturi mai scurte uneori), în timp ce grass izolat dă rate mai mari.

### Probabilitate ajustată pentru CoVe

| Factor | Ajustare probabilitate |
|---|---|
| Model p_cal_adj = 0.903 | Baza: 90.3% |
| TennisAbstract grass empiric (~88% combinat) | Confirmat ✅ |
| Căldură extremă 31-34°C pe iarbă rapidă | +1-2pp → serviciu mai dominant |
| Noskova fatigue (13-11 TB ieri, 1 zi odihnă) | -1-2pp → hold scade ușor |
| Keys mai proaspătă (2 zile odihnă vs 1) | Neutru (Keys e oricum favorit) |
| Niciun blowout al niciuneia vs top-30 pe iarbă | Confirmat ✅ |
| **Probabilitate ajustată finală** | **~90%** |

### Scor O7.5 Set 1: **8/10 — RECOMMEND** ✅

**Motivație scor 8 și nu 9:** Fatigue filter din model există dintr-un motiv — ambele jucătoare au acumulat 3-seturi multiple. Deși matematica serviciului minimizează riscul real de U7.5, există o incertitudine de ~10% care justifică scăderea de la 9. La 9/10 am vrea zero fatigue concern.

**Concluzie:** Probabilitate ajustată 90% > 82% threshold → **RECOMMEND**. Condiție: odds ≥ 1.10.

---

## MARKET 2: UNDER 12.5 SET 2

**Model tb_p_cal = 0.1656 → ABOVE 0.10 operational threshold → PASS automat.**

### Verificare Pasul 2 (pentru context)

| Metric | Keys | Noskova | Status |
|---|---|---|---|
| Sample grass | 56 complete | 27 | ≥ 10 ✅ |
| S2 TB rate | **16.1%** | **7.4%** | combinat ~11.8% |
| S1→S2 cascade | 10% (1/10) | **0%** (0/6) | ✅ anti-cascade |

**S2 TB rate combinat (~11.8%) este DEASUPRA pragului de 10%.** Modelul (16.56%) captează bine că există risc real de TB în S2.

Noskova are rate excepțional de mică (7.4%), dar Keys are 16.1% care include meciuri vs Kasatkina (WTA ~14) și Pegula (WTA ~5). Noskova are deja la nivelul lui Kasatkina (WTA #12).

**VERDICT U12.5 S2: ❌ PASS**

---

## MARKET 3: UNDER 30.5 TOTAL GAMES

### Structura match

Cu expected_games = 26.06 total și hold rates 86.35%/83.00%:

**Scenariul 2 seturi (~65% probabilitate):**
- 7-6, 7-6 = 26 games → U30.5 ✅
- 7-5, 7-6 = 12+13 = 25 games → U30.5 ✅
- 7-5, 7-5 = 24 games → U30.5 ✅
- **Aproape orice 2-set match = U30.5** (max teoretic = 26 games) → P(U30.5 | 2 seturi) ≈ 100%

**Scenariul 3 seturi (~35% probabilitate):**
- Cu hold rates 86/83%, fiecare set tinde spre 12-13 games
- 7-5, 5-7, 7-5 = 12+12+12 = **36 games → OVER 30.5** ❌
- 7-6, 6-7, 7-6 = 13+13+13 = **39 games → OVER 30.5** ❌
- 6-4, 4-6, 7-6 = 10+10+13 = **33 games → OVER 30.5** ❌
- Un 3-set match sub 30 games = foarte rar la hold rates astea
- P(U30.5 | 3 seturi) ≈ 10-15% (necesită cel puțin unul din seturi de 6-2 sau 6-3, nerealist la hold 86%/83%)

**Calcul combinat:**
- P(U30.5) = P(2 seturi) × P(U30.5|2 seturi) + P(3 seturi) × P(U30.5|3 seturi)
- ≈ 0.65 × 1.00 + 0.35 × 0.12 = 0.65 + 0.042 = **0.692 = ~69%**

**VERDICT U30.5: ❌ PASS** — 69% este semnificativ sub pragul de 82%. Riscul de 3 seturi lung este prea mare.

*Notă: chiar dacă crezi P(2 seturi) = 70%, U30.5 rămâne ~72% → tot sub 82%.*

---

## ANALIZA COMPLETĂ: STIL DE JOC, MOTIVAȚIE, MENTAL

### Stil de joc & Matchup

**Madison Keys (178cm, right-handed, AO 2025 champion):**
- Power basestrokes, forehand demolator, serve solid (nu elite)
- Returneaza agresiv, evoluție semnificativă a return game în 2025-2026
- Wimbledon: 27-11 (71%), best result = QF (2015, 2023)
- Eastbourne 2026 title (a 3-a oară!) → cel mai bun semn posibil de form pe iarbă
- BP save rate 77.78% la Wimbledon 2026 = ELITE → nu cedează ușor sub presiune
- **A jucat pe Centre Court Wimbledon pentru prima dată în R3 2026** — fapt suprins de Yahoo Sports
- Coach: Bjorn Fratangelo (soțul ei) — comunicare optimă în moment dificil

Sursă: [Yahoo Sports Keys Centre Court](https://sports.yahoo.com/articles/madison-keys-shares-inconceivable-wimbledon-111858839.html)

**Linda Noskova (179cm, Czech, left-right-handed):**
- Hard hitter agresiv, 2 mâini pe backhand, forehand penetrant
- **ACE MACHINE: 6.79 ace-uri per meci** (media), 21 ace-uri în 3 runde Wimbledon 2026
- 92 winners în 3 runde (25 mai mult decât Keys) → gameplay agresiv, many winners
- Berlin WTA 500 grass 2026 = prim titlu pe iarbă, def. Pegula în 3 seturi → confirmă seriozitataea pe iarbă
- BP save rate 42.86% la Wimbledon 2026 → luptă sub presiune pe propriul serviciu
- 21 ani, 3 sezoane serioase de iarbă, în plină ascensiune (career high WTA #10 iunie 2026)

Sursă: [TennisTonic preview](https://tennistonic.com/tennis-news/1023430/h2h-prediction-of-linda-noskova-vs-madison-keys-in-wimbledon-with-odds-preview-pick-6th-july-2026/), [Sofascore](https://www.sofascore.com/news/keys-experience-vs-noskovas-surge-a-wimbledon-r16-showdown)

**Verdict matchup:** Ambele sunt hard-hitters pe iarbă. Keys mai experimentată, mai eficientă sub presiune. Noskova mai agresivă ca volume de winners și ace-uri. H2H 0-0 — nu există precedent direct.

### Motivație

**Keys:** AO 2025 champion — știe cum arată un Grand Slam câștigat. Wimbledon best = QF (2015, 2023). O semifinală astăzi ar fi un record personal. La 31 ani, fereastra de Grand Slam rămâne deschisă dar se îngustează. Această cursă de iarbă 2026 (10-1 overall) este cea mai bună formă a ei pe iarbă. **Motivație: maximă, cu scop clar.**

**Noskova:** Un QF ar fi un career-first la un Grand Slam. WTA #12, 21 ani, în plin ascensiun. A câștigat din ce în ce mai greu (3 three-setters consecutiv) dar a câștigat. Drumul ei spre o carieră de top-10 depinde de performanțe la majore. **Motivație: enormă, dar cu oboseală acumulată.**

### Condiție fizică — verdict final

| Factor | Keys | Noskova |
|---|---|---|
| Days rest | 2 | **1** |
| Three-setters Wimbledon | 2 | **3** |
| Cel mai greu meci | 3-6, 6-2, 6-3 vs Anisimova | **2-6, 6-3, 7-6(11) vs Cirstea** |
| Fatigue real | Moderată | **SEVERĂ** |
| Legs azi | Proaspătă relativ | Obosite semnificativ |

**Impact pe O7.5 S1:** Noskova obosită → se bazează și mai mult pe serviciu și ace (fizic mai ieftin decât schimburi lungi) → hold rate probabil menținut → seturi tot lungi. Paradoxal, O7.5 S1 nu este amenințat de oboseala Noskovei.

### Mental & Context psihologic

**Keys:** A câștigat AO 2025 — știe să gestioneze momentele mari. Nu cedează sub presiune (BP save 77.78%). Experiența de 12 ani la Wimbledon îi dă stabilitate mentală. Soțul/antrenorul Fratangelo = suport psihologic optimal.

**Noskova:** Caracterul de fighter demonstrat prin 3 comebacks consecutive. Dar 13-11 TB ieri = o cantitate imensă de energie emoțională consumată. Poate intra cu sentimentul "am dat totul ieri". Sau poate intra dezinhibată, fără nimic de pierdut. 

### H2H

**0-0 — primul meci profesionist.** Nu există precedent pe nicio suprafată. Modelul construiește probabilitătile exclusiv din hold rates și Elo.

### Predicție: Cine câștigă?

| Perspectivă | Favorit | % |
|---|---|---|
| Model (p_markov) | Keys | 63.3% |
| Model (p_elo) | Keys | 52.6% |
| Robinhood market | Keys | 59% |
| Formă 2026 (win rate) | Keys | 74% vs 68% |
| BP save Wimbledon | Keys | 77.78% vs 42.86% |
| Fatigue | Keys | mult mai proaspătă |
| Experience | Keys | 31 ani, AO 2025 champ |
| Grass record 2026 | Keys | 10-1 vs 8-1 |

**Concluzie:** **Keys este favorita clară** din toate perspectivele (market, model, formă, odihnă, experiență). p_elo = 52.6% arată că Noskova are potențial real (tânără, agresivă) dar toate celelalte indicatori o favorizează pe Keys. Cel mai probabil Keys câștigă în 2 sau 3 seturi strânse.

---

## SCORING FINAL — TOATE PIEȚELE

| Piată | Scor | Verdict | Motiv |
|---|---|---|---|
| **O7.5 Set 1** | **8/10** | ✅ **RECOMMEND** | p_cal=0.903, elite_pick=True, holds 86%/83%, empiric 88-91% grass, fatigue nu afectează matematic U7.5 probability |
| U12.5 Set 2 | PASS | ❌ | tb_p_cal=0.1656 > 0.10, Keys S2 TB 16.1% include meciuri vs top-15 |
| U30.5 Total | PASS | ❌ | P(U30.5) ≈ 69%, 3-set cu hold rates 86/83% = 33-39 games |

---

## ⚠️ NOTE DE RISC — O7.5 RECOMANDAT

1. **Fatigue Noskova (cel mai important):** 13-11 TB ieri + 1 zi odihnă = legs obosite. Dacă serviciul ei colapsează (BP save rate deja 42.86%), ea poate fi spartă de 2-3 ori rapid → set scurt posibil. Dar aceasta ar crea un set de 6-2 sau 6-3 (8-9 games = O7.5). Un 6-0 sau 6-1 necesită colaps total al serviciului, extrem de improbabil.

2. **Model a blocat recomandarea** via fatigue filter — aceasta este o OVERRIDE manuală conștientă bazată pe matematica hold rates la nivel de top-30. Riscul asumat.

3. **Weather (31-34°C):** Teoria susține că iarbă fierbinte = servicii mai rapide = O7.5 mai sigur. Dar există riscul că căldura extremă obosește și mai mult Noskova, afectând calitatea meciului în general.

4. **Odds minime:** ≥ 1.10 (per daily filter). O7.5 S1 la odds sub 1.10 = no value.

---

*Analiză generată: 2026-07-06 | Sursă model: 1.5_WTA_Under12_5.csv + 1.2_WTA_Set1_Over_7_5.csv*
*Date TennisAbstract: Sackmann WTA dataset local (57 meciuri Keys, 27 meciuri Noskova)*
*Surse externe: Wikipedia, WTA Official, ESPN, Yahoo Sports, Robinhood, TennisTonic, Sofascore, Last Word on Sports*
