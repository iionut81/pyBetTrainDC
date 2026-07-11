# WTA Analysis — Professional CoVe
# Coco Gauff vs Jessica Pegula
# Wimbledon 2026 — QF — 7 iulie, Centre Court, 13:30 BST

---

## DATE MODEL (1.2_WTA_Set1_Over_7_5.csv + 1.5_WTA_Under12_5.csv — run 06-07-2026)

| Metric | Valoare |
|---|---|
| p_hold_a (Gauff, iarbă) | **0.839** |
| p_hold_b (Pegula, iarbă) | **0.8202** |
| hold_asym | **0.0188** (practic ZERO — echilibru perfect) |
| p_markov (Gauff câștigă) | 0.5816 (58.16%) |
| p_elo (Gauff câștigă) | 0.4858 (48.58%) |
| Gap Elo/Markov | **9.58pp** (|0.4858-0.5816|×100) |
| expected_games | **25.81** |
| p_cal_adj O7.5 S1 | **0.8954** |
| blowout_score | 1 |
| competitive_set | True |
| elite_pick | **True** ← confirmat |
| O7.5 signal model | **no** (blocat de fatigue_flag) |
| fatigue_flag_a / _b | True / True |
| last_3sets_a / _b | True / True |
| had_3sets_7d_a / _b | True / True |
| days_rest (model July 6) | 1/1 → actual July 7: **2/2** (ambele au jucat R4 pe 5 iulie) |
| tb_p_cal (U12.5 S2) | **0.1656** (> 0.10 — PASS Pasul 1) |
| UNSTABLE | False |

---

## CONTEXT MECI

Wimbledon 2026 QF, Centre Court — All-American battle.

| Factor | Coco Gauff (Seed 7) | Jessica Pegula (Seed 4) |
|---|---|---|
| WTA Rank | No. 7 | No. 4 |
| Vârstă | 22 ani | 32 ani |
| H2H overall | — | Conduce **5-3** |
| H2H iarbă | **Tied 1-1** | **Tied 1-1** |
| Ultimul duel iarbă | Eastbourne 2023: Gauff def. Pegula **6-3, 6-3** | Berlin 2024: Pegula def. Gauff **7-5, 7-6** |
| Best GS | US Open 2023 **campioane** | US Open 2024 **finalistă** |
| Wimbledon history | **Primul QF din carieră** ← "breaking new ground" | Al 2-lea QF Wimbledon (2023, 2026) |
| 2026 win rate | 73% (27/37) | **81.4%** (35/43) — cel mai bun sezon al carierei |
| Coach | Jean-Christophe Faurel (FRA) | Mark Knowles (former doubles No. 1 mondial) |
| CC appearances | 9 (experimentată) | **Prima apariție pe CC** |
| Comeback wins 2026 | **8** (lidera WTA Tour) | Solid |

Surse: [WTA QF preview](https://www.wtatennis.com/news/4531217/wimbledon-quarter-preview-osaka-muchova-gauff-pegula-bad-homburg-final-rematch-all-american-battle-in-store) · [WTA Gauff-Bencic QF](https://www.wtatennis.com/news/4531103/quarterfinals-at-last-gauff-edges-bencic-to-break-new-ground-at-wimbledon) · [Olympics.com Pegula-Jovic](https://www.olympics.com/en/news/wimbledon-2026-jessica-pegula-iva-jovic-fourth-round-tennis-results)

---

## DRUMUL SPRE QF — FACTORI FIZICI

### Coco Gauff

| Tur | Adversar | Scor | Detalii |
|---|---|---|---|
| R1 | Korpatsch | 6-2, 6-1 | 54 min, dominant |
| R2 | Sierra | 6-3, 3-6, 7-6(10) | 3 seturi, super-TB |
| R3 | Liu | 6-3, 6-7, 6-2 | 3 seturi |
| **R4** | **Bencic** | **4-6, 6-3, 6-4** | **3 seturi, 2h18m, finalizat aproape de curfew-ul de 23:00 (5 iulie)** |

- 3 din 4 meciuri în 3 seturi → oboseală cumulativă semnificativă
- R4 vs Bencic: **9 double faults** — cel mai prost meci al ei pe iarbă la DF
- Câștigă mereu din spatele scorului (8 comeback wins în 2026)

### Jessica Pegula

| Tur | Adversar | Scor | Detalii |
|---|---|---|---|
| R1 | Vidmanova | 7-5, 6-3 | Seturi drepte |
| R2 | Sorribes Tormo | 7-6, 6-1 | Seturi drepte |
| R3 | Bouzas Maneiro | 6-1, 6-3 | Dominant |
| **R4** | **Jovic** | **4-6, 6-3, 6-1** | **3 seturi, dar a dominat după primul set** |

- Fizic: "Everything feels pretty good, I can't complain. I am definitely very healthy right now." — Pegula, post-R4
- **0 DF** în R4 vs Jovic → serviciu impecabil în 2h11m
- Al 10-lea QF Grand Slam din carieră — știe cum să gestioneze presiunea

**Concluzie fatigue:** Ambele au jucat pe 5 iulie și au o zi de odihnă completă (6 iulie) înaintea QF de pe 7 iulie. Gauff a fost mai epuizată după R4 (meci lungit, târziu seara, 9 DF). Pegula intră mai proaspătă și mai sigură pe serviciu.

Surse: [Olympics.com Gauff-Bencic](https://www.olympics.com/en/news/wimbledon-2026-coco-gauff-belinda-bencic-fourth-round-results) · [TennisHead Pegula fitness](https://tennishead.net/jessica-pegula-provides-update-on-her-fitness-ahead-of-facing-coco-gauff-at-wimbledon/)

---

## CONDIȚII — 7 IULIE, CENTRE COURT

- Temperatură: **30-31°C** (căldură extremă, peak după-amiaza)
- Precipitații: < 5% (condiții uscate)
- Vânt: 8-31 km/h (gusts moderate)
- Umiditate: 40-55%
- **Centre Court = îngrijit, iarbă rapidă** → favorizează mingea plată a Pegulei

Surse: [TennisTemple matchpage](https://en.tennistemple.com/match/pegula-gauff-wimbledon-2026/9471352/) · [ESPN weather Wimbledon](https://www.espn.com/tennis/story/_/id/49155718/wimbledon-2026-today-order-play-daily-schedule-results-weather-forecast-how-watch)

---

## STILURI DE JOC PE IARBĂ

### Coco Gauff (175cm, 22 ani)
- Jucătoare agresivă de baselinie, top-spin puternic pe forehand
- Atlet elit (retrieval, mișcare), poate acoperi tot terenul
- Serviciu puternic pe prima, dar vulnerabilă pe a doua (5.28 DF/meci general)
- **Pe iarbă: EXCEPȚIE** — doar 2.94 DF/meci (44% sub medie generală) → serviciul mai curat pe iarbă
- Dezavantaj structural: top-spin-ul devine mai puțin eficient când adversarul returnează cu minge plată, joasă
- Lider mondial în comeback wins 2026 (8) → caracter de luptătoare

### Jessica Pegula (32 ani, minge plată)
- Lovitură plată, directă — ideal pentru iarbă unde mingea rămâne joasă
- Consistentă, face puține greșeli neforțate
- Returnează excelent — una din cele mai bune returnetiste din WTA
- Grass Elo advantage: **+80.6pp** față de Gauff (Tennis Abstract) — real avantaj structural
- Coach Knowles: fostul nr. 1 mondial la dublu (iarbă specialist)
- Fitness excelent în 2026 ("last 6 months I became a better player")

**Avantaj suprafață:** Pegula beneficiază mai mult de iarba rapidă Wimbledon.

Surse: [ESPN preview](https://www.espn.com/tennis/story/_/id/49287914/wimbledon-quarterfinals-preview-coco-gauff-belinda-bencic-jessica-pegula) · [Last Word on Sports QF analysis](https://lastwordonsports.com/tennis/2026/07/06/wimbledon-quarterfinal-predictions-gauff-pegula/)

---

## PIEȚE DE PREDICȚIE — ROBINHOOD CHECK

| Sursă | P(Pegula câștigă) | P(Gauff câștigă) |
|---|---|---|
| **Robinhood** | **62%** | 41% |
| TennisTemple community | 64.7% (271 voturi) | 35.3% |
| Model p_markov | 42% | **58%** |
| Model p_elo | 51% | 49% |

**Divergență Robinhood vs p_markov:** |58% Gauff model - 38% Gauff market| = **20pp** ← semnificativă!

**Investigare divergență (OBLIGATORIU la >15pp):**
- Pegula grass Elo: +80.6pp față de Gauff (Tennis Abstract) — modelul nu captează asta
- Pegula 2026 win rate: 81.4% (cel mai bun sezon al carierei) — neincorporat în Sackmann
- Gauff 9 DF în R4 — formă serviciu curentă slăbită
- H2H momentum: Pegula conduce 5-3, câștigat ultimul duel pe iarbă (Berlin 2024)
- **Concluzie:** Explicație clară. Piața este informată cu date 2026 reale care nu sunt în modelul WElo. Nu este injurie/surpriză ascunsă.

Surse: [Robinhood QF Match Market](https://robinhood.com/us/en/prediction-markets/tennis/events/jessica-pegula-vs-coco-gauff-quarter-finals-match-jul-07-2026/)

---

# PIAȚA 1: OVER 7.5 SET 1

## Pasul 1 — Model

| Check | Valoare | Status |
|---|---|---|
| p_cal_adj | **0.8954** | ✅ Puternic |
| elite_pick | **True** | ✅ |
| O7.5 model output | **no** | ❌ Blocat de fatigue |
| Gauff hold_a | **0.839** | ✅ Excepțional |
| Pegula hold_b | **0.8202** | ✅ Excepțional |

**Fatigue flag activ:** ambele jucătoare au last_3sets = True + had_3sets_7d = True.  
Cu days_rest = 2 și last_3sets = True: (2 ≤ 2 AND True) → fatigue_flag = True pentru ambele.

## Pasul 2 — TennisAbstract (iarbă, date Sackmann)

| Metric | Gauff (33 meciuri) | Pegula (40 meciuri) |
|---|---|---|
| S1 O7.5 rate | **87.9%** (29/33) | **92.5%** (37/40) |
| S1 avg total games | 9.70 | 9.70 |
| S1 TB rate | 15.2% (5/33) | 15.0% (6/40) |
| DF/meci iarbă | **2.94** (vs 5.28 overall → 44% mai puține!) | N/A |

**Ambele jucătoare sunt în top-5 WTA din punct de vedere al S1 O7.5 pe iarbă.**  
Nicio scurtare structurală a seturilor — media de 9.70 jocuri per set pentru amândouă.

## Pasul 3 — Justificare Override Fatigue

**Calculul matematic (riguros):**

| Scenariul | Hold rates | P(S1 ≤ 7 jocuri) |
|---|---|---|
| Fără ajustare fatigue | 0.839 / 0.8202 | ~2-3% |
| Cu ajustare -6pp fatigue | 0.779 / 0.760 | ~8-12% |
| Cu ajustare -8pp extremă | 0.759 / 0.740 | ~12-15% |

**Concluzie matematică:** P(U7.5 S1) = 8-15% în cel mai pesimist scenariu.  
P(O7.5 S1) = **85-92%** indiferent de oboseală.

**Override justificat** prin aceleași criterii ca Keys/Noskova:  
- Setul 6-0 sau 6-1 este esențialmente imposibil cu hold rates 76%+
- Ambele jucătoare au S1 O7.5 rate > 87% pe iarbă (date Sackmann)
- QF Grand Slam = motivație maximă → fiecare joc contează

## Ajustare Cercetare vs Model

| Factor | Ajustare p |
|---|---|
| Gauff S1 O7.5 grass 87.9% ✅ | +0pp |
| Pegula S1 O7.5 grass 92.5% ✅ | +0pp |
| Fatigue ambele jucătoare (3 seturi R4) | **-2pp** |
| Gauff 9 DF în R4 vs Bencic (serviciu instabil recent) | **-2pp** |
| Motivație maximă QF — primul QF Wimbledon pentru Gauff | +1pp |
| Condiții cald extrem (30°C+) → pot scurta ușor anumite seturi | -1pp |
| **Total ajustare netă** | **-4pp** |

**p_cal_adj model** = 0.8954  
**p_cal_adj final** = **0.8554 (85.5%)**

## Verdict O7.5 S1

**85.5% ≥ 82% → RECOMANDAT**

**Scor: 8/10**

Argumente principale:
1. Ambele jucătoare au S1 O7.5 > 87% pe iarbă (Sackmann 33-40 meciuri)
2. P(U7.5 S1) < 12% chiar și cu maximum de oboseală (hold rates 76%+)
3. Model p_cal_adj = 0.8954, elite_pick = True → fundamentul modelului solid
4. Medie S1 identică: 9.70 jocuri/set pentru ambele → nicio tendință de seturi scurte
5. Berlin 2024 (H2H iarbă): 7-5, 7-6 → ambele seturi O7.5 ✅

Riscuri reziduale:
- Gauff serviciu instabil recent (9 DF vs Bencic) → poate pierde serve rapid dacă apare presiunea
- 30°C+ → setul 1 de testare fizică; dacă una cedează complet = set sub 7
- Pegula TennisStats 2026 (toate suprafețe): O7.5 per set = 74% → ușor sub medie, dar pe iarbă specific = 92.5%

**La odds ≥ 1.10 → DA**

---

# PIAȚA 2: U12.5 SET 2

## PASUL 1 — CSV Model + Market Check

| Check | Valoare | Status |
|---|---|---|
| tb_p_cal ≤ 0.10 | **0.1656** | **❌ FAIL → SKIP** |
| Gap Elo/Markov | 9.58pp ≤ 35pp | ✅ |
| p_elo = 0.0 | Nu | ✅ |
| UNSTABLE flag | False | ✅ |
| Robinhood P(fav=Pegula) | 62% (60-74%) | ✅ (continuă dar cu notă) |
| Divergență market vs p_markov | **20pp** > 15pp | ⚠️ (investigat, explicație clară) |

**PASUL 1: FAIL** — tb_p_cal = 0.1656 > 0.10 → oprire automată. Nu se continuă.

## Context Suplimentar (informativ, nu modifică decizia)

**TennisAbstract iarbă — S2 TB profile:**

| Metric | Gauff (33 meciuri) | Pegula (40 meciuri) |
|---|---|---|
| S2 TB rate | **12.1%** (4/33) | **25.0%** (10/40) ← ridicat! |
| S1TB→S2TB cascade | **0.0%** (0/5) ← ZERO | **33.3%** (2/6) ← la prag cap |

**Pegula S2 TB pe iarbă — toate 10 meciurile cu TB în S2:**
Spre deosebire de Gauff (0 cascade), Pegula a intrat de 2 ori în cascadă S1TB→S2TB pe iarbă:
- s'Hertogenbosch 2024 vs Krunic (rank ~400): S1 7-6(3) → S2 6-7(3) ← cascade vs jucătoare de rang 400!
- Berlin 2026 vs Keys (rank ~28): S1 7-6(5) → S2 7-6(8) ← cascade vs top-30

Acest profil arată o tendință structurală la Pegula pe iarbă, nu un artefact al adversarilor de calitate.

**Divergență market (20pp):** Investigată și explicată (Pegula grass Elo +80.6pp, form 2026, H2H). Nu este semnal ascuns.

**Concluzie U12.5 S2:** Modelul a blocat corect. Pegula's 25% S2 TB rate pe iarbă este un risc structural real. tb_p_cal = 16.56% reflectă acest risc calculat prin Markov.

## Verdict U12.5 S2: ❌ PASS (Pasul 1 fail, tb_p_cal > 0.10)

---

# PIAȚA 3: U30.5 TOTAL GAMES

## Analiza

| Factor | Valoare | Impact |
|---|---|---|
| expected_games (2 seturi) | 25.81 → **U30.5** ✅ | Puternic pozitiv dacă 2 seturi |
| H2H iarbă (N=2) | Eastbourne 2023: 6-3, 6-3 = **18 jocuri** / Berlin 2024: 7-5, 7-6 = **25 jocuri** | Ambele MULT sub 30.5 |
| H2H total (8 meciuri) | 6/8 = **75% în 2 seturi** | Istoric favorabil |
| P(2 seturi) estimat | **~70%** (H2H + hold rates) | — |
| P(3 seturi) estimat | **~30%** | Risc semnificativ |

**Calculul combinat:**

| Scenariu | P(scenariu) | P(U30.5 \| scenariu) | Contribuție |
|---|---|---|---|
| 2 seturi | 70% | ~98% (max 26-27 jocuri în 2 seturi cu hold ≥ 83%) | 0.686 |
| 3 seturi | 30% | ~27% (3 seturi cu hold ≥ 83% = frecvent 32-38 jocuri) | 0.081 |
| **P(U30.5) total** | — | — | **~0.767 = 77%** |

77% < 82% → PASS.

**Riscul principal:** 3 seturi cu hold rates 83%+ generează frecvent scoruri ca 7-5+5-7+7-5 = 36 jocuri sau 6-4+4-6+7-6 = 33 jocuri — ambele OVER 30.5. Numai scorurile tip 6-3+3-6+6-3 (27 jocuri) sunt sub 30.5 în 3 seturi.

## Verdict U30.5: ❌ PASS (~77%, sub pragul de 82%)

---

# MOTIVAȚIE, PSIHOLOGIE, PREDICȚIE JOC

## Context psihologic

**Coco Gauff:**
- La 22 ani, primul QF Wimbledon din carieră → presiune suplimentară dar și liberare
- Mentă de luptătoare cu 8 comeback wins în 2026 — cel mai bun indicator psihologic din WTA
- Știe cum să câștige Grand Slam (US Open 2023) → nu e copleșită de moment
- Centre Court: 9 apariții — teren familiar, nu intimidant
- Gauff: "She has too much experience on the big stages" (referitor la Pegula) → respect și vigilență

**Jessica Pegula:**
- La 32 ani, în cel mai bun sezon al carierei → sentimentul că acum e momentul
- Never won a Grand Slam → SF ar fi un milestone major
- Prima apariție pe Centre Court → poate fi factor de presiune sau adrenalină
- Descrisă ca "measured, experienced" — emotiv controlată
- Pegula: "Whatever kind of happens happens...I just know where I am" → mental stabil

## Previziune scor

**Scenariu cel mai probabil (60%):** Pegula câștigă în 2 seturi, 7-5 sau 6-4 la primul set, 7-6 sau 7-5 la al doilea. Similar pattern Berlin 2024.

**Scenariu alternativ (30%):** Gauff revine din 0-1 seturi → 3 seturi. Gauff este liderul la comebacks → dacă pierde primul set, nu abandoneaza.

**Scenariu șoc (10%):** Gauff câștigă în 2 seturi dacă serviciul funcționează la nivel obișnuit pe iarbă (2.94 DF/meci medie) și Pegula are o zi mai puțin bună.

## Cine câștigă?

| Sursă | P(Pegula) | P(Gauff) |
|---|---|---|
| Robinhood market | **62%** | 41% |
| TennisTemple community | **64.7%** | 35.3% |
| Grass Elo (Tennis Abstract) | Pegula +80.6pp | — |
| H2H | Pegula **5-3** | — |
| H2H iarbă | 1-1 | 1-1 |
| 2026 win rate | **81.4%** | 73% |
| Coach calitate iarbă | Knowles (grass expert) | Faurel (clay-bred) |
| Fitness intrare QF | ✅ Proaspătă | ⚠️ 3 seturi la miezul nopții |

**Predicție:** **Pegula câștigă, ~62%**

Cel mai probabil scor: **Pegula 7-5, 7-6** (pattern Berlin 2024 repetat)  
Alternativă realistă: **Gauff 6-7, 6-4, 7-5** (comeback specialist)

---

# RECOMANDARE FINALĂ

| Piață | Model p | Research p | Ajustare | Scor | Decizie |
|---|---|---|---|---|---|
| **O7.5 Set 1** | 89.54% (elite_pick, fatigue blocat) | **85.5%** | -4pp (fatigue + DF) | **8/10** | ✅ **RECOMANDAT** |
| U12.5 Set 2 | 83.44% (tb_p_cal=16.56%) | PASS Pasul 1 | — | — | ❌ **PASS** |
| U30.5 Total | ~77% estimat | ~77% | — | — | ❌ **PASS** |

## Singura piață viabilă: Over 7.5 Set 1

**Probabilitate estimată: 85.5%**  
**Scor: 8/10**  
**Odds minim: ≥ 1.10**

**Nu recomanda dacă odds < 1.10** (lipsă de valoare).

---

*Analiză generată: 2026-07-07*  
*Model: 1.2_WTA_Set1_Over_7_5.csv (run 06-07-2026) + 1.5_WTA_Under12_5.csv*  
*Date TennisAbstract: Sackmann WTA dataset (iarbă) — Gauff 33 meciuri, Pegula 40 meciuri*  
*Surse context:*
- *[Robinhood QF Match Market](https://robinhood.com/us/en/prediction-markets/tennis/events/jessica-pegula-vs-coco-gauff-quarter-finals-match-jul-07-2026/)*
- *[WTA QF Preview](https://www.wtatennis.com/news/4531217/wimbledon-quarter-preview-osaka-muchova-gauff-pegula-bad-homburg-final-rematch-all-american-battle-in-store)*
- *[Olympics.com Gauff-Bencic R4](https://www.olympics.com/en/news/wimbledon-2026-coco-gauff-belinda-bencic-fourth-round-results)*
- *[Olympics.com Pegula-Jovic R4](https://www.olympics.com/en/news/wimbledon-2026-jessica-pegula-iva-jovic-fourth-round-tennis-results)*
- *[TennisHead Pegula fitness](https://tennishead.net/jessica-pegula-provides-update-on-her-fitness-ahead-of-facing-coco-gauff-at-wimbledon/)*
- *[WTA Berlin 2024 scoreboard](https://www.wtatennis.com/tournaments/2012/berlin/2024/scores/LS002)*
- *[TennisTemple matchpage](https://en.tennistemple.com/match/pegula-gauff-wimbledon-2026/9471352/)*
- *[Last Word on Sports QF analysis](https://lastwordonsports.com/tennis/2026/07/06/wimbledon-quarterfinal-predictions-gauff-pegula/)*
- *[Roland Garros on Faurel](https://www.rolandgarros.com/en-us/article/rg2025-coach-faurel-driven-by-developing-gauff)*
- *[Sportskeeda on Knowles](https://www.sportskeeda.com/tennis/jessica-pegula-tennis-coach)*
- *[Gauff 2026 season Wikipedia](https://en.wikipedia.org/wiki/2026_Coco_Gauff_tennis_season)*
