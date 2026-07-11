# WTA CoVe — Under 12.5 Games Set 2
# Jasmine Paolini vs Alexandra Eala
# Wimbledon 2026 — R4 (Last 16) — Centre Court
# 15:30 BST | July 6, 2026
# Triple Filter Workflow v1.1

---

## MODEL DATA (1.5 + 1.2 CSV)

| Parametru | Valoare |
|---|---|
| tb_p_cal | **0.0824** ✅ (≤ 0.10) |
| blowout_score | **1** ✅ |
| UNSTABLE | **False** ✅ |
| p_hold_a (Paolini) | 0.7058 |
| p_hold_b (Eala) | 0.6613 |
| hold_asym | 0.0445 (very low = balanced) |
| p_markov (Paolini wins) | 0.6102 |
| p_elo (Paolini wins) | 0.5941 |
| Gap Elo/Markov | **1.6pp** ✅ |
| expected_games | 24.42 (sub 24.5 → elite_pick=False, marginal) |
| days_rest A/B | 2 / 2 |
| had_3sets_7d A/B | True / True |
| fatigue_flag A/B | True / True |

---

## PASUL 1 — CSV Model + Market Check

### 1.1 Filtre model
- tb_p_cal = 0.0824 ✅ (prag operațional ≤ 0.10)
- Gap Elo/Markov = |0.5941 − 0.6102| × 100 = **1.6pp** ✅ (≤ 35pp)
- p_elo ≠ 0.0 ✅
- UNSTABLE = False ✅

### 1.2 Robinhood Market Check
**Sursă:** [robinhood.com — paolini-vs-eala-jul-06-2026](https://robinhood.com/us/en/prediction-markets/tennis/events/paolini-vs-eala-jul-06-2026/)

- **P(Eala câștigă) = 61%**
- **P(Paolini câștigă) = 39%**

| Criteriu | Status |
|---|---|
| P(favorita) ≥ 60% | ✅ 61% Eala → zona moderată |
| Divergență market vs p_markov | ⚠️ MODEL = Paolini favorit 61% vs MARKET = Eala favorit 61% |
| Gap = \|0.6102 − 0.39\| × 100 | **22pp > 15pp → INVESTIGHEAZA** |

### 1.3 Investigare divergență

Modelul folosește hold rates istorice (Sackmann). Piața procesează în timp real:

| Factor explicativ | Detaliu |
|---|---|
| Swiatek upset R3 | Eala def. Swiatek (defending champion) **7-6(9) 6-2** — 84 minute |
| H2H recent | Eala def. Paolini Dubai 2026 **6-1, 7-6(5)** (hard court) |
| Grass record | Eala 72.7% career pe iarbă vs Paolini 51.6% |
| Formă 2026 | Eala 61.7% win rate vs Paolini 54.2% |
| Stângăcia pe iarbă | Left-hand serve = avantaj structural pe iarbă rapidă uscată |

**Concluzie:** Divergența este **explicată clar** de momentum curent + H2H + suprafața. Nu există indicații de injury sau informații ascunse. Modelul nu captează momentum-ul din ultimele 2 săptămâni.

→ **PASUL 1: PASS** ✅ (cu notă de avertizare divergență 22pp)

---

## PASUL 2 — TennisAbstract (Grass)

*Sursă: dataset local Sackmann (import TennisAbstract). Pagina web TA este JavaScript-rendered.*

### 2.1 Jasmine Paolini — Grass (19 meciuri complete)

| Statistică | Valoare |
|---|---|
| Total meciuri iarbă | 20 (19 complete) |
| S2 Tiebreaks | **4/19 = 21.1%** |
| S1 Tiebreaks | 3 |
| S1TB → S2TB cascade | **1/3 = 33.3%** ⚠️ |

**Detaliu meciuri cu S2 Tiebreak:**

| Data | Turneu | Adversar | WTA rank | Scor complet | Tip meci |
|---|---|---|---|---|---|
| 2023-07-03 | Wimbledon | Kvitova (pierdut) | **~8** | 6-4 **6-7(5)** 6-1 | Top-10 grass specialist, 2x Wimbledon winner |
| 2024-06-24 | Eastbourne | Boulter (câștigat) | **~28** | 6-1 **7-6(0)** | Jucătoare locală GB, home crowd advantage |
| 2024-07-01 | Wimbledon | Keys (câștigat→RET) | **~14** | 6-3 **6-7(6)** RET | Keys retrasă, meci incomplet |
| 2025-06-22 | Bad Homburg | Fernandez (câștigat) | **~22** | **7-6(8) 7-6(6)** | Double TB — SINGURUL CASCADE S1→S2 |

**Analiza meciurilor S2 TB:**

- **Kvitova 2023**: Kvitova (WTA ~8) era la vârful formei pe iarbă, 2x Wimbledon champion. TB S2 natural în context high-level. Paolini pierdut meciul — nu aplicabil motivațional pentru astăzi.
- **Boulter 2024**: Jucătoare britanică (WTA ~28) în fața publicului de acasă. Crowd factor + grass specialist local. Paolini a câștigat S1 facil (6-1), S2 strâns. Nu comparabil cu Eala.
- **Keys 2024**: Keys (WTA ~14) a abandonat. TB S2 urmat de abandon — meci anomalie. Nu reprezentativ.
- **Fernandez 2025 (cascade)**: WTA ~22, meci cu totul echilibrat pe Bad Homburg. Ambele seturi au mers la tiebreak lung (S1: 8-6 TB, S2: 6-4 TB). Context: Fernandez este jucătoare solidă dar cu ranking similar Ealei — cel mai relevant precedent. Dar Eala este semnificativ mai bună decât Fernandez ca formă actuală.

**S1TB→S2TB cascade Paolini (detaliu):**

| Meci S1 TB | Câștigat S1 TB? | S2 outcome |
|---|---|---|
| vs Andreescu: 7-6(4) 6-1 | DA | S2 = **6-1 (CLAR, no TB)** ✅ |
| vs Minnen: 7-6(5) 6-2 | DA | S2 = **6-2 (CLAR, no TB)** ✅ |
| vs Fernandez: 7-6(8) 7-6(6) | DA (barely, 8-6) | S2 = **7-6(6) (TB!)** ❌ |

Pattern: când Paolini câștigă TB S1 cu scor strâns (8-6 vs Fernandez), S2 poate fi la fel de tight. Când câștigă TB S1 clar (4-1 vs Andreescu, 5-1 vs Minnen), S2 se deschide.

---

### 2.2 Alexandra Eala — Grass (17 meciuri complete)

| Statistică | Valoare |
|---|---|
| Total meciuri iarbă | 17 |
| S2 Tiebreaks | **3/17 = 17.6%** |
| S1 Tiebreaks | 2 |
| S1TB → S2TB cascade | **1/2 = 50%** ⚠️ (N=2 — NEREPREZENTATIV) |

**Detaliu meciuri cu S2 Tiebreak:**

| Data | Turneu | Adversar | WTA rank | Scor complet | Tip meci |
|---|---|---|---|---|---|
| 2025-06-02 | Birmingham 125 | Fruhvirtova (pierdut) | **~80-90** | 7-5 **6-7(5)** 6-1 | Jucătoare junioară WTA 125 |
| 2025-06-16 | Nottingham | Todoni (câștigat) | **~115-120** | 6-3 **6-7(4)** 6-3 | Adversar mult inferior |
| 2025-06-23 | Eastbourne | Baptiste (câștigat) | **~110-120** | **6-7(1) 7-6(4)** 6-1 | CASCADE S1→S2, adversar inferior |

**Analiza critică Eala S2 TBs:**

TOATE cele 3 S2 TBs ale Ealei au fost **vs jucătoare WTA 80-120**, nu vs top-30/top-50. Nu există niciun precedent de S2 TB al Ealei contra unui adversar de nivelul Paolini (WTA 17).

Cel mai relevant precedent anti-cascade: **Eala vs Swiatek R3 Wimbledon 2026**: S1 = **7-6(9)** (epic 11-9 TB, câștigat de Eala), S2 = **6-2** (CLAR, NO TB). Eala a dominat S2 după câștigarea unui tiebreak extrem de lung contra celei mai bune jucătoare din lume. Aceasta este **semnalul cel mai relevant** pentru meciul de astăzi.

**Pattern Eala în 2026 pe iarbă vs adversari de calibru:**
- Queens Club, Berlin, Bad Homburg warm-up events: 3 meciuri, toate pierdute în straight sets (niciun S2 TB). Indică că jucătoarele mai bune au câștigat clar față de ea ÎNAINTE de Wimbledon.
- La Wimbledon 2026: formă în ascensiune, R1-R3 dominate.

---

### 2.3 Precedent H2H — Dubai 2026 (Hard)

**Eala def. Paolini 6-1, 7-6(5)** — singurul meci direct jucat vreodată.

- S1: Eala a dominat clar (6-1) — NO S2 relevant în acel set
- S2: **Tiebreak 7-5** — S2 TB în singurul lor H2H!
- Context: hard court rapid (Dubai), nu iarbă. Dar TB S2 există ca precedent.
- Paolini: a câștigat S2 până la 6-6 (a rezistat în S2 după 6-1 S1), dar a pierdut TB.

---

### 2.4 Rezumat Pasul 2

| Metric | Paolini | Eala | Combinat |
|---|---|---|---|
| Grass matches | 19 (complete) | 17 | — |
| S2 TB rate | 21.1% | 17.6% | **~19.4%** (zona 15-25%) |
| S1TB→S2TB cascade | **33.3% (1/3)** ⚠️ | 50% (N=2) | N=2 nereprezentativ |
| Adversari în S2 TBs | WTA 8/28/14/22 | WTA 80-120 (toți) | Eala TBs = nivel inferior |

→ **PASUL 2: PASS** ⚠️ (cu semnal negativ cascade Paolini la limita pragului)

---

## PASUL 3 — Context Manual

### 3.1 Fatigă

**Paolini:**
- R1 (Jun 30): **3 seturi, 2h21m** vs Montgomery (0-6, 6-4, 7-5) — epuizant, dar 6 zile în urmă
- R2 (Jul 2): vs Golubic 7-6, 6-4 — tiebreak S1, efort moderat
- R3 (Jul 4): vs Sakkari **6-1, 6-2** — ~60 min, recuperare completă
- Evaluare: **oboseală scăzută**. R3-ul ușor a resetat rezervele. [Sursă: WTA R3](https://www.wtatennis.com/tournaments/wimbledon/scores/LS72320394)

**Eala:**
- R1 (Jun 28): vs Zarazua 6-1, 6-2 — dominant, ușor
- R2 (Jul 1): vs Joint **3-6, 6-2, 6-0** — 3 seturi, comeback
- R3 (Jul 4): vs Swiatek **7-6(9), 6-2** — 84 min, epic 11-9 TB fizic și mental solicitant
- Evaluare: **oboseală moderată-ridicată**. Tiebreakul de 11-9 = +25 min extra față de normal = legs acumulate.
- [Sursă: Sky Sports](https://www.skysports.com/tennis/news/12110/13560408/wimbledon-iga-swiatek-suffers-third-round-exit-as-defending-womens-champion-beaten-by-alexandra-eala-in-huge-shock)

**Verdict fatigă:** Avantaj Paolini în proaspătă față de Eala, în ciuda ambelor cu fatigue_flag=True în model.

### 3.2 Motivație

**Paolini (ITA, 30 ani, seed 13):**
- Finalistă Wimbledon 2024 (pierdut vs Krejcikova în finală)
- Eliminată R2 în 2025 (vs Rakhimova — upset dureros)
- 2026 = cursă de reabilitare și revenire. Știe că poate câștiga un Grand Slam.
- R1 2026: a câștigat după ce a pierdut primul set 0-6 — mental de oțel.
- Cunoaște Centre Court din interior (finala 2024). Nu e prima dată.
- [Sursă: Wimbledon 2024](https://www.wimbledon.com/en_GB/news/articles/2024-07-11/paolini_edges_vekic_in_semifinal_thriller.html)

**Eala (PHI, 21 ani, seed 29):**
- Prima filipineză în top-30 WTA (career-high 29 în martie 2026)
- Prima jucătoare filipineză care elimină o deținătoare a titlului la Wimbledon (Swiatek, R3)
- Dacă bate Paolini = primul QF la Grand Slam vreodată pentru Filipine
- Presiune națională enormă: zeci de milioane de oameni urmăresc
- "This goes out to all the little girls" — declarație post-R3
- A câștigat WTA 125 Birmingham 2026 pe iarbă — confirmarea formei
- [Sursă: WTA News](https://www.wtatennis.com/news/4530668/eala-dethrones-swiatek-at-wimbledon-to-make-new-history-for-the-philippines), [Inquirer Sport](https://sports.inquirer.net/685132/alex-eala-targets-wimbledon-quarterfinals-more-history-for-the-philippines/amp)

**Verdict motivație:** Amândouă extrem de motivate. Eala are mai mult de câștigat existențial, dar și mai multă presiune emoțională.

### 3.3 Condiții

- **Court:** Centre Court, Wimbledon
- **Temperatură:** 28°C, soare, iarbă uscată rapidă
- **Vânt:** gusts 19 mph — afectează serviciul → avantaj Eala (stângace, poate controla direcția)
- **Umiditate:** 40-55% — normală, nu opresivă
- **Prima sesiune Centre Court** (15:30 BST = deschidere) → iarbă la maxim de viteză, bounce jos
- [Sursă: ESPN Wimbledon](https://www.espn.com/tennis/story/_/id/49155718/wimbledon-2026-today-order-play-daily-schedule-results-weather-forecast-how-watch)

**Implicații pentru U12.5:** Iarbă uscată rapidă la 28°C = servicii mai eficiente → mai puțini breaks → S2 mai probabil curată sau tiebreak (nu break mix). Modelul: hold Paolini 0.71, hold Eala 0.66 → confirmă că ambele vor ține serviciul frecvent.

### 3.4 Centre Court Factor

- Paolini a jucat finala Wimbledon 2024 pe Centre Court — știe dimensiunile arenei, acustica, viteza suprafeței
- Eala joacă pe Centre Court **pentru prima dată** în carieră
- Diferență de experiență relevanță: presiunea primului QF + prima apariție pe cel mai iconic teren din lume = factor psihologic semnificativ

### 3.5 Stil de joc & Matchup

**Paolini (165cm, right-handed):**
- Counter-puncher agresiv, forehand topspin greu semi-western
- Ia mingea timpuriu, preferă schimburi rapide din linia de fund
- A îmbunătățit semnificativ jocul la fileu (2024+)
- Servire = slăbiciune structurală (înălțime mică). Al doilea serviciu exploatabil.
- Resilience extraordinară (R1 2026: comeback de la 0-6 — istorică)
- [Sursă: Tennis Abstract blog](https://www.tennisabstract.com/blog/2024/11/21/jasmine-paolinis-high-wire-act/)

**Eala (175cm, left-handed):**
- Attacker de baza, groundstrokes flat penetrante, structurată să finalizeze rapid
- **Left-hand serve pe iarbă rapidă** = avantaj tactic major: slice wide pleacă din reachul lui Paolini, T serve intră în corpul ei
- Forehand semi-western cu profunzime și viteză excepționale
- Returnare agresivă: 68% puncte câștigate pe al 2-lea serviciu al Swiatekăi (R3)
- Short-point player: 74 puncte câștigate în rally-uri sub 4 lovituri vs Swiatek
- Psihologic: fearless, instinctivă pe scenele mari
- [Sursă: ESPN tactical analysis](https://www.espn.com/tennis/story/_/id/49277325/wimbledon-alex-eala-iga-swiatek-jasmine-paolini-filipina-player)

**Verdict matchup pentru S2:** Eala are avantaj de serviciu structural pe iarbă (stângace). Paolini are avantaj de experiență şi rezilienţă mentală. Meciul este echilibrat (hold_asym=0.0445 confirmat de model). Ambele jucătoare tind să câștige seturi clar (Paolini 25% straight sets, Eala 45%) — nu este un meci de tiebreak cronic.

### 3.6 Antrenori

| Jucătoare | Antrenor principal | Observație |
|---|---|---|
| Paolini | Danilo Pizzorno + Sara Errani (2026) | Errani = fost nr.1 dublu, tactician pe iarbă. Cunoaște stilul italian de grasscourt. |
| Eala | Joan Bosch + Toni Nadal (mentorat) | Bosch = coach principal, Toni Nadal = mentorat pentru mari ocazii |
| | [Sursă WTA](https://www.wtatennis.com/news/4417001/jasmine-paolini-adds-sara-errani-to-coaching-staff-ahead-of-2026-season) | [Sursă Sunday Guardian](https://sundayguardianlive.com/sports/who-is-alexandra-eala-filipino-tennis-star-trained-at-rafa-nadal-academy-stuns-iga-swiatek-in-wimbledon-2026-226980/) |

---

## PREDICȚIE: CINE CÂȘTIGĂ?

| Perspectivă | Favorit | % |
|---|---|---|
| Model (p_markov) | Paolini | 61% |
| Model (p_elo) | Paolini | 59% |
| Robinhood market | **Eala** | **61%** |
| Formă recentă 2026 | Eala | ~60% |
| Grass specialization | Eala | ~58% |
| Mental/Experience | Paolini | ~55% |

**Sinteză analist:** Meciul este extrem de echilibrat cu ușoară tendință spre Eala datorită formei și avantajului structural pe iarbă. Modelul are Paolini favorit bazat pe hold rates istorice, dar piața captează mai bine contextul actual.

**Scenariul cel mai probabil:** Eala câștigă S1 (left-hand serve + momentum), Paolini luptă în S2. Meci în 2-3 seturi, probabil hotărât prin 1-2 break-uri în fiecare set, nu tiebreaks. Eala câștigă 6-4/7-5 sau Paolini revine 3-6 6-4 6-4.

---

## SCORING FINAL — U12.5 SET 2

### Aplicarea scoring table (din CLAUDE.md v1.1)

| Criteriu evaluare | Status |
|---|---|
| tb_p_cal = 0.0824 ≤ 0.10 | ✅ Semnal primar |
| Elo/Markov gap = 1.6pp ≤ 35pp | ✅ Model consistent |
| p_elo ≠ 0 | ✅ |
| UNSTABLE = False | ✅ |
| Robinhood P(favorita) = 61% (60-74%) | ✅ Zonă moderată |
| Divergență market 22pp > 15pp | ⚠️ Explicată — continuă cu notă |
| Sample Paolini (19 meciuri) ≥ 10 | ✅ |
| Sample Eala (17 meciuri) ≥ 10 | ✅ |
| S2 TB rate Paolini 21.1% | ⚠️ Zona 15-25% (neutru) |
| S2 TB rate Eala 17.6% | ⚠️ Zona 15-25% (neutru) |
| S2 TB rate combinat ~19.4% | ⚠️ 15-25% → neutru |
| S1TB→S2TB Paolini = 33.3% (1/3) | ❌ **>33% → scoring table Row 4: max 6/10** |
| S1TB→S2TB Eala = 50% (N=2) | ⚠️ N=2 nereprezentativ, ignorat statistic |
| Fatigue Eala > Paolini | ⚠️ Moderată (Eala mai obosită) |
| Centre Court experience | ✅ Paolini avantaj |
| UNSTABLE flag | ✅ False |

### Scoring table aplicat:

```
| Pași OK, S2 TB 15-25%, S1→S2 20-33% | 8/10 |  ← baseline
| UNSTABLE flag SAU S1→S2 > 33%       | max 6/10 | ← se aplică (33.3% > 33%)
```

> **Notă marginality:** Paolini S1→S2 = 1/3 = 33.33% depășește pragul cu 0.33pp.  
> Contextul: sample S1TB = 3 meciuri → interval de confidență extrem de larg [1%, 87%].  
> Singurul cascade (Bad Homburg 2025 vs Fernandez, 7-6(8) 7-6(6)) a fost meci double-TB complet diferit de structura de astăzi.  
> Aplicăm regulile strict: **33.3% > 33% → max 6/10**.

### ⭐ SCOR FINAL: **6/10** (cap scoring table)

---

## ⚠️ ATENȚIONARE BACKTEST — GRASS

Per [`reference_u125_s2_backtest_surfaces.md`]:

| Scor CoVe (proxy) | Hit Rate Grass |
|---|---|
| 9/10 | **94.7%** |
| 8/10 | 88.1% |
| 7/10 | 82.4% (sub baseline 86.6%) |
| **6/10** | **~79% estimat** (semnificativ sub baseline) |

**Scor minim pe iarbă pentru recomandare = 9/10** (per `feedback_u125_score_minimum_per_surface.md`)

Scor actual: 6/10 = cu **3 puncte sub minimum**.

---

## VERDICT FINAL

### ❌ PASS — Nu recomandăm

**Motivație principală:** Paolini S1TB→S2TB = 33.3% (1/3) depășește pragul de 33% din scoring table → max 6/10. Pe iarbă, minimul necesar pentru recomandare este 9/10. Gap de 3 puncte față de minimum.

**Factori pozitivi (documentați):**
- tb_p_cal = 0.0824 — semnal model excelent
- Eala's S2 TBs au fost ALL vs WTA 80-120 (nivel inferior Paolini WTA 17)
- Anti-cascade precedent: Eala vs Swiatek (S1 TB 11-9 câștigat, S2 = 6-2 clar)
- Holds echilibrate (0.71/0.66) fără asimetrie — meci structurat
- H2H: singurul meci a avut S2 TB (Dubai 6-1, 7-6) — dar pe hard, diferit

**Factori negativi care blochează:**
- Paolini cascade 33.3% → pragul scoring table depășit
- Market divergenți 22pp (piața vede matchup diferit față de model)
- Eala mental fortitudine excepțională (anti-collapse precedent vs Swiatek)
- First Centre Court match pentru Eala = factor imprevizibil

**Dacă utilizatorul decide să parieze în ciuda PASS-ului (la propria discreție):**
- Maxim considerat: 6/10 → sub grass baseline → risc asumat
- Condiție minimă: odds ≥ 1.10 (per daily filter)
- TennisStats confirmat: U12.5 per set = ~88% Paolini / ~93% Eala → corelat cu model 8.24%

---

*Analiză generată: 2026-07-06 | Sursă model: 1.5_WTA_Under12_5.csv + 1.2_WTA_Set1_Over_7_5.csv*
*Date TennisAbstract: Sackmann WTA dataset local*
*Surse externe: WTA Official, Sky Sports, ESPN, Robinhood Prediction Markets, Olympics.com, Inquirer Sport*
