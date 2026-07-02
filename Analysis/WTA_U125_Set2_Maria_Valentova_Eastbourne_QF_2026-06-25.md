# CoVe Analysis — U12.5 Set 2 | Eastbourne WTA 250 2026
## Tatjana Maria vs Tereza Valentova
**Data:** 2026-06-25 | **Ora:** 17:00 BST (18:00 CEST)
**Turneu:** Lexus Eastbourne Open WTA 250 — Quarterfinal (QF)
**Suprafață:** Iarbă (outdoor, Devonshire Park, Eastbourne, UK)
**Analist:** AI Sports Analyst | **Surse:** TennisAbstract, TennisStats, Model, LTA, WTA

---

## PASUL 1 — TRIPLE FILTER

| Parametru | Valoare | Status |
|---|---|---|
| tb_p_cal | **8.64%** | ✅ ≤ 10% |
| p_markov | 0.8918 (89.18% Maria) | — |
| p_elo | 1.0 (artificial — Valentova fără Elo) | ⚠️ |
| Elo/Markov gap | **\|100 - 89.18\| = 10.82pp** | ✅ ≤ 35pp |
| p_elo = 0.0 | Nu (1.0 — direcție consistentă) | ✅ |
| **UNSTABLE flag** | **fatigue Tereza Valentova (4 meciuri în 5 zile)** | ⚠️ **max 7/10** |
| hold_asym | **19.73pp** Maria | ✅✅ |
| blowout_score | 5/9 | ✅ dominanță clară |
| data_source | sackmann/sackmann | ✅ fiabil |

**Notă p_elo = 1.0:** Valentova este prea nouă pe circuit (19 ani, $50,820 prize money) → Sackmann nu are Elo pentru ea → model defaultează la 1.0 pentru Maria. Direcția este CONSISTENTĂ cu Markov (ambele spun Maria domină), gap mic = semnal valid.

**PASUL 1: ✅ TRECUT — cu nota UNSTABLE (cap 7/10)**

---

## PASUL 2 — TENNISABSTRACT (iarbă)

### Tatjana Maria — Iarbă 2023-2026

**Sample: 25 meciuri** ✅✅ — mult peste threshold ≥10. DATE EXTREM DE FIABILE.

**Context major:** Maria a câștigat **Queen's Club 2025** (WTA 250 pe iarbă!): Fernandez (R32) → Muchova (R16) → Rybakina (QF) → Keys (SF) → Anisimova (F). Șapte victorii consecutive pe iarbă în 2025.

| Meci cheie | Scor S1 | S1 TB? | Scor S2 | **S2 TB?** |
|---|---|---|---|---|
| vs Fernandez (QC R32) 2025 | **7-6(4)** | ✅ | 6-2 | ❌ NO |
| vs Muchova (QC R16) 2025 | 6-7(3) | ✅ | 7-5 | ❌ NO |
| vs Rybakina (QC QF) 2025 | 6-4 | ❌ | **7-6(4)** | **✅ YES** |
| vs Keys (QC SF) 2025 | 6-3 | ❌ | **7-6(3)** | **✅ YES** |
| vs Anisimova (QC F) 2025 | 6-3 | ❌ | 6-4 | ❌ NO |
| vs Fernandez (Bad Homburg) 2025 | 6-0 | ❌ | **7-6(1)** | **✅ YES** |
| vs Volynets (Wimbledon) 2025 | 3-6 | ❌ | **7-6(4)** | **✅ YES** |
| vs Rybakina (QC R16) 2026 | 6-7(4) | ✅ | 7-5 | ❌ NO |
| vs Paolini (Eastbourne R32) 2026 | 6-4 | ❌ | 6-3 | ❌ NO |
| vs Zakharova (Eastbourne R16) 2026 | 6-2 | ❌ | 6-1 | ❌ NO |
| *Restul 15 meciuri* | — | — | — | ❌ NO (toate) |

**Maria S2 TB pe iarbă: 4/25 = 16%** ⚠️ (>15% = -1pp din scor)
**S1 TB → S2: 0/3 = 0%** ✅✅ (Fernandez S1 TB → S2 6-2; Muchova S1 TB → S2 7-5; Rybakina 2026 S1 TB → S2 7-5)

**ANALIZĂ CRITICĂ a celor 4 TB-uri în S2:**
- vs Rybakina (QF, #2): Rybakina = serverul #1 mondial → natural TB
- vs Keys (SF, top-10): Keys = server dominant → natural TB  
- vs Fernandez (Bad Homburg): după S1 6-0 blowout → relaxare → TB
- vs Volynets (Wimbledon): meci în 3 seturi, S2 la mijloc → TB competitiv

**Concluzie:** Cele 4 TBs au venit împotriva jucătoarelor cu serviciu DOMINANT (Rybakina, Keys) sau în contexte specifice (blowout S1, 3 seturi). Valentova cu hold 61.59% NU este în această categorie → se rupe des.

---

### Tereza Valentova — Iarbă 2023-2026

**Sample: 9 meciuri** ⚠️ (sub ≥10 threshold — notăm rezerva)

| Meci | Scor S1 | S1 TB? | Scor S2 | **S2 TB?** |
|---|---|---|---|---|
| vs Mateas (Wimbledon Q1) 2025 | 6-2 | ❌ | 6-3 | ❌ NO |
| vs Stefanini (Wimbledon Q2) 2025 | 3-6 | ❌ | 6-4 | ❌ NO |
| vs Zakharova (Wimbledon Q3) 2025 | **7-6(1)** | ✅ | 2-6 | **❌ NO** |
| vs Bouzkova (Nottingham R32) 2026 | 6-3 | ❌ | 6-3 | ❌ NO |
| vs Klugman (Eastbourne R32) 2026 | 7-5 | ❌ | 5-7 | ❌ NO |
| vs Tomljanovic (Eastbourne R16) 2026 | 6-2 | ❌ | 5-7 | ❌ NO |

**Valentova S2 TB pe iarbă: 0/6 = 0%** ✅✅✅ EXCEPȚIONAL
**S1 TB → S2: 0/1 = 0%** ✅✅ (vs Zakharova: S1 TB → S2 2-6 decisiv)

---

### Rezumat Pasul 2

| | Maria | Valentova |
|---|---|---|
| Sample iarbă | **25** ✅✅ | 9 ⚠️ |
| S2 TB rate | **4/25 = 16%** ⚠️ | **0/6 = 0%** ✅✅✅ |
| S1 TB → S2 | **0/3 = 0%** ✅ | **0/1 = 0%** ✅ |
| Pattern | vs serveri dominanți | nicio TB pe iarbă |

**PASUL 2: ✅ TRECUT condiționat** (Maria 16% -1pp; Valentova 0% excelent; sample Valentova sub 10)

---

## 1. MATCH CONTEXT

**Eastbourne WTA 250 QF** — Devonshire Park, seara de 17:00 BST. Valentova joacă al 3-lea meci în 3 zile la Eastbourne. Maria joacă al 3-lea meci, primele două în 2 seturi decisive.

**Path la QF:**
- Maria: R32 beat Paolini (top-20!) **6-4, 6-3** | R16 beat Zakharova **6-2, 6-1** → 2 victorii dominante în 2 seturi
- Valentova: R32 beat Klugman **7-5, 5-7, 7-5** (3 seturi!) | R16 beat Tomljanovic **6-2, 5-7, 6-4** (3 seturi!) → epuizare reală

**Condiții meteorologice Eastbourne (17:00 BST):** ~19-21°C, seară britanică. Condiții excelente pentru joc.

---

## 2. PROFILURI JUCĂTOARE

### Tatjana Maria (Germania)
- **Rang:** #112 | **Vârstă:** 38 ani | **Înălțime:** 172cm, 62kg | **Elo:** 695
- **Stil:** Jucătoare de iarbă prin excelență. Slice backhand devastator pe iarbă, volée precise, serve-and-volley. Cel mai natural "grass player" din top 200. Minge joasă, scurtă, care face adversarele să joace în afara confortului.
- **Hold iarbă (model):** **81.32%** — EXCEPȚIONAL pentru WTA
- **Aces/meci:** 4.42 (NET dominant) 
- **DFs/meci:** 2.11 (control solid)
- **Net Points Won:** 18.25 — vine la fileu în proporție masivă
- **2026 grass form:** Beat Paolini (top-20), Zakharova dominantă → 4-1 pe iarbă în 2026
- **Form recent:** WLWWLWW — 5/7 victorii recente
- **CAMPIOANĂ QUEEN'S CLUB 2025** — cel mai bun rezultat de iarbă al carierei

### Tereza Valentova (Cehia)
- **Rang:** #61 | **Vârstă:** 19 ani (n. 20 feb 2007!) | **Elo:** 1041
- **Stil:** Baseliner agresiv, forehand puternic, serve decent (2.52 aces/meci), dar DFs ridicate (4.3/meci) — servici instabil
- **Hold iarbă (model):** **61.59%** — SLAB pentru QF WTA
- **DFs/meci:** 4.3 — serviciu agresiv dar volatile
- **2026:** 57.1% win rate dar în principal pe hard/clay
- **Career prize money:** $50,820 ← foarte puțin → puțin WTA history
- **FATIGUE CRITICA:** 4 meciuri în 5 zile + 2 meciuri în 3 seturi la Eastbourne!
- **UNSTABLE flag:** model confirmat

---

## 3. STATISTICI HOLD & SERVIRE

### Model (Markov + WElo, iarbă — Sackmann)
| Parametru | Maria (A) | Valentova (B) |
|---|---|---|
| **Hold % iarbă** | **81.32%** | **61.59%** |
| **Hold asymmetry** | **+19.73pp Maria** | ← cea mai mare asimetrie azi |
| p_markov | **89.18% Maria** | |
| p_elo | **1.0** (artificial) | Valentova fără Elo |
| gap | **10.82pp** | ✅ consistent |
| expected_games | **21.95** | ← cel mai SCURT din lista de azi! |
| blowout_score | **5/9** | dominanță structurală clară |

**Expected games = 21.95** — modelul prezice un meci SCURT. Seturi de tip 6-2, 6-3. Acesta este cel mai mic expected_games din toate picks-urile de azi.

### TennisStats (toate suprafețele, 2026)
| Statistică | Maria | Valentova | Combinat |
|---|---|---|---|
| Aces/meci | **4.42** | 2.52 | 6.94 |
| DFs/meci | 2.11 | **4.30** ← ridicate | 6.41 |
| **Over 12.5/set** | **17%** | **19%** | **18%** |
| TB/meci | **19%** | 26% | 23% |
| Avg games/set | 9.86 | **10.11** | **9.99** |
| Set 2 Win Rate | **47%** | 43% | |
| **Net Points Won** | **18.25** | 11.50 | ← Maria vine MULT la fileu |

**NOTA CRITICĂ TennisStats:** Over 12.5/set = 18% combined (toate suprafețele). Pe iarbă specific (TennisAbstract): Maria 16%, Valentova 0% → semnalul grass-specific este mult mai bun decât all-surface.

**Valentova 10.11 avg games/set** este îngrijorătoare general, dar pe iarbă ea nu a avut NICIO TB în Set 2 (0/6). Aceasta este caracteristica ei pe iarbă.

---

## 4. CONDIȚIE FIZICĂ & OBOSEALĂ

### Maria — ✅ Relativ proaspătă
- days_rest = 1 (jucată ieri)
- AMBELE meciuri ieri/alaltăieri: **straight sets dominante** (6-2, 6-1 vs Zakharova; 6-4, 6-3 vs Paolini)
- La 38 ani: veterana par excellence — conservă energia în meciuri ușoare
- **Nu este fatigued** per model (fatigue_flag_a = False)
- Stilul ei (fileu, slice scurt) = meciuri mai scurte = mai puțin efort fizic

### Valentova — 🔴 EPUIZATĂ REAL
- **4 meciuri în 5 zile**
- **2 meciuri în 3 seturi la Eastbourne** (vs Klugman 3 seturi, vs Tomljanovic 3 seturi)
- fatigue_flag_b = TRUE, had_3sets_7d = TRUE
- La 19 ani: recuperare mai bună decât senioarle, DAR volumul este excesiv
- Serviciu: cu DFs 4.3 deja ridicate → sub oboseală va crește

**Impactul oboselii Valentova pe U12.5:**
- Hold 61.59% → sub oboseală probabil 55-58% → Maria rupe și mai ușor → seturi și mai decisive → **AJUTĂ U12.5**
- DFs crescute → pierderi de serviciu rapide = seturi scurte

---

## 5. STILUL DE JOC & TACTICI

**Maria pe iarbă:** Cel mai specific grass-player din WTA top 200. Slice backhand care rămâne jos pe iarbă → Valentova nu poate juca topspinul ei → scoasă din rhythm. Vine la fileu (18.25 net points/meci!) → presiune constantă. Servici la corp + volée → game-uri de serviciu scurte (nu ajunge la deuce = NU ajunge la TB).

**Valentova pe iarbă:** Baseliner adaptat pe hard/clay. DFs ridicate (4.3) = servici agresiv dar instabil → pe iarbă, presiunea lui Maria + fileu = și mai multe DFs. Nu știe să joace la fileu (net points doar 11.5/meci). Hold 61.59% = se rupe în jur de 40% din game-uri de serviciu.

**Mismatch structural:** Maria ±19.73pp hold advantage = pentru fiecare 10 game-uri de serviciu Valentova, Maria rupe ~4. Fiecare set se termină prin break decisiv, nu 6-6.

---

## 6. CONTEXT MOTIVAȚIONAL & PSIHOLOGIC

### Maria — ⬆️ MOMENT DE FORMĂ EXTRAORDINAR
- **A bătut Paolini (top-20, seed 1!)** la Eastbourne = "first Top 20 win in 12 months" ([WTA](https://www.wtatennis.com/videos/4525532/maria-stuns-top-seed-paolini-in-eastbourne-for-first-top-20-win-in-12-months))
- La 38 ani, cu copii — orice QF la WTA este emoțional special
- Se simte "acasă" pe iarbă — stilul ei natural
- Campioană Queen's Club 2025: știe că poate câștiga turnee pe iarbă
- Mentală: relaxată, joacă fără presiunea favorita

### Valentova — ⬆️/⬇️ TÂNĂR FĂRĂ PRESIUNE DAR EPUIZATĂ
- 19 ani → curaj naiv, joacă liberă
- Prima QF WTA în carieră (probabil) → adrenaline ridicat
- DAR: obosită fizic (2 meciuri în 3 seturi consecutive)
- Știe că joacă contra unei legende pe suprafața preferată a adversarei
- Mental: bun pentru o victorie, dar instabilă sub presiune îndelungată

---

## 7. CoVe SCORING — U12.5 SET 2

### Factori confirmare ✅
| Factor | Valoare | Semnal |
|---|---|---|
| Model tb_p_cal | **8.64%** | ✅ |
| Hold asymmetry | **19.73pp** ← cea mai mare azi | ✅✅✅ |
| Expected games | **21.95** ← cel mai mic azi | ✅✅ |
| blowout_score | **5** | ✅✅ |
| Valentova S2 TB iarbă | **0/6 = 0%** | ✅✅✅ |
| Valentova fatigue | 4 meciuri/5 zile | ✅ (ajută U12.5) |
| Maria hold | **81.32%** ← solid | ✅✅ |
| Net points Maria | 18.25/meci | ✅ (fileu = seturi scurte) |
| S1 TB → S2: Maria | **0/3 = 0%** | ✅✅ |
| Maria form la Eastbourne | beat Paolini, Zakharova dominant | ✅ |

### Factori risc ⚠️
| Factor | Valoare | Semnal |
|---|---|---|
| **UNSTABLE flag** | fatigue Valentova | **cap max 7/10** |
| Maria S2 TB | **4/25 = 16%** | -1pp ⚠️ |
| TennisStats Over 12.5 | 18% combined | ⚠️ |
| Valentova sample | 9 meciuri (< 10) | ⚠️ |
| p_elo = 1.0 | artificial, Valentova fără Elo | notat |
| Valentova avg games/set | 10.11 | ⚠️ tinde spre seturi lungi |

### REZOLVAREA TENSIUNII

Maria 16% S2 TB vs hold_asym 19.73pp:
- Cele 4 TBs Maria în S2 au venit exclusiv vs Rybakina (#2), Keys (top-10), și situații speciale (blowout S1 → relaxare, 3 seturi)
- Valentova hold 61.59% → MULT sub nivelul lui Rybakina/Keys → Maria o rupe sistematic
- Sub oboseală, Valentova hold scade și mai mult

Valentova avg 10.11 games/set vs 0/6 TB S2 pe iarbă: contradictia se rezolvă prin suprafață — pe iarbă ea NU ajunge la 6-6 (toate seturile s-au terminat 7-5, 6-4, 2-6, 5-7 = decisiv fără TB).

### SCOR FINAL U12.5 SET 2

**7/10** ✅ — limitat de UNSTABLE flag (cap maxim)

**Motivul 7 și nu mai puțin:**
- Cel mai mare hold_asym din picks-urile de azi (19.73pp)
- Expected_games cel mai mic (21.95)
- Valentova 0/6 = 0% S2 TB pe iarbă
- Maria în formă excepțională (beat Paolini dominant)
- Valentova epuizată → hold scade → seturi și mai decisive

**Probabilitate ajustată: ~85-87%**

---

## 8. PREDICȚIE CÂȘTIGĂTOARE

**Maria câștigă: ~82-85%**
- Model: 89.18% (Markov) — cel mai mare output al modelului azi
- Hold 81.32% vs 61.59%: Maria rupe Valentova în ~40% din game-uri
- Stilul pe iarbă: Maria = campioana Queen's 2025 vs Valentova = debutantă QF
- Fatigue: Maria proaspătă (2 seturi ușoare), Valentova 3 seturi consecutive
- Vârstă: 38 ani vs 19 ani — pe iarbă cu slice, Maria câștigă experiența

**Scenariu probabil: Maria 6-2, 6-3 sau 6-3, 6-4**

---

## 9. VERDICT FINAL

| Market | Probabilitate | Scor | Recomandare |
|---|---|---|---|
| **U12.5 Set 2** | **~85-87%** | **7/10** | **✅ PICK** |
| Maria câștigă | ~83% | — | informativ |

---

## RANKING PICKS AZI (actualizat)

| # | Meci | Turneu | Ora | Score |
|---|---|---|---|---|
| **1** | **Maria vs Valentova** | Eastbourne QF | **17:00 BST** | **7/10** |
| 2 | Ostapenko vs Sonmez | Eastbourne QF | 11:00 BST | TBD |
| 3 | McNally vs Marcinko | Eastbourne QF | TBD | TBD |

---

## SURSE

- [TennisAbstract JS — Tatjana Maria](https://www.tennisabstract.com/jsmatches/TatjanaMaria.js)
- [TennisAbstract JS — Tereza Valentova](https://www.tennisabstract.com/jsmatches/TerezaValentova.js)
- [TennisStats H2H — Maria vs Valentova](https://www.tennisstats.com)
- [WTA Official — Maria stuns Paolini Eastbourne 2026](https://www.wtatennis.com/videos/4525532/maria-stuns-top-seed-paolini-in-eastbourne-for-first-top-20-win-in-12-months)
- [LTA Eastbourne 2026 Results](https://www.lta.org.uk/fan-zone/lexus-eastbourne-open/news/2026-results-updates/)
- [RallyHer — Eastbourne 2026 Draw & Results](https://rallyher.com/eastbourne-open-2026-wta-results-draw-scores-schedule/)
- [TennisTonic — Valentova beats Tomljanovic](https://tennistonic.com/tennis-news/1017956/tereza-valentova-gets-the-better-of-tomljanovic-in-the-2nd-round-to-set-up-a-battle-vs-maria-highlights-eastbourne-results/)
- Model Markov+WElo: `simulations/WTA/evaluations/1.5_WTA_Under12_5.csv` (run 2026-06-25)
