# WTA CoVe — U12.5 Set 2 — Triple Filter v1.1
## Jessica Pegula vs Iva Jovic
### Wimbledon 2026 — R4 — Grass — 5 iulie, 13:00 BST

---

## DATE MODEL (1.5_WTA_Under12_5.csv)

| Parametru | Valoare |
|---|---|
| Tournament | WIMBLEDON, Grand Slam, Grass, R4 |
| Player A | Jessica Pegula (WTA #4) |
| Player B | Iva Jovic (WTA #16) |
| p_hold_a (Pegula) | 0.7888 |
| p_hold_b (Jovic) | 0.7523 |
| hold_asym | 0.0364 |
| blowout_score | 1 |
| UNSTABLE | False |
| tb_p_raw | 0.0911 |
| **tb_p_cal** | **0.0824** |
| **P(U12.5 S2)** | **91.76%** |
| p_markov | 0.6096 (Pegula favorita) |
| p_elo | 0.5991 (confirmare) |
| fatigue_flag_b (Jovic) | True |

---

## PASUL 1 — CSV Model + Market Check

### 1.1 — Prag operațional
- tb_p_cal = 0.0824 ≤ 0.10 ✅
- Semnal U12.5 S2 activ

### 1.2 — Elo/Markov double guard
- gap = |p_elo − p_markov| × 100 = |0.5991 − 0.6096| × 100 = **1.05pp**
- ≤ 35pp ✅ — ambele modele confirmă Pegula ca favorit cu marjă egală
- p_elo ≠ 0.0 ✅

### 1.3 — UNSTABLE flag
- UNSTABLE = False ✅ — fără cap la scor din model

### 1.4 — Robinhood/Market Check
- **Polymarket (Sportytrader confirmat):** P(Pegula câștigă meciul) = **~66%** (odds implicite 1.52)
- P(Jovic) = ~34%
- Zona 60-74% → **continuă, notează divergența** față de p_markov
- Divergență market (66%) vs p_markov (61%) = **5pp** → sub pragul de 15pp ✅ — nicio investigație necesară
- Concluzie Pasul 1: market confirmă Pegula ca favorit, clasa de piață moderată (nu blowout), semnal validat

**Surse:** [Sportytrader odds Pegula-Jovic](https://www.sportytrader.com/en/odds/jessica-pegula-iva-jovic-8602757/) | Polymarket WTA R4

**→ PASUL 1: PASS ✅**

---

## PASUL 2 — TennisAbstract (suprafață Grass)

### 2.1 — Jessica Pegula — Grass History (22 meciuri confirmate)

**Meciuri complete cu scoruri (Wimbledon + turneele de iarbă semnificative):**

| An | Turneu | Runda | Adversar | Scor complet | S1 TB | S2 TB |
|---|---|---|---|---|---|---|
| 2021 | Wimbledon | R1 | Caroline Garcia | 6-3, 6-1 | Nu | Nu |
| 2021 | Wimbledon | R2 | Liudmila Samsonova | 6-4, 3-6, 6-3 | Nu | Nu |
| 2022 | Wimbledon | R1 | Donna Vekic (#~30) | 6-3, **7-6** | Nu | **DA** |
| 2022 | Wimbledon | R2 | Harriet Dart | 4-6, 6-3, 6-1 | Nu | Nu |
| 2022 | Wimbledon | R3 | Petra Martic (#~45) | 6-2, **7-6** | Nu | **DA** |
| 2023 | Wimbledon | R1 | Lauren Davis (#~78) | 6-2, **6-7(8)**, 6-3 | Nu | **DA** (pierdut TB) |
| 2023 | Wimbledon | R2 | Cristina Bucsa | 6-1, 6-4 | Nu | Nu |
| 2023 | Wimbledon | R3 | Cocciaretto | 6-4, 6-0 | Nu | Nu |
| 2023 | Wimbledon | R4 | Lesia Tsurenko | 6-1, 6-3 | Nu | Nu |
| 2023 | Wimbledon | QF | Vondrousova | 6-4, 2-6, 6-4 | Nu | Nu |
| 2024 | Wimbledon | R1 | Ashlyn Krueger | 6-2, 6-0 | Nu | Nu |
| 2024 | Wimbledon | R2 | Xinyu Wang (#~65) | 6-4, **6-7**, 6-1 | Nu | **DA** (pierdut TB) |
| 2024 | Berlin | QF | Katerina Siniakova | **7-6(2)**, 3-6, 6-3 | **DA** | Nu |
| 2024 | Berlin | SF | Coco Gauff (#~3) | 7-5, **7-6(2)** | Nu | **DA** (câștigat TB) |
| 2024 | Berlin | F | Anna Kalinskaya | **6-7(0)**, 6-4, **7-6(3)** | **DA** (pierdut) | Nu |
| 2025 | Wimbledon | R1 | Cocciaretto | 2-6, 3-6 (înfrângere) | Nu | Nu |
| 2026 | Berlin | R16 | Katerina Siniakova | 6-2, 6-4 | Nu | Nu |
| 2026 | Berlin | QF | Madison Keys (#~17) | **7-6(5)**, **7-6(8)** | **DA** | **DA** ← CASCADE |
| 2026 | Berlin | SF | Aryna Sabalenka (#~2) | 6-4, **6-7(4)**, 6-0 | Nu | **DA** |
| 2026 | Berlin | F | Linda Noskova | 4-6, 6-4, 6-3 (înfrângere) | Nu | Nu |
| 2026 | Wimbledon | R1 | Darja Vidmanova | 7-5, 6-3 | Nu | Nu |
| 2026 | Wimbledon | R2 | Sara Sorribes Tormo | **7-6(6)**, 6-1 | **DA** | Nu |
| 2026 | Wimbledon | R3 | Jessica Bouzas Maneiro | 6-1, 6-3 | Nu | Nu |

**STATISTICI PEGULA (Grass):**
- Total meciuri: 22 ✅ (≥10, sample solid)
- **S2 TB count: 6/22 = 27.3%** (zona 15-33%, neutru — sub pragul de risc 33%)
- **Meciuri cu S1 TB: 4** (Berlin QF 2024, Berlin F 2024, Berlin QF 2026, Wimbledon 2026 R2)
- **S1→S2 cascade: 1/4 = 25%** (zona 20-33%, neutru)
  - Singurul cascade: Berlin QF 2026 vs Keys (#17) — ambele TB câștigate de Pegula

### Contextul S2 TBs Pegula (CRITIC — calitatea adversarelor):

1. **vs Vekic (#30)** — Vekic specialist iarbă (ex-QF Wimbledon), Pegula a câștigat TB. Nu echivalentă Jovic.
2. **vs Martic (#45)** — opponent solid dar sub Top 30. Pegula a câștigat TB.
3. **vs Davis (#78)** — adversară inferioară mult, a ratat 3 match-uri în TB S2 (6-8). Context: Davis a jucat extraordinar, Pegula prea relaxată. A câștigat S3 6-3 ușor. Cel mai relevant: **Pegula riscă TB S2 când subestimează adversara**.
4. **vs Wang (#65)** — a pierdut S2 TB 6-7, a câștigat S3 6-1. Pattern: Pegula pierde TB, revine imediat.
5. **vs Gauff (#3)** — peer Top 5. Pegula a câștigat TB S2 7-2 clar. Nu e vulnerabilă la peer-uri.
6. **vs Keys (#17) — CASCADE** — ambele sets TB, Pegula a câștigat ambele. Cel mai relevant comparator ranking-wise pentru Jovic (#16): Pegula a controlat perfect.

**Concluzie Pegula:** S2 TB-urile apar predominant vs adversare din afara Top 30 care joacă peste așteptări. Vs adversare top-20 (Keys, Gauff) → Pegula câștigă TB-urile clar. Jovic e #16 — în zona unde Pegula performează bine în TB.

**Surse:** [tennis-x.com Pegula Wimbledon history](https://www.tennis-x.com/results/wimbledon/jessica-pegula.php) | [WTA Pegula Berlin 2026 SF](https://www.wtatennis.com/news/4522854/pegula-nears-second-berlin-crown-after-fourth-win-over-sabalenka-sends-her-into-final) | [Tennis Now Berlin QF Pegula-Keys](https://tennisnow.com/pegula-tops-keys-in-all-american-clash/) | [Wimbledon 2023 R1 Forum](https://www.tennisforum.com/threads/wimbledon-r1-4-jessica-pegula-survives-test-against-lauren-davis-6-2-6-7-8-6-3.1405222/page-2)

---

### 2.2 — Iva Jovic — Grass History (8 meciuri confirmate — BORDERLINE)

| An | Turneu | Runda | Adversar | Scor complet | S1 TB | S2 TB |
|---|---|---|---|---|---|---|
| 2025 | Wimbledon | R1 | Suzan Lamens | 1-6, 1-6 (înfrângere) | Nu | Nu |
| 2026 | Queen's | R1 | Antonia Ruzic | 6-3, 6-4 | Nu | Nu |
| 2026 | Queen's | R16 | Alexandra Eala | 6-2, 6-2 | Nu | Nu |
| 2026 | Queen's | QF | Amanda Anisimova (#~30) | 6-2, 3-6, 6-3 | Nu | Nu |
| 2026 | Queen's | SF | Emma Raducanu (#~47) | 2-6, 2-6 (înfrângere) | Nu | Nu |
| 2026 | Wimbledon | R1 | Jacqueline Cristian (#~120) | **7-6(1)**, 6-0 | **DA** | Nu |
| 2026 | Wimbledon | R2 | Tatjana Maria (#~110) | 6-1, 6-2 | Nu | Nu |
| 2026 | Wimbledon | R3 | Ekaterina Alexandrova (#~18) | 6-3, 3-6, 6-4 | Nu | Nu |

**STATISTICI JOVIC (Grass):**
- Total meciuri: 8 ⚠️ BORDERLINE (sub pragul de 10 → cap 7/10 din sample)
- **S2 TB count: 0/8 = 0%** — semnal extrem de puternic
- **Meciuri cu S1 TB: 1** (vs Cristian R1 Wimbledon 2026)
- **S1→S2 cascade: 0/1 = 0%** → ≤20% → +1pp confirmare
  - vs Cristian: S1=7-6(1), S2=**6-0** — blowout imediat după TB S1

**Pattern Jovic pe iarbă:** Zero S2 TB în toate cele 8 meciuri, inclusiv în meciul 3 seturi vs Alexandrova unde S2 = 3-6 (pierdut clar, nu TB). Pattern consistent: chiar când pierde un set, o face decisiv (nu TB). Același lucru când câștigă.

**Contextualizare Wimbledon 2026 în turneu (ambele jucătoare):**
- Pegula R2: **S1=7-6(6) → S2=6-1** — exact pattern U12.5 S2 ✅
- Jovic R1: **S1=7-6(1) → S2=6-0** — exact pattern U12.5 S2 ✅✅
- Ambele jucătoare au demonstrat în acest turneu că după un S1 TB, S2 se termină decisiv

**Surse:** [WTA Olympics.com Jovic R4 Wimbledon 2026](https://www.olympics.com/en/news/wimbledon-2026-iva-jovic-fights-through-to-the-fourth-round-by-beating-ekaterina-alexandrova) | [LTA HSBC Championships 2026](https://www.lta.org.uk/fan-zone/international/hsbc-championships/news/2026/) | [JustWomensSports Jovic R1 Wimbledon](https://justwomenssports.com/reads/iva-jovic-wimbledon-2026-tennis-scores/)

---

### 2.3 — Scor Pasul 2

| Factor | Valoare | Impact |
|---|---|---|
| Pegula sample | 22 meciuri | ✅ solid |
| Pegula S2 TB rate | 27.3% (15-33% zona) | Neutru (cap 7/10) |
| Pegula S1→S2 cascade | 1/4 = 25% (20-33% zona) | Neutru |
| Jovic sample | 8 meciuri | ⚠️ BORDERLINE → cap 7/10 |
| Jovic S2 TB rate | 0/8 = 0% | ✅✅ +1pp |
| Jovic S1→S2 cascade | 0/1 = 0% (≤20%) | ✅ +1pp |

**Scor Pasul 2: 7/10** (capped de Jovic sample borderline + Pegula 27.3%; offset parțial de Jovic 0%)

**→ PASUL 2: CONTINUĂ la 7/10 ⚠️**

---

## PASUL 3 — Context Manual

### 3.1 — Condiție Fizică

**Jessica Pegula:**
- Wimbledon 2026: R1 6-3 1h | R2 7-6(6)/6-1 1h29m | R3 6-1/6-3 ~1h15m
- **Zero seturi pierdute în 3 runde** — cea mai consistentă performanță a ei la Wimbledon
- Pre-Wimbledon: Berlin finalist (6/7 wins pe iarbă 2026), recuperată după problema de gleznă de la Queen's
- **2 zile rest înainte de R4** — fizic complet proaspăt
- Fatigue flag: False (model confirmă)

**Iva Jovic:**
- Wimbledon 2026: R1 7-6(1)/6-0 | R2 6-1/6-2 | R3 **6-3/3-6/6-4 → 2h30min (3 seturi)**
- **had_3sets_7d = True** (model confirmă) → fatigue flag activat
- R3 vs Alexandrova (#18) jucat pe **3 iulie** — cu 2 zile în urmă
- Pre-Wimbledon: **retragere de la Nottingham cu gleznă stângă** — nu a jucat Eastbourne, Bad Homburg
- A jucat Queen's (SF), dar cu pauza pre-Wimbledon mai mică decât Pegula
- Glezna stângă: latent concern — joc de 3 seturi în R3 poate fi semnificativ

**Surse:** [WTA Pegula R2 Wimbledon 2026](https://www.wtatennis.com/news/4529508/pegula-rallies-in-first-set-tiebreak-wins-last-six-games-to-reach-wimbledon-third-round) | [Jovic Nottingham withdrawal](https://www.tennisworldusa.org/tennis/news/WTA_Tennis/167372/iva-jovic-cancels-her-nottingham-debut-after-queen-s-injury/) | [Olympics.com Jovic R3](https://www.olympics.com/en/news/wimbledon-2026-iva-jovic-fights-through-to-the-fourth-round-by-beating-ekaterina-alexandrova)

### 3.2 — Stil de Joc

**Pegula pe iarbă:**
- Return dominant: 54% puncte câștigate pe primul serviciu al adversarului, 60% pe al doilea
- First serve won: 67-79% în funcție de rundă — eficient dar nu dominant
- Al doilea serviciu: **~48-54% puncte câștigate** — mediu, expus pe iarbă rapidă
- Câștigă prin consistență + return, nu prin ace
- Joacă mai bine pe măsură ce meciul avansează (demonstrat în R2 după ce a scăpat de TB)

**Jovic pe iarbă:**
- Stil: flat, agresiv, tempo rapid — "this surface takes me the least time to adjust"
- Carieră pe iarbă: **18-3 (85.7%)** — performanță excepțională pentru vârsta ei (20 ani)
- **Al doilea serviciu: ~70% puncte câștigate pe iarbă** — net superior lui Pegula pe această suprafață
- Aces: ~2.2/meci — nu heavy server, dar serviciu curat și eficient
- Avantaj structural minor al lui Jovic la serviciu pe iarbă rapidă

**Surse:** [TennisRatio Jovic grass stats](https://www.tennisratio.com/players/IvaJovic.html) | [Olympics.com Jovic grass quote](https://www.olympics.com/en/news/wimbledon-2026-iva-jovic-jacqueline-cristian-round-one-results)

### 3.3 — Condiții Meteorologice

- **Temperatura: ~29°C (85°F)**, predominant senin, fără ploaie anticipată
- **Iarbă uscată + căldură = suprafață mai rapidă, sărire joasă**
- Condiții favorizează serverul → hold rates susținute → seturi mai scurte
- Context: reduce parțial avantajul de return al lui Pegula, dar o iarbă rapidă tensionează musculatura — factor de risc pentru glezna lui Jovic (3 seturi în urmă cu 2 zile)

**Sursă:** [AccuWeather Wimbledon July 5](https://www.accuweather.com/en/gb/wimbledon/sw19-4/july-weather/323341) | [ESPN Wimbledon schedule](https://www.espn.com/tennis/story/_/id/49155718/wimbledon-2026-today-order-play-daily-schedule-results-weather-forecast-how-watch)

### 3.4 — Context Psihologic și Motivație

**Jessica Pegula:**
- Best Wimbledon result: **QF 2023** — astăzi are șansa a doua QF Wimbledon
- 2026 = cel mai bun sezon al carierei (SF AO 2026, finalist Berlin)
- Motivare excepțională: victorie azi = QF Wimbledon
- Mental mai stabilă: a trecut prin faze de slăbiciune la Slam-uri, acum demonstrează maturitate
- A câștigat 4 meciuri consecutiv vs Sabalenka (#2) în istoricul head-to-head — nu e intimidată de presiune

**Iva Jovic:**
- Best Wimbledon: **R4 2026 — aceasta este prima oară în R4 la orice Grand Slam**
- Joacă "cu house money" — nicio presiune externă, totul e bonus
- Mentor informal: Novak Djokovic ("if Novak gives you advice, you follow it")
- A eliminat Alexandrova (#18 seed) în R3 — capabilă de rezultate mari
- Antrenor: Thomas Gutteridge (a crescut CiCi Bellis, #565→#35)
- **Risc psihologic:** Prima R4 de Slam + față de Top 5 = presiune diferitcalitativ. În plus, a luptat 2h30m acum 2 zile.

**Surse:** [WTA Pegula R4 Wimbledon](https://www.wtatennis.com/news/4530274/pegula-continues-strong-grass-form-to-reach-fourth-round-at-wimbledon) | [Djokovic mentor Jovic](https://www.claytenis.com/features/djokovic-iva-jovics-secret-coach-if-novak-gives-you-advice-you-follow-it/)

### 3.5 — Head to Head

- **Pegula 2-0** overall: Charleston 2026 (Clay, 2-1), Dubai 2026 (Hard, 2-0)
- **ZERO head-to-head pe iarbă**
- Ambele meciuri anterior pe non-grass → H2H pe suprafața curentă nu e informativ direct
- La Charleston: meci 3 seturi (Jovic a luat un set) — Jovic poate compete
- La Dubai: Jovic 0-2 — Pegula a dominat clar

**Sursă:** [Wimbledon H2H Stats](https://www.wtatennis.com/)

### 3.6 — Coach

| Jucătoare | Antrenor |
|---|---|
| Pegula | **Mark Knowles + Mark Merklein** (din 2024) — Knowles ex-#1 mondial dublu, 3 GS titles |
| Jovic | **Thomas Gutteridge** + Novak Djokovic (mentor informal) |

**Sursă:** [Pegula coaches](https://www.sportskeeda.com/tennis/jessica-pegula-tennis-coach) | [Jovic coach](https://www.essentiallysports.com/tennis-news-who-is-iva-jovic-coach-thomas-guttheridges-net-worth-wife-parents-contract/)

### 3.7 — Summary Factori Context

| Factor | Impact pe U12.5 S2 |
|---|---|
| Jovic fatigue (3 seturi, 2h30m) | ✅✅ Major pozitiv |
| Jovic gleznă stângă (retragere Nottingham) | ✅ Pozitiv (risc fizic S2) |
| Pegula zero seturi pierdute în turneu | ✅✅ Major pozitiv |
| Temperatura 29°C → iarbă rapidă | ✅ Pozitiv (seturi mai scurte) |
| Wimbledon 2026 pattern ambele (S1 TB → S2 blowout) | ✅✅ Confirmare structurală |
| Pegula motivare maximă (QF pe masă) | ✅ Pozitiv (calitate ridicată) |
| Jovic prima R4 Grand Slam (presiune) | ✅ Ușor pozitiv |
| Jovic serviciu doi mai bun pe iarbă (~70%) | ❌ Risc minor |
| Jovic record iarbă 85.7% carieră | ❌ Risc minor |
| Market 66% (nu blowout) | ❌ Risc moderat |
| H2H zero pe iarbă | Neutru |

---

## SCOR FINAL U12.5 SET 2

| Criteri | Rezultat |
|---|---|
| Pasul 1 | PASS ✅ |
| Pasul 2 Sample | Borderline (Jovic 8 meciuri) → cap 7/10 |
| Pasul 2 S2 TB | Pegula 27.3% (15-33%, neutru) + Jovic 0% (+1pp) |
| Pasul 2 Cascade | Pegula 25% (neutru), Jovic 0% (+1pp) |
| UNSTABLE | False → fără cap suplimentar |
| Pasul 3 Context | Puternic pozitiv (fatigue Jovic + Pegula fresh) |

**SCOR: 7/10**

---

## PROBABILITATE CONTEXTUALĂ

- Model: tb_p_cal = 0.0824 → P(U12.5 S2) = **91.76%**
- Ajustare context: +1pp (fatigue Jovic, iarbă rapidă, pattern turneu)
- Risc offset: -1pp (Jovic serviciu doi puternic, 66% market nu e blowout)
- **Probabilitate contextuală finală: ~91-92%**

---

## PREDICȚIE MECI — CINE CÂȘTIGĂ

**Pegula câștigă, probabil în 2 seturi.**

- Set 1: Ușor favorit Pegula (61-66% market). Jovic poate câștiga S1 (34%). Dacă merge la TB, Pegula a demonstrat la Berlin că câștigă TB vs adversare top-20 (Keys).
- Set 2: Indiferent de cine câștigă S1, S2 va fi decisiv (fără TB). Dacă Pegula câștigă S1 → probabil 6-2/6-3 în S2 (Jovic obosit). Dacă Jovic câștigă S1 → Pegula revine cu 6-2/6-1 în S2 (exact pattern Berlin SF vs Sabalenka: Pegula a pierdut S2 TB 6-7, a câștigat S3 6-0).
- Scor cel mai probabil: **6-4 6-2** sau **7-6 6-2**
- S2 fără tiebreak: **probabilitate contextuală ~91-92%**

---

## VERDICT

**SCOR 7/10 — MODEL PASS, SCOR SUB MINIMUL RECOMANDAT PE IARBĂ**

Piața: tb_p_cal 0.0824 ✅ | P(U12.5 S2) 91.76% | Scor CoVe 7/10

---

## ⚠️ ATENȚIONARE BACKTEST (Iarbă)

**Scor 7/10 pe iarbă = sub minimul recomandat de 9/10.**

- HR backtest pentru scor proxy 7/10 pe Grass (p_tb 0.070-0.098): **82.4%**
- Baseline iarbă general: **86.6%** (N=335 meciuri)
- **Scor 7/10 este SUB baseline** — nu adaugă valoare față de a paria la întâmplare
- Minimum recomandat pe iarbă: **9/10** (HR ~94.7%, N=19)

**Limitări principale:**
1. Jovic: 8 meciuri grass (borderline sample) → imposibil de confirmat pattern robust
2. Pegula S2 TB rate 27.3% (neutru, dar nu confirmat sub 15%)
3. Market 66% → Jovic are șansă reală de 34%

**Context favorabil decizie personală:**
- Jovic fatigue (3 seturi acum 2 zile + gleznă) = factor concret, nu statistic
- Pegula proaspătă, motivată, în cel mai bun sezon al carierei
- Pattern Wimbledon 2026: ambele jucătoare au dat S2 decisiv după S1 TB

**Decide tu dacă joci la 7/10 pe iarbă. Recomandarea sistemului: PASS la scor < 9/10.**

---

*Generat: 2026-07-05 | Triple Filter v1.1 | Model: tb_p_cal=0.0824 | Surse: TennisAbstract, WTA, Olympics.com, LTA, Polymarket, AccuWeather*
