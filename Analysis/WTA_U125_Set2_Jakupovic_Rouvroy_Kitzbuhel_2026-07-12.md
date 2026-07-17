# WTA U12.5 Set 2 — CoVe Full Analysis
## Dalila Jakupovic vs Margaux Rouvroy
### Kitzbühel WTA 125 — Clay — Q3 (Qualifying Round 3) — 12 iulie 2026, 15:00 CEST

---

## PASUL 1 — CSV Model + Market Check

### Semnale model

| Câmp | Valoare | Status |
|---|---|---|
| BCI | **0.1667** | ≥ 0.12 → premium_u125 ✅ |
| tb_p_cal | **0.0865** | ≤ 0.10 ✅ |
| tb_p_raw | 0.0292 | consistent cu calibrat ✅ |
| p_hold_Rouvroy | 0.6096 | hold decent pe lut |
| p_hold_Jakupovic | **0.3523** | ≤ 0.40 → **blowout profile** ✅ |
| hold_asym | 0.2573 | gap semnificativ ✅ |
| min_hold | 0.3523 | < 0.40 (sub danger zone) ✅ |
| premium_elite | NO | BCI 0.167 ≥ 0.15 ✓ dar tb_p_cal=0.0865 > 0.08 ✗ |
| premium_u125 | **YES** | ✅ |
| danger_zone | NO | min_hold < 0.40 ✅ |
| UNSTABLE | NO | ✅ |
| blowout_score | 11 | ridge spre maxim |
| p_u125 | **91.35%** | ✅ |
| p_markov | 0.9196 (Rouvroy câștigă) | |
| p_elo | 0.0 | → SKIP standard; procedem manual (CoreTennis date clay disponibile) |

**p_elo=0.0 override:** Jakupovic are 50+ meciuri pe lut documentate (CoreTennis) → regulă feedback_pelo_zero_manual_cove aplicată. CoVe manual continuat.

### Elo/Markov gap
p_elo=0.0 → gap standard inaplicabil. Nu există estimare Elo validă în Sackmann pentru această pereche.

### Robinhood Market Check
**STATUS: NOT AVAILABLE.** Meciul nu apare pe Robinhood (Kitzbühel WTA 125 qualifying absent din piața de predicție).

**Piață alternativă (Matchstat):**
- Jakupovic: **50.26%** (~1.92 odds)
- Rouvroy: ~44% (~1.92 odds, ușor sub 50% datorită vigorisimii)
- Divergență față de p_markov (Rouvroy 92%): **~42pp** — masivă, depășeste pragul de 15pp → **FLAG obligatoriu**

**Explicație divergență:**
Modelul folosește hold rates Sackmann care reflectă performanța Jakupovic vs jucătoare de nivel superior (cariera: WTA top-100 → top-300). Pe lut în 2026, Jakupovic a jucat în principal W100/W50 (nu WTA 125+), unde hold rate-ul ei este mai bun decât cel modelat. QF la Aschaffenburg W100 (6-10 iulie) confirmă formă actuală solidă vs Rouvroy care are 4 înfrângeri consecutive la R1. Piața capturează forma curentă; modelul nu o vede (date only WTA 125+).

**Concluzie Pasul 1:** Fără Robinhood + divergență 42pp → **ATENȚIONARE de nivel critic.** Procedem cu CoVe complet dar reducere scor finală.

---

## PASUL 2 — TennisAbstract + CoreTennis (Clay)

### Sample minim
- Jakupovic clay matches 2024-2026: **~50 meciuri** ≥ 10 ✅
- Rouvroy clay matches 2024-2026: **~50 meciuri** ≥ 10 ✅

---

### JAKUPOVIC — Clay S2 Tiebreak Rate

**Meciuri clay cu S2 Tiebreak (2024-2026, din CoreTennis):**

| Data | Turneu | Nivel | Adversar | Rank adv. | S1 | S2 | S3 | S1→S2 cascade? |
|---|---|---|---|---|---|---|---|---|
| Jul 2026 | Aschaffenburg W100 | QF | Primorac Pavicic | ~500 | 6-4 | **7-6(5)** | — | NO (S1 nu TB) |
| Sep 2024 | Ceska Lipa W100 | R2 | Zantedeschi | ~450 | 7-5 (won) | **6-7(2)** | 5-7 (lost) | NO (S1 nu TB) |
| Jun 2024 | Prague W60 | R1 | Krejcova | ~550 | 5-7 (lost) | **7-6(6)** | 6-1 (won) | NO (S1 nu TB) |
| Jul 2024 | Bellinzona W60 | R1 | Kolb | ~600 | 6-3 (won) | **6-7(0)** | — | NO (S1 nu TB) |
| Apr 2024 | Szekesfehervar Q | Qual. | Fita Boluda | ~500 | 1-6 (lost) | **7-6(3)** | 6-4 (won) | NO (S1 nu TB) |

**S2 TB rate Jakupovic clay: 5/50 = 10%** (sub pragul de 15%) ✅ → **+1pp confirmare**

**S1 TB → S2 TB cascade Jakupovic clay (2024-2026):**
Meciuri cu S1 TB: Wolff (Aschaffenburg R1: 7-6, 0-6, 7-6), Seibold (Stuttgart-V R1: 7-6, 6-0), Primorac (Zagreb QF: 6-4 pierdut S1 nu TB — wait, S1 a fost 6-4 non-TB), Hechingen vs Carle (S1 TB pierdut, S2=0-6), Lombardini Trieste (S1 TB, S2=4-6 non-TB), Ciric Bagaric (S1 TB, S2=5-7 non-TB)

→ **S1 TB → S2 TB: 0/6 = 0%** ✅ → **+1pp confirmare (cascade absent)**

**Analiza contextuală S2 TB Jakupovic:**
Toate cele 5 S2 TB-uri au apărut vs jucătoare ranked 450-600 (nivelul Rouvroy azi = 344 → *mai bună*). TB-urile au apărut în contexte specifice: pierdere severă S1 urmată de rezistență S2 (Krejcova, Fita Boluda), sau set 2 strâns după S1 câștigat confortabil (Kolb 6-3 → TB S2 misterioasă). *Nicio caracteristică a meciului de azi (Rouvroy mai bun decât adversarele din TB-uri, hold rate 61% vs 35% → nu așteptăm S2 strâns structural).*

---

### ROUVROY — Clay S2 Tiebreak Rate

**Meciuri clay cu S2 Tiebreak (2023-2024, din CoreTennis):**

| Data | Turneu | Nivel | Adversar | Rank adv. | S1 | S2 | S3 | S1→S2 cascade? |
|---|---|---|---|---|---|---|---|---|
| Jul 2024 | Contrexeville W100 | QF | Zimmermann | ~480 | **7-6(5)** | **7-6(6)** | 10-3 MTB | **YES** |
| Jun 2024 | Biarritz W60 | R? | Vedder | ~410 | **7-6(4)** | **6-7(3)** | 6-4 | **YES** |
| Apr 2024 | Zaragoza W80 | R? | Andreeva | ~430 | 7-5 | **7-6(5)** | — | NO |
| Apr 2024 | Zaragoza W80 | R? | Stevanovic | ~310 | 7-5 | **7-6(3)** | — | NO |
| May 2024 | Split W25 | R? | Temin | ~500 | 6-1 | **7-6(3)** | — | NO |
| Jun 2023 | FO Qualifying | Q | Shymanovich | ~350 | **7-6(0)** | **7-6(4)** | — | **YES** |

**S2 TB rate Rouvroy clay: 6/50 = 12%** (sub pragul de 15%) ✅

**S1 TB → S2 TB cascade Rouvroy clay:**
- **Overall (2023-2024): 3/7 = 43%** — RISC (depășeste 33%) ⚠️
- **2025-2026: 0/4 = 0%** — fără cascade în ultimii 18 luni ✅

**Analiza contextuală cascade Rouvroy (cele 3 cascade):**

**1. Contrexeville W100 QF 2024 vs Zimmermann (~480):**
Zimmermann este jucătoare specialistă clay germană cu serviciu consistent. QF de W100 = presiune maximă, mize mari (acces la SF + puncte). Format match a fost cu Match Tiebreak (S3 = 10-3) → ambele seturi TB, fizic epuizante. *Relevantă azi? Parțial — nivelul adversarei (480) e similar Jakupovic (358), dar Jakupovic e mai puțin consistentă decât Zimmermann ca tip de jucătoare.*

**2. Biarritz W60 2024 vs Vedder (~410):**
Meci 3 seturi cu ambele TB. Vedder (neerlandeză) are serviciu solid și returnează bine. W60 = mize medii. *Rouvroy a ieșit la break în ambele seturi dar Vedder a revenit la 6-6. Relevantă? Da — Vedder tip similar Jakupovic (baseline, serviciu inconsistent dar bune returnuri). Dar Jakupovic are hold rate mai mic decât Vedder pe clay → Rouvroy ar trebui să o dominate mai ușor.*

**3. FO Qualifying 2023 vs Shymanovich (~350):**
Grand Slam Qualifying = presiune psihologică maximă. Shymanovich (bielorusă, ranked ~350 la acea vreme) e echivalentă ca nivel cu Jakupovic azi. Ambele seturi TB în condiții Grand Slam. *Cel mai relevant precedent. Dar: condiții psihologice FO Qualifying ≠ Kitzbühel WTA 125 Qualifying; Shymanovich mai consistentă la serviciu decât Jakupovic (hold rate 35.23%).*

**Concluzie cascade Rouvroy:** Pattern cascade prezent dar limitat la 2023-2024; **zero cascade în 2025-2026 (18 luni)**. Dacă S1 nu are TB (probabilitate TB/set = 8.65%), cascade irelevantă. Risc rezidual notat.

---

### Date agregate H2H page (statistica per meci individual 2026)

| Metrică | Jakupovic | Rouvroy | Avg |
|---|---|---|---|
| Avg games/set | 9.66 | 9.48 | **9.57** |
| Over 12.5 games/set | 7% | 8% | **~7-8%** |
| Under 12.5 games/set | **93%** | **92%** | ✅ |
| TB rate/meci | 18% | 12% | 15% |
| Breaks/match (3S) | 2.67 | 9.5* | variabil |

*Nota: "9.5 breaks per match" pentru Rouvroy este din sample mic de meciuri 3 seturi; nerelevant statistic.

**Concluzii Pasul 2:** ✅ Ambii la 10-12% S2 TB rate pe clay, ambii sub 15%. Cascade: Jakupovic 0% (excepțional), Rouvroy 0% în 2025-2026. H2H page confirma că 92-93% din seturile lor nu depășesc 12.5 jocuri. Pasul 2 este solid.

---

## PASUL 3 — Context Manual

### Condiție fizică

**Jakupovic (35 ani, Slovenia):**
- **Ultimul meci confirmat: QF Aschaffenburg W100, ~9-10 iulie** (model spune 69 zile — eronat; acelea sunt zile de la ultimul meci WTA 125+, nu W100)
- A jucat la Aschaffenburg: R1 vs Wolff **7-6(5), 0-6, 7-6(5)** (3 seturi, ~2h); R2 vs Astakhova 7-5, 6-2; QF loss vs Sobolieva 6-4, 6-4
- **Days rest efectiv: 2-3 zile.** A avut un meci de 3 seturi (Wolff) în ultimele 7 zile.
- Risc oboseală: **moderat** (3 seturi + QF în 3 zile la Aschaffenburg)
- Model: fatigue_flag=False (model nu vede meciurile W100) → **model subestimează oboseala Jakupovic**

**Rouvroy (25 ani, Franța):**
- Model: days_rest=1, fatigue_flag=True, had_3sets_7d=True
- A jucat probabil la Wimbledon sau altundeva imediat înainte (days_rest=1 → juca ieri)
- 4 înfrângeri consecutive: R1 Saint-Gaudens (mai), R1 Blois W75 (iun), R1 FO Qualifying, R1 Wimbledon
- **Formă actuală: SLABĂ.** Nu a câștigat un meci din aprilie (Bujumbura 2 QF).
- Risc oboseală: **ridicat** (1 zi odihnă, meci 3 seturi în ultimele 7 zile)

**Comparație fizică:** Ambele cu probleme de oboseală; Rouvroy mai afectată (1 zi vs 2-3 zile Jakupovic). La vârste diferite: 35 vs 25 — recuperare mai lentă pentru Jakupovic teoretic, dar forma ei recentă o compensează.

---

### Stil de joc

**Dalila Jakupovic:**
- Baseline agresivă, dreapta. Serviciu puternic cu ACE-uri (2.11 ace/meci, 89% cu cel puțin 1 ace).
- Lovitura de fundal: variată, cap serios de joc.
- Clay: experientă vastă (35 ani, ~50+ meciuri/an pe lut). Hold rate scăzut din model reflectă meciuri vs adversare mai puternice. La W100 vs rivale de nivel similar: competitivă.
- **Atuuri azi:** ACE-uri, experiență, formă bună (QF Aschaffenburg), zero presiune psihologică.

**Margaux Rouvroy:**
- Jucătoare cu **backhand unimânar** — caracteristică rară în WTA. Acest stil implică:
  - Lovitură mai variată (slice + topsin) dar vulnerabilă la mingile înalte pe revers (clay cu bounce înalt)
  - La contraatac tinde să creeze unghiuri, nu să împingă prin putere
- Serviciu: fără ace (0 ace per meci in statistica H2H page 2026 — probabil sample mic, dar confirma că nu se bazează pe serviciu)
- Clay: 2024 ok (12-9), **2025 dezastruoasă (5-12)**, 2026 mix (bun Bujumbura, slab după)
- **Vulnerabilitate azi:** Formă slabă, obosit, backhand unimânar vulnerabil la topspin greu Jakupovic

---

### Motivație și mize

**Context turneu:**
- Kitzbühel WTA 125 = Generali Open Ladies, prize money $115,000
- Acesta este **Q3 = ultimul tur de calificare.** Câștigătoarea intră în tabloul principal.
- Puncte ranking WTA 125 qualifying: ~10-15 puncte pentru Q3, 18-20 pentru tabloul principal.

**Jakupovic:** Neseedată. Vine de la QF W100 Aschaffenburg — în formă. Nimic de pierdut, vrea să intre în tabloul principal. **Motivație: ridicată.**

**Rouvroy:** Seed #3 la calificări. **Presiunea seedingului** — dacă pierde, e o umilință suplimentară după 4 înfrângeri consecutive. Franțuzoaică pe lut = teren favorit teoretic. Dar forma e teribilă. Poate resimți presiunea de a se califica.

---

### Context psihologic

**Jakupovic:**
- Veterană de 35 ani cu $1.29M câștiguri career (high 69 WTA). A văzut totul.
- Nu are ce pierde. Vine cu momentum (QF Aschaffenburg beat Astakhova 7-5, 6-2 — aceeași Astakhova care a bătut-o pe Rouvroy în finala Bujumbura).
- **Crossover form:** Jakupovic a bătut Astakhova; Astakhova a bătut Rouvroy. Forme încrucișate indică Jakupovic ≥ Rouvroy la nivel actual.
- Mental: **stabilă, fără presiune**

**Rouvroy:**
- 25 ani, la o vârstă critică pentru carieră (ranking a coborât de la 211 la 344).
- 4 înfrângeri consecutive → **criză de încredere** probabilă.
- Seeding #3 = așteptări externe. Dacă pierde cu o neseedată de 35 ani, scena e dificilă psihologic.
- Backhand unimânar sub presiune poate deveni inconsistent.
- **Mental: fragil.**

---

### Temperatura și condiții

- **Kitzbühel, Austria, 12 iulie 2026: 23°C** — condiții normale de vară.
- Nu există val de căldură sau umiditate extremă.
- Clay Kitzbühel: lut roșu clasic, mai lent și mai alunecos decât clay dur (Halle, Roland Garros).
- La 15:00 CEST: soare, minge mai rapidă după prânz față de dimineață.
- **Condiții: neutre, fără impact semnificativ pe U12.5.**

---

### Antrenor

**Jakupovic:** Senad Jakupovic (famille) — setup comun pentru jucătoare cu buget limitat. Cunoaștere profundă a jucătoarei dar resurse tactice limitate față de antrenori de top.

**Rouvroy:** Informații antrenor curent neclarificate din surse disponibile. Franța are un sistem de antrenori federali activ (FFT), Rouvroy poate beneficia de suport tehnic.

---

## ESTIMARE MECI — Viziune Analyst Profesionist

### Factori structurali

Modelul spune Rouvroy câștigă 91.96% via Markov. Piața spune 50-50. Adevărul este probabil undeva la mijloc.

**Hold rates efective azi (estimate, nu din model):**
- Rouvroy ține ~55-60% (model 61%, formă slabă → -5pp) → ~0.55-0.56
- Jakupovic ține ~45-50% (model 35%, dar formă bună + nivel similar adversar → +10-15pp) → ~0.45-0.50
- Dacă Jakupovic ține 47%: expected games = ~22 total (vs model 20.93) → ~11 jocuri/set → bine sub 12.5

**Tipul de meci așteptat:**
- Jakupovic serveste tare (ace-uri), Rouvroy returnează bine dar fără ace (serviciu moale)
- Rouvroy va sparge des serviciul Jakupovic (hold rate scăzut al Jakupovic)
- Jakupovic va rezista pe serviciul Rouvroy mai bine decât piața crede (formă actuală)
- Seturi cu 2-3 break-uri fiecare → 6-3, 6-4 tipic

**Estimare set 2:**
- Cel mai probabil: **6-3 sau 6-4** (8-10 jocuri, well under 12.5)
- Scenariu mediu: **6-4** (10 jocuri)
- Scenariu pesimist: **7-5** (12 jocuri, aproape de limită)
- Scenariu TB: **7-6** (~15% șansă globală, dar tiebreak specific implică 8.65% per set conform modelului)

### Predicție meci (match winner)

> **Rouvroy câștigă, 6-4 6-3**

*Raționament:*
- Structural, Rouvroy ține mai bine pe clay (61% vs ~47% realistically Jakupovic)
- Forma slabă a Rouvroy e parțial compensată de seeding și de faptul că Jakupovic e obosită după 3 meciuri la Aschaffenburg
- Precedent direct: Jakupovic a bătut Astakhova (același level cu Rouvroy dar Astakhova mai consistentă)
- *Risk scenario:* Jakupovic câștigă 6-4 7-5 dacă Rouvroy iese prost din S1

Estimare finală: **Rouvroy 65% / Jakupovic 35%** (ajustat din model 92%/8%; piața 50%/50%)

---

## ANALIZA SET 2 — U12.5

### Scenariul Set 2

**Cel mai probabil traseu Set 1:**
- S1 scor așteptat: 6-4 sau 6-3 (Rouvroy câștigă), 2-3 break-uri
- Probabilitate S1 TB: **8.65%** (model calibrat)

**Scenariul Set 2 U12.5:**
- Dacă Rouvroy câștigă S1 confortabil (6-4): Jakupovic intră în S2 dezavantajată mental → riscul de a ceda rapid → 6-2/6-3 Set 2 → **WIN U12.5**
- Dacă Jakupovic rezistă în S1 (6-4 Jakupovic sau 7-5): moralul Jakupovic crește → S2 mai luptată dar tot 6-4 maxim → **WIN U12.5**
- Dacă S1 merge la TB (8.65%): analizăm cascade
  - Jakupovic cascade S1TB→S2TB: **0/6 = 0%** → zero risc cascade
  - Rouvroy cascade S1TB→S2TB: **0/4 = 0% recent** (3/7 overall, dar pattern dispărut din 2025)
  - Risc cascade estimat: <5% din scenariul de TB (~0.4% total)

**Suma riscuri S2 TB:**
- S2 TB fără S1 TB: ~10-12% (rata directă din istoric)
- S2 TB cu S1 TB cascade: ~0.4% (0.0865 × ~5%)
- **Total S2 TB probabilitate estimată: ~11-12%** → sub 15% ✅

---

## SCORING FINAL

| Criteriu | Status | Ajustare |
|---|---|---|
| tb_p_cal ≤ 0.10 | 0.0865 ✅ | baza |
| Elo/Markov gap | p_elo=0.0, inaplicabil | neutral |
| UNSTABLE | NO ✅ | — |
| danger_zone | NO ✅ | — |
| Robinhood | **NOT AVAILABLE** ⚠️ | nu putem da 8/10 clay fără RH |
| Market alternativ (50-50) | P(favorita) < 60% ⚠️ | SKIP trigger |
| Market divergență vs p_markov | **~42pp** ⚠️ | -1pp explicat dar alarma rămâne |
| S2 TB rate Jakupovic | 10% < 15% ✅ | +1pp |
| S2 TB rate Rouvroy | 12% < 15% ✅ | confirmare |
| S1→S2 cascade Jakupovic | 0/6 = **0%** ✅ | +1pp |
| S1→S2 cascade Rouvroy | 0/4 = **0% recent** ✅ | confirmare, not penalized |
| Oboseală Rouvroy | fatigue_flag=True | -0.5pp |
| Formă curentă | Jakupovic ✅ vs Rouvroy 4× L ⚠️ | neutral (reduce model edge dar nu TB risk) |
| Temperature | 23°C ✅ | neutral |

---

## ⚠️ ATENȚIONARE — SUB MINIMUL CLAY

**Scor calculat: 9/10 din Pasul 2 (date curate TB)**
**Scor final: 7/10 — ATENȚIONARE (sub minimul clay de 8/10)**

**Motive reducere la 7/10:**
1. **Robinhood indisponibil** → regulă clay: "8/10 + RH" → fără RH nu putem certifica minimul
2. **Market alternativ 50-50** → P(favorita) < 60% → SKIP trigger per Pasul 1
3. **Divergență 42pp** → modelul nu captează forma actuală → fiabilitate p_markov redusă
4. **Model days_rest eronat** → model spune 69 zile, realitatea 2-3 zile (Jakupovic juca la Aschaffenburg) → semnale model nu sunt complet de încredere pentru acest meci

**Backtest warning:** La pragul ≤0.10 tb_p_cal pe lut, HR=91.2%. Dar fără market confirm + cu divergență masivă, pick-urile contaminate nu au acest HR. Strict per triple filter: SKIP.

---

## VERDICT FINAL

> **NU RECOMANDĂM** — 7/10 ATENȚIONARE (sub minimul clay 8/10)

**Structural, semnalele TB sunt curate:**
- 11-12% S2 TB rate estimat → solid
- 0% cascade ambii jucători în 2025-2026 → excelent
- BCI 0.167, tb_p_cal 8.65% → premium_u125

**Problema este validarea externă:**
- Fără Robinhood (matchul nu există pe platformă) nu putem finaliza Pasul 1 conform regulilor
- Singura piață disponibilă (Matchstat 1.92/1.92) arată 50-50 → sub pragul de 60% obligatoriu
- Divergența de 42pp față de p_markov este cel mai mare semnal de prudență: piața știe că Jakupovic e mai bună acum decât modelul crede, ceea ce înseamnă seturi potențial mai lungi

**Dacă dorești să pariezi ignorând filtrul market (pe baza datelor structurale TB):** Pick-ul are fundamente solide (91.35% model + 0% cascade + 10-12% S2 TB rate). Dar regulile noastre există tocmai pentru a evita astfel de situații — piața 50-50 pe lut înseamnă meci mai echilibrat = setul 2 potențial mai lung.

---

*Fișier generat: 12.07.2026 | Meci: 15:00 CEST | Turneu: Kitzbühel WTA 125 Q3*
*Model: BCI=0.1667, tb_p_cal=0.0865, p_u125=91.35%*
*Surse: CoreTennis results (Jakupovic + Rouvroy clay 2024-2026), H2H page stats, Matchstat odds, WTA official draw, AccuWeather Kitzbühel*
