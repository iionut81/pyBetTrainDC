# WTA U12.5 Set 2 — CoVe Analysis
## Gina Feistel vs Diletta Cherubini
### Kitzbühel WTA 125 | Clay | Q3 | July 12, 2026 | 13:30 CEST

---

## MODEL SNAPSHOT

| Câmp | Valoare |
|---|---|
| tb_p_cal | **0.0927** ← ≤ 0.10 ✅ |
| p_u125 | 90.73% |
| premium_u125 | **YES** (BCI=0.1728 ≥ 0.12 + tb_p_cal < 0.10) |
| premium_elite | no (tb_p_cal=0.0927 > 0.08) |
| danger_zone | **NO** (min_hold=0.3573 < 0.40) |
| UNSTABLE | **NO** |
| min_hold | **0.3573** (Cherubini — extrem de slab) |
| hold_asym | **0.2689** (Feistel 62.61% vs Cherubini 35.73%) |
| BCI | **0.1728** — cel mai mare pe clay azi |
| blowout_score | 10/11 (structurally one-sided) |
| p_markov | 0.9288 (Feistel) |
| p_elo | 0.0 → SKIP standard → manual CoVe (CoreTennis data disponibil) |
| fatigue_flag_a (Feistel) | True (days_rest=1, had_3sets_7d=True) |
| days_rest_b (Cherubini) | **MODEL ERONAT: 76** → real = 3 zile (cf. CoreTennis + Sofascore) |

---

## PASUL 1 — CSV Model + Market Check

### 1.1 Triple Guard Model

```
✅ tb_p_cal = 0.0927 ≤ 0.10          — semnal primar U12.5 S2
✅ UNSTABLE = NO                      — fără outlier structural
✅ danger_zone = NO                   — min_hold 0.3573 sub pragul 0.40
⚠️  p_elo = 0.0                       — SKIP standard
   → override: CoreTennis clay data disponibil (127 meciuri Feistel, 12+ Cherubini)
   → manual CoVe permis per protocol
✅ Elo/Markov gap: incomensurabil (p_elo=0.0) — nu aplicăm regula 35pp
```

### 1.2 Robinhood Market Check

**STATUS: NOT AVAILABLE** — Kitzbühel WTA 125 qualifying nu este listat pe Robinhood.
Toate URL-urile testate → HTTP 404. Niciun turneu qualifying de nivel WTA 125 nu apare în sistem.

**Alternativă — TennisStats Elo estimat:**

| Indicator | Feistel | Cherubini |
|---|---|---|
| WTA Rank | 336 | 394 |
| TennisStats Elo | 193 | 157 |
| Win rate 2026 | **62%** (21/34) | **36%** (9/25) |
| Form recent | LWWLWWW | LLLLLWL |

Estimare piață implicită: Feistel 65-70% favorită → P(favorita) ≥ 60% ✅
Divergență față de p_markov (92.88%): ≈25pp → **investigăm** (explicat mai jos: p_elo=0.0 trage spre 50%, piața vede corect ~65-70%)

**Explicație divergență:** p_cal=54% e artefact (p_elo=0.0 → model blendează la 50%). Piața estimată 65-70% este mai realistă și consistent cu gap-ul de clasament/formă. Nu e semnal de injury sau surpriză — Feistel este pur și simplu mai bună structural.

**⚠️ ATENȚIONARE Robinhood**: Absența market check explicit înseamnă că nu putem certifica filtrul ≥75% (class gap confirmat). Continuăm cu ATENȚIONARE activă.

---

## PASUL 2 — CoreTennis Clay + S2 TB Pattern

### 2.1 Gina Feistel — Clay (127 meciuri career, CoreTennis ID 110942)

**Sample:** 127 meciuri clay ✅ (mult peste pragul de 10)

**S2 TB rate clay: 5/127 = 3.9%** ← CONFIRMARE PUTERNICĂ ✅

| Data | Turneu | Adversar | S1 | S2 | S3 | S1 TB? | S2 TB? |
|---|---|---|---|---|---|---|---|
| Jun 2025 | W50 Gdansk | Raluka Serban | 6-4 | 6-7(5) | 6-3 | NO | YES |
| Jun 2025 | W50 Gdansk | Weronika Ewald | 6-1 | 7-6(1) | — | NO | YES |
| Jun 2023 | W35 Klosters | Julie Belgraver | 6-7(4) | 7-6(3) | 6-1 | YES | YES 🔴 CASCADE |
| Nov 2022 | W15 Valencia | T. Rakotomanga | 6-3 | 7-6(3) | — | NO | YES |
| Aug 2019 | W35 Braunschweig | Lea Boskovic | 6-3 | 7-6(5) | — | NO | YES |

**S1 TB matches clay identificate:** ~10 meciuri
**S1 TB → S2 TB cascade: 1/10 = 10%** ✅ (≤20% → +1pp confirmare)

**Contextul celor 5 S2 TB:**
- Toate vs jucătoare de rang similar (W15-W50 level, rang ~350-550)
- Cel mai recent: Jun 2025 (>12 luni în urmă)
- **Zero S2 TB în 2026** pe clay
- Cascade unicul (Belgraver 2023): meci W35 vs adversar rang ~500, Set 3 dominat 6-1 → nu e pattern sistematic

**Concluzie Feistel Pasul 2:** Semnal EXCELENT. 3.9% S2 TB pe clay = structural nu merge în TB. ✅✅

---

### 2.2 Diletta Cherubini — Clay (2026 complet + 2023-2025 parțial, CoreTennis ID 118055)

**Sample disponibil:** 12 meciuri clay (7 din 2026 complet + 5 din 2023-2025 parțial)
⚠️ Sample borderline (8-12 meciuri conform tabel scor) — date istorice incomplete

**S2 TB rate clay disponibil: 2/12 = 16.7%** (zona 15-25% → -1pp)

**2026 clay — 7 meciuri complete:**

| Data | Turneu | Adversar | S1 | S2 | S3 | S1 TB? | S2 TB? |
|---|---|---|---|---|---|---|---|
| Jul 06, 2026 | W75 The Hague | A-L Friedsam | 2-6 | 6-2 | 6-1 | NO | NO |
| **Jun 22, 2026** | **W35 Tarvisio** | **Maria Toma** | **7-6(6)** | **7-6(5)** | **—** | **YES** | **YES 🔴 CASCADE** |
| Jun 15, 2026 | WTA Brescia 125 | Tatiana Pieri | 6-1 | 6-7(4) | 6-3 | NO | YES |
| Jun 01, 2026 | W75 Caserta | Ayana Akli | 6-1 | 1-6 | 7-6(2) | NO | NO (S3 TB) |
| Jun 01, 2026 | W75 Caserta | Alice Rame | 6-2 | 6-3 | — | NO | NO |
| Jun 01, 2026 | W75 Caserta | Nastasja Schunk | 6-4 | 6-3 | — | NO | NO |
| May 25, 2026 | W35 Bol | Aurora Zantedeschi | 7-5 | 6-2 | — | NO | NO |

**S1 TB matches 2026: 1** (Toma, Tarvisio)
**S1 TB → S2 cascade 2026: 1/1 = 100%** ← ALERTĂ statistică (sample=1, nu e pattern, dar notăm)

**2023-2025 partial (5 meciuri): 0/5 S2 TB** → historicul anterior 2026 e clean

**Analiza cascadei unice (Toma, Jun 22, W35 Tarvisio):**
- Maria Toma: jucătoare română, rang estimat ~450-500
- Turneu W35 (cel mai mic nivel profesional)
- Meci extrem de echilibrat: S1=7-6(6), S2=7-6(5) → un meci de aceași nivel
- Contextul vs Feistel este **complet diferit**: Cherubini va fi clară inferioară (rang 336 vs 394, hold asym 26.89pp)
- Concluzie: cascada Toma e produs al unui meci de nivel egal; irelevanță structurală vs Feistel

**Concluzie Cherubini Pasul 2:** S2 TB 16.7% (zona 15-25% = -1pp). Cascada 1/1 statistic nereprezentativă (context meci complet diferit). Sample borderline. ⚠️

---

## PASUL 3 — Context

### Fizic

| Factor | Feistel | Cherubini |
|---|---|---|
| Days rest | **1 zi** (jucat Jul 11 The Hague) | 3 zile (jucat Jul 9 The Hague) |
| Last match | Pierdut vs Vedder 0-6, 2-6 | **Pierdut vs Lim 6-0, 6-4** (bagel!) |
| 3-set recent | ✅ (vs Ebeling Koning 7-6/4-6/6-3) | Nu |
| Fatigue model | True (corect) | False (model arată 76 zile — ERONAT) |

**Notă critică:** Ambele jucătoare vin de la **același turneu** (W75 The Hague), ambele cu pierderi recente. Cherubini a primit un bagel (6-0) în ultimul meci → impact psihologic potențial negativ. Feistel a pierdut 0-6, 2-6 → vrea revanșă.

**Temperatura Kitzbühel:** Iulie, Austria, clay exterior. Estimat 22-26°C. Condiții normale.

### Stil de joc

**Gina Feistel (Germania, 23 ani, rang 336):**
- Jucătoare baseline agresivă cu forehand penetrant
- 0 acuri în 2026 (serve fără risc mare) → rely on placement
- 9 break points câștigate pe meci (2026) → excelent la break
- Set 2 win rate 68% → dominant în seturi secundare
- Wins Straight Sets 47% → poate termina rapid

**Diletta Cherubini (Italia, rang 394):**
- Baseline defensivă, inconsistentă
- 0.60 acuri/meci (occasional weapon)
- 2.6 double faults/meci → serve instabilă
- 3 break points câștigate pe meci (2026) → slabă la break
- Set 2 win rate **36%** → clar inferioară în seturi secundare

### Motivație și miză

- **Q3**: Câștigătoarea intră în main draw WTA 125 → motivație ridicată pentru ambele
- Feistel e mai aproape de main draw regulat (rang mai bun) → mizează mai mult pe calificare
- Cherubini are nevoie de puncte după formă slabă → motivată, dar poate panică

### Analiza meciurilor cu S2 TB (context adversari)

**Feistel — 5 meciuri S2 TB clay:**
- Toți adversarii: rang 350-550, W15-W50 level
- Meciul din 2023 vs Belgraver (cascade): W35 meci echilibrat, nu pattern sistematic
- **vs Cherubini (rang 394, hold 35.73%)**: Cherubini NU poate forța TB structural

**Cherubini — 2 meciuri S2 TB clay 2026:**
- Toma (~450, W35): meci 7-6(6)/7-6(5) — nivel perfect egal
- Pieri (WTA Brescia 125): Cherubini a câștigat S1 6-1, pierdut S2 7-6(4) → a ceda dominanța
- **vs Feistel**: Feistel NU va ceda dominanța (hold 62.61%, asym 26.89pp)

---

## PREDICȚIE MECI

**Estimare probabilitate:**
- Model p_markov: Feistel 92.88% (inflated, p_elo=0.0)
- Estimare realistă: Feistel 68-72% (rank gap + formă + clay stats)
- Cherubini: 28-32%

**Dinamică anticipată:**
- Set 1: Feistel controlată, 6-3 sau 6-4 (Cherubini ia 1-2 game-uri de break)
- Set 2: Feistel menține ritmul, 6-2 sau 6-3 cel mai probabil
- Set 3: puțin probabil (dacă Cherubini bagel-ul de la The Hague nu i-a distrus complet motivația)

**Predicție scor:** Feistel 6-3 / 6-2 (straight sets, ~70 min)

**Set 2 estimare games:** 8 game-uri (6-2) → bine sub 12.5

**Probabilitate Set 2 under 12.5:** ~92-93% structural (BCI=0.1728, min_hold Cherubini 35.73%)

---

## ATENȚIONARE — CONTEXTUL BACKTESTULUI

Scor CoVe calculat: **7/10** (explicat mai jos)

Per backtest U12.5 Set 2 clay:
- Baseline fără filtru (clay): ~87% HR
- Cu tb_p_cal ≤ 0.10 (clay): ~91% HR
- Cu premium_u125 (clay): **93.7% HR**

**Scor 7/10 pe clay este SUB minimul de 8/10** recomandat pentru recomandare formală. HR la scor 7/10 proxy: ~90-91% (între baseline și premium complet). Pariul rămâne probabil corect structural dar nu atinge standard-ul de calitate pentru recomandare formală.

---

## SCOR FINAL U12.5 SET 2

| Factor | Ajustare |
|---|---|
| Pasul 1: tb_p_cal=0.0927 ✅, premium_u125 ✅ | Bază 8/10 |
| Pasul 1: Robinhood NOT AVAILABLE | -1pp |
| Pasul 2: Feistel S2 TB 3.9% (<15%) | +1pp confirmare ✅ |
| Pasul 2: Cherubini S2 TB 16.7% (15-25%) | -1pp |
| Pasul 2: Feistel cascade 10% (≤20%) | +0 (neutru) |
| Pasul 2: Cherubini cascade 1/1 (sample=1, irel.) | -0 (ignorăm, insuficient statistic) |
| Pasul 2: Sample Cherubini borderline (12 meciuri) | -0 (notat, nu penalizat suplimentar) |
| Pasul 3: Feistel fatigue (1 day rest, 3-set recent) | -1pp |
| Pasul 3: Cherubini bagel recent (psihologic negativ) | +0 (neutru pentru U12.5) |
| Pasul 3: min_hold Cherubini 35.73% structural | confirmare implicită |

**SCOR FINAL: 7/10**

### VERDICT: NU RECOMANDĂM ⚠️

**Motivație:**
- Scor 7/10 este sub minimul 8/10 pentru clay fără Robinhood confirmation
- Robinhood indisponibil la nivel WTA 125 qualifying
- Cherubini S2 TB 2026 (16.7%) introduce incertitudine la sample mic
- Feistel fatigue (1 day rest, 3-set meci în 7 zile) e factor real

**Context structural pozitiv (dacă userul vrea să ia decizia proprie):**
- BCI=0.1728 (cel mai mare candidat clay azi)
- Feistel S2 TB 3.9% pe clay (127 meciuri) = nu merge în TB
- min_hold Cherubini 35.73% = structural imposibil să forțeze 6-6
- Ambele pierderi recente (The Hague) → meciul e mai motivațional pentru Feistel
- HR estimat la structura acestui meci: ~92-93%

**Dacă userul decide să parieze:** odds recomandate ≥ 1.10, stake redus față de standard (max 2.5% bankroll).

---

## SURSE

- [CoreTennis Feistel (ID 110942)](https://www.coretennis.net/tennis-player/gina-feistel/110942/results.html)
- [CoreTennis Cherubini (ID 118055)](https://www.coretennis.net/tennis-player/diletta-cherubini/118055/results.html)
- [TennisRatio Feistel](https://www.tennisratio.com/players/GinaFeistel.html)
- [TennisRatio Cherubini](https://www.tennisratio.com/players/DilettaCherubini.html)
- [Sofascore Cherubini](https://www.sofascore.com/tennis/player/cherubini-diletta/276445)
- [WTA Kitzbühel 2026](https://www.wtatennis.com/tournaments/1162/kitzbhuel-125/2026)
- Model output: `simulations/WTA/evaluations/1.5_WTA_Under12_5.csv` (run 2026-07-12)
- TennisStats H2H data (provided by user)

---

*Generat: 2026-07-12 | Model run: 2026-07-12 | Template: Triple Filter U12.5 S2 v1.1*
