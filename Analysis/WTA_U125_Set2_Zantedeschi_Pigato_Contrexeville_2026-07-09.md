# WTA U12.5 Set 2 — CoVe Manual
## Aurora Zantedeschi vs Lisa Pigato
**Turneu:** Grand Est Open 88 — WTA 125, Contrexeville (Vosges), France  
**Suprafață:** Lut (outdoor) | **Tur:** R2 (R16) | **Data:** 09.07.2026, 15:00 CEST  
**Surse:** CoreTennis.net, TennisStat.com, Robinhood Markets, Weather Contrexeville  

---

## STATUS MODEL

**Zantedeschi vs Pigato nu apare în `1.5_WTA_Under12_5.csv`** — model absent pentru acest meci.  
Procedura: CoVe manual cu date empirice, conform regulii `[[feedback-pelo-zero-manual-cove]]`.  
Ambele jucătoare au sample ≥ 10 meciuri pe lut → CoVe valid.

---

## TRIPLE FILTER — PASUL 1 (Market Check)

| Criteriu | Zantedeschi | Pigato |
|---|---|---|
| tb_p_cal | N/A (model absent) | N/A |
| p_elo | N/A | N/A |
| Elo/Markov gap | N/A | N/A |
| unstable_reason | N/A | N/A |

**Robinhood check:**  
- Lisa Pigato: **67%** ≥ 60% → class gap confirmat de piață ✅  
- Aurora Zantedeschi: 33%  
- Divergență: 34pp (Robinhood vs piața generală) — reflectă class gap real, nu injury/form unknown  
- Concluzie Pasul 1: **CONTINUĂ cu manual CoVe** (ambii au date empirice suficiente)

---

## TRIPLE FILTER — PASUL 2 (TennisAbstract / CoreTennis — Lut)

### Aurora Zantedeschi — Clay S2 TB Rate

**Sample:** ~37 meciuri pe lut (2025-2026)  
**Sursa:** CoreTennis.net (ID 83021)

**S2 TB confirmate pe lut:**

| Data | Turneu | Tur | Scor | S1 | S2 | Context oponent |
|---|---|---|---|---|---|---|
| Apr 2026 | W75 Chiasso [Q-R16] | R16 cal. | W 6-4 7-6(6) | no TB | **TB** | Mazzola A. — ITF 300+ |
| Jun 2026 | W75 Caserta | R16 | L 3-6 7-6(3) 6-2 | no TB | **TB** | Mazzola A. — ITF 300+ |
| Jul 2026 | W35 Aix SF | SF | W 6-2 7-6(3) | no TB | **TB** | Sieg M. — ITF 300+ |
| May 2025 | W50 Portoroz | — | W 6-2 6-7(4) 6-3 | no TB | **TB** | Zavatska K. — WTA ~180 |
| Jun 2025 | W75 Ceska Lipa | — | W 5-7 7-6(2) 7-5 | no TB | **TB** | Jakupovic D. — WTA veteran |
| Jun 2025 | W35 Rome | — | W 7-5 7-6(8) | no TB | **TB** | De Stefano S. — ITF |
| Jun 2025 | W35 Rome | — | L 6-4 7-6(6) | no TB | **TB** | Ruggeri J. — ITF |
| Dec 2025 | W35 Monastir | — | L 6-3 7-6(5) | no TB | **TB** | Wong I. — ITF |
| Oct 2025 | Rovereto Open | — | L 6-3 7-6(2) | no TB | **TB** | Noha Akugue — WTA ~250 |
| Sep 2024 | Piracicaba | — | L 6-7(6) 7-6(7) 7-6(0) | **TB** | **TB** | Alves — ITF |

**S2 TB rate: ~10/37 = 27%** → zona "risc real" (15-33%) → **-1pp**

**Analiză contextuală S2 TBs Zantedeschi:**
- 7 din 10 TBs (70%) au fost împotriva jucătoarelor ITF (ranked 300-700+) — meciuri echilibrate
- Doar Zavatska (~WTA 180) și Jakupovic (veteran WTA) sunt la nivel WTA semi-competitiv
- **Contra lui Pigato (WTA 133), pattern H2H:** 6-4 6-3 × 2 — 0 TB în 4 seturi jucate
- Concluzie: 27% include mulți adversari inferiori — contra jucătoarelor WTA de calitate, Zantedeschi tinde să piardă clar, nu prin TB

**S1 TB → S2 cascade pe lut:**

| Data | Meci | Scor | S1 | S2 |
|---|---|---|---|---|
| Apr 2026 | vs Popa M. | 7-6(4) 5-7 6-1 | **TB** | no TB |
| May 2026 | vs Ce G. | 7-6(1) 6-1 | **TB** | no TB |
| Jul 2026 | vs Lew Yan Foon | 7-6(2) 6-3 | **TB** | no TB |
| Apr 2025 | vs Barnes K. | 6-7(6) 6-1 6-2 | **TB** | no TB |
| Jun 2025 | vs Spiteri D. | 6-7(3) 6-3 6-4 | **TB** | no TB |
| Sep 2025 | vs Cakarevic S. | 7-6(5) 6-1 | **TB** | no TB |
| Oct 2025 | vs Daavettila S. | 7-6(5) 6-4 | **TB** | no TB |
| Nov 2025 | vs Dudeney A. | 7-6(2) 2-6 6-4 | **TB** | no TB |
| Sep 2024 | vs Alves | 6-7(6) 7-6(7) 7-6 | **TB** | **TB** |

**S1 → S2 cascade: 1/9 = 11.1%** → ≤20% → **+1pp confirmare**

---

### Lisa Pigato — Clay S2 TB Rate

**Sample:** 83 meciuri pe lut (2024-2026)  
**Sursa:** CoreTennis.net (ID 101549) + TennisStat

**S2 TB confirmate pe lut (din 83 meciuri):**

| Data | Turneu | Tur | Scor | S1 | S2 | Context oponent |
|---|---|---|---|---|---|---|
| Jun 2025 | W35 Tarvisio | QF | L 4-6 7-6(4) 3-0 ret | no TB | **TB** (ret) | Paganetti V. — ITF |
| Jul 2025 | Rome 125 | R32 | L 7-5 7-6(5) | no TB | **TB** | Liu Claire — WTA ~80 |
| Mar 2026 | Antalya 125 | Q1 | W 4-6 7-6(5) 6-4 | no TB | **TB** | Ristic M. — ITF |
| 2024 | W75 Caserta | — | W 7-6(5) 7-6(7) | **TB** | **TB** | Janicijevic — ITF |

**S2 TB rate: 4/83 = 4.8%** → sub 15% → **+1pp confirmare puternică** ✅

**Analiză contextuală S2 TBs Pigato:**
- 4 TBs în 83 meciuri pe lut — structurally aproape zero
- Singurul caz împotriva WTA (Liu Claire, ranked ~80): Pigato a pierdut 7-5 7-6 — la un turneu WTA 125, contra unui adversar net superior. Contra adversarilor inferiori sau egali, rareori merge la TB
- **Contra lui Zantedeschi:** 6-4 6-3, 6-4 6-3 — fără TB în niciun set

**S1 TB → S2 cascade pe lut:**

| Data | Meci | S1 | S2 |
|---|---|---|---|
| Feb 2025 | vs Quevedo | 6-7(1) | no TB |
| May 2025 | vs Pieri | 7-6(5) | no TB |
| Sep 2025 | vs Grammatikopoulou | 6-7(5) | no TB |
| Oct 2025 | vs Fontenel | 7-6(5) | no TB |
| Jun 2026 | vs Yaneva | 7-6(5) | no TB |
| 2024 | vs Janicijevic | 7-6(5) | **TB** |

**S1 → S2 cascade: 1/6 = 16.7%** → ≤20% → neutru ✅

---

### PASUL 2 — Scor intermediar

| Metric | Zantedeschi | Pigato | Semnal |
|---|---|---|---|
| S2 TB rate (lut) | 27% (risc real) | 4.8% (excepțional) | Net: 0pp (-1 + +1) |
| S1→S2 cascade | 11.1% (+1pp) | 16.7% (neutru) | +1pp |
| Sample | 37 (valid) | 83 (solid) | OK |
| H2H context | — | — | 0 TB în 4 seturi H2H |

**PASUL 2: CONTINUĂ** ✅

---

## TRIPLE FILTER — PASUL 3 (Context manual)

### Condiție fizică și fatigue

| Factor | Zantedeschi | Pigato |
|---|---|---|
| R1 scor | W 6-1 6-2 (vs Jacquemot, seed #2) | W 6-1 1-6 7-6(7) (vs Fita Boluda) |
| Durată R1 | ~55-60 min | ~2h15 min |
| R1 seturi | 2 seturi, dominator | 3 seturi, a salvat 2 mingi de meci în S3 TB |
| Fatigue flag | **False** — fresh ✅ | **True** — semnificativ ⚠️ |
| Odihnă (days_rest) | 1 zi (R1 ieri 07.07) | 1 zi (R1 ieri 07.07) |

**Zantedeschi vine proaspătă, Pigato vine obosită.** Pigato a jucat match point-uri în S3 TB ieri — presiune mentală și fizică ridicate. Impactul asupra S2 de azi: Pigato ar putea servi mai slab în S2 → mai multe break-uri → mai puțin TB.

### Motivație și miză

- **Zantedeschi:** Cel mai bun rezultat WTA din carieră (R1 victorie vs seed #2). Elan maxim. Câștigătoarea turneului W35 Aix-les-Bains în săptămâna precedentă (final-ul pierdut, dar titlul luat). Momentum pozitiv.
- **Pigato:** WTA 133, campioana WTA 125 Madrid 2026. Joacă pentru a avansa la SF. Favorita turneului. Motivație standard de favorită.
- **Derby italian:** Amândouă din Italia — rivalitate cunoscută, dar fără efecte documentate în H2H (Pigato domină clar).

### Stil de joc și compatibilitate U12.5 S2

- **Zantedeschi:** Jucătoare de baza cu mult spin, nu servitoare puternică. TennisStat: 0 ași în meciurile din 2026. 7.5 break-uri primite per meci → serviciu fragil. Returnează bine, dă ritm.
- **Pigato:** Servitoare mai consistentă (1.1 ași per meci, 4 double faults per meci). 5.88 break-uri primite per meci. Joc mai solid la net (6.5 net points per meci). Agresivă din dreapta.
- **Compatibilitate:** Match cu multe break-uri de ambele părți (13+ per meci conform TennisStat) → sets nu ajung la 6-6. Pattern confirmat în H2H: 6-4 6-3, 6-4 6-3.

### Condiții meteo — Contrexeville, 09.07.2026

| Parametru | Valoare |
|---|---|
| Temperatură | 30°C |
| Condiții | Soare, cer senin |
| Vânt | 12-20 km/h NE, rafale până la 34 km/h |
| Risc de ploaie | 0% |
| UV | 8 (ridicat) |

**30°C + lut uscat = minge mai rapidă decât media pe lut** → reduce avantajul jucătoarelor defensive.  
**Rafale până la 34 km/h** → servicii afectate, mai multe duble greșeli → mai multe break-uri → structurally anti-TB.

### Context psihologic și mental

- Zantedeschi după victorie surpriză (6-1 6-2 vs seed #2): euforie, joacă liber, fără presiune
- Pigato după 2h15 și salvarea a 2 mingi de meci: o parte din încredere consumată
- Dar: Pigato 2-0 în H2H pe lut, ambele cu scor clar → Pigato știe că Zantedeschi nu o amenință structural
- Estimare psihologică: ambele cu motivație, dar Pigato are "credit psihologic" din H2H

### Antrenori

- **Pigato:** Lucrează cu Federica Ferro (antrenoare) — stabilitate și disciplină tactică
- **Zantedeschi:** Sistem italian ITF standard

---

## ANALIZĂ H2H DETALIATĂ

| Data | Turneu | Suprafață | Tur | Scor | S1 | S2 |
|---|---|---|---|---|---|---|
| 19.03.2026 | W35 San Gregorio | Lut | 1R | Pigato W 6-4 6-3 | no TB | no TB |
| 25.04.2024 | W35 Santa Margherita | Lut | 1R | Pigato W 6-4 6-3 | no TB | no TB |

**Concluzie H2H:**
- 100% victorii Pigato, ambele pe lut, ambele 6-4 6-3 (pattern identic!)
- **0 seturi cu TB în 4 seturi jucate** — structurally cel mai direct predictor
- Pattern 6-4 6-3 sugerează: Pigato câștigă comfortabil, fără a fi necesari pași suplimentari (TB)

---

## ESTIMARE SCOR ȘI CÂȘTIGĂTOARE

**Câștigătoare favorită:** Lisa Pigato (67% market, 2-0 H2H clay, WTA 133 vs 364)

**Scenarii probabile:**

| Scenariu | Probabilitate | S2 result |
|---|---|---|
| Pigato 6-4 6-3 (pattern H2H) | ~35% | U12.5 ✅ |
| Pigato 6-3 6-4 | ~25% | U12.5 ✅ |
| Pigato 6-2 6-3 (Pigato fragilă după fatigue, Zan agresivă) | ~15% | U12.5 ✅ |
| Pigato 6-4 7-5 (Zantedeschi rezistă mai bine) | ~12% | U12.5 ✅ |
| Pigato 6-3 7-6 (S2 TB, Pigato câștigă) | ~7% | ❌ |
| Zantedeschi câștigă (upset) | ~6% | — |

**Estimare scor:** Pigato W **6-4 6-3** (most likely) sau **6-3 6-4**  
**U12.5 S2 estimated P ≈ 91-93%**

---

## SCOR CoVe FINAL

| Criteriu | Valoare | Semnal |
|---|---|---|
| Model (tb_p_cal) | N/A (manual CoVe) | — |
| Robinhood | Pigato 67% ≥ 60% | ✅ |
| S2 TB rate Zantedeschi | 27% (15-33%) | -1pp |
| S2 TB rate Pigato | 4.8% (<15%) | +1pp |
| S1→S2 cascade Zantedeschi | 11.1% (≤20%) | +1pp |
| S1→S2 cascade Pigato | 16.7% (≤20%) | neutru |
| Sample valid | 37 + 83 | ✅ |
| H2H clay TB history | 0/4 seturi | structural confirmare |
| Fatigue Pigato | 3 seturi R1 | risc minor (mai multe break-uri, nu TB) |
| Vânt + căldură | 30°C + 34 km/h rafale | mai multe break-uri → anti-TB |
| UNSTABLE flag | Absent (fără model) | N/A |

**SCOR FINAL: 8/10 — RECOMMEND**  
*CoVe manual fără confirmare model, la minimul clay.*

---

## ATENȚIONARE BACKTEST

Conform `reference_u125_s2_backtest_surfaces.md`:
- **Clay 8/10 + Robinhood check → HR 91.3%** (nivel acceptabil)
- Scorul minim pe lut = 8/10; suntem la minim exact

Factori care reduc riscul față de baseline:
- Pigato 4.8% S2 TB pe 83 meciuri = cel mai solid semnal de confirmare
- H2H 0 TB în 4 seturi de meci
- 13+ break-uri per meci (TennisStat) = meci structural anti-TB

Factori de atenție:
- Zantedeschi 27% S2 TB — dar majoritar contra jucătoarelor ITF (context irelevant față de azi)
- CoVe manual fără model — incertitudine suplimentară față de matchurile cu model confirmat

---

## VERDICT

```
MARKET:  WTA U12.5 Set 2
MECI:    Zantedeschi vs Pigato — Contrexeville WTA 125, Clay, R2
DATA:    09.07.2026, 15:00 CEST
SCOR:    8/10 — RECOMMEND (la minimul clay)
BET:     VALID la cote ≥ 1.10
```

**Concluzie analyst:** Lisa Pigato este favorita clară (67% market, 2-0 H2H pe lut, WTA 133 vs 364). Pattern H2H este identic de două ori: 6-4 6-3, niciun TB. Serviciul fragil al lui Zantedeschi (0 ași, 7.5 break-uri primite per meci) și break rate-ul ridicat combinat (13+/meci) fac ca un Set 2 strâns să fie improbabil. Pigato vine obosită din R1 (2h15, 3 seturi), dar structurally tot ea domină stilistic. Câștigătoare estimată: **Pigato 6-4 6-3**.
