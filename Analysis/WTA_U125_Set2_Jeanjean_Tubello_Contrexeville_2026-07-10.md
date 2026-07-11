# WTA U12.5 Set 2 — CoVe Full (Model Confirmat, PREMIUM)
## Leolia Jeanjean vs Alice Tubello
**Turneu:** Grand Est Open 88 — WTA 125, Contrexeville (Vosges), France  
**Suprafață:** Lut (outdoor) | **Tur:** QF (round=3) | **Data:** 10.07.2026, ~17:30 CEST  
**Surse:** CSVs model, CoreTennis.net (ID Tubello: 108523, Jeanjean: 9983), TennisStat, Robinhood Markets, TennisTemple, Meteoblue

---

## STATUS MODEL

**Prezent în `1.5_WTA_Under12_5.csv`** — model complet, semnal PREMIUM.

```
tb_p_raw   = 0.0465   ← extrem de mic
tb_p_cal   = 0.0927   ≤ 0.10 ✅
p_u125     = 0.9073   (90.73% P(U12.5 S2))
blowout_score = 8     ← ridicat
hold_asym  = 0.2057   ← asimetrie RIDICATĂ (Tubello domină serviciu vs Jeanjean)
fatigue_flag_a = True ← Jeanjean obosită (3 seturi ieri vs Jones)
fatigue_flag_b = True ← Tubello (3 seturi în R1 vs Ren)
had_3sets_7d_a = True
had_3sets_7d_b = True
unstable_reason = ''  ✅
premium_u125 = YES    ← semnal maxim
recommended  = True
```

**Din `1.1_WTA_Winner.csv`:**
```
p_markov = 0.1324   (Jeanjean 13.24% prin Markov — Tubello domină hold rates)
p_elo    = 1.0      ← ARTIFICIAL: Tubello nu are date în Sackmann → model assign 100% Jeanjean
gap Elo/Markov = |1.0 - 0.1324| × 100 = 86.76pp  → > 35pp ⚠️
data_source = sackmann/sackmann  ✅
predicted_winner = Alice Tubello (p_cal=0.5144 = 51.4%) — extrem de echilibrat
expected_games = 21.82  ← mic (blowout așteptat)
days_rest_a = 1  (Jeanjean — meci ieri vs Jones)
days_rest_b = 1  (Tubello — meci ieri vs Werner)
p_hold_a (Jeanjean) = 0.4163  (41.6% hold rate — extrem de breakabilă)
p_hold_b (Tubello)  = 0.6220  (62.2% hold rate — servitoare solidă)
```

---

## TRIPLE FILTER — PASUL 1

### Criterii CSV

| Criteriu | Valoare | Semnal |
|---|---|---|
| tb_p_cal | **0.0465** ≤ 0.10 | ✅ |
| p_elo | 1.0 (Tubello fără date Sackmann → artificial) | ⚠️ flag |
| Elo/Markov gap | **86.76pp** > 35pp | ⚠️ SKIP trigger |
| UNSTABLE flag | Absent | ✅ |
| premium_u125 | **YES** | ✅ |

### Nota critică — Gap artificial, nu conflict real de formă

`p_elo = 1.0` pentru Jeanjean apare deoarece **Alice Tubello nu are date în baza Sackmann** — modelul nu poate calcula Elo pentru Tubello, deci îi asignează Jeanjean 100% probabilitate Elo. Aceasta nu este o divergență reală de formă.

Verificare externă:
- **Jeanjean TennisStat Elo: 577** (WTA 132, carieră solidă)
- **Tubello TennisStat Elo: 326** (WTA 229, nivel challenger)
- Piața (Robinhood) CONFIRMĂ Jeanjean ca favorită → gap Markov/market explicat prin absența datelor Sackmann, nu printr-un conflict real

**Procedură:** Override parțial per `feedback_pelo_zero_manual_cove.md` — CoVe manual cu date empirice CoreTennis (≥10 meciuri lut disponibile).

### Robinhood Market Check

**URL:** robinhood.com/us/en/prediction-markets/tennis/events/jeanjean-vs-tubello-jul-10-2026/

| Jucătoare | P(câștig) | Nivel |
|---|---|---|
| Leolia Jeanjean | **64%** | 60-74% zonă → continuă, notează divergența |
| Alice Tubello | 39% | — |

**Praguri:**
- P(favorita) ≥ 60% ✅ → continuă analiza (nu SKIP)
- P(favorita) < 75% → class gap NU confirmat complet

**Divergență Robinhood vs p_markov:** |0.64 − 0.1324| × 100 = **50.76pp** > 15pp → investigheaza

**Investigare:** Divergența masivă se explică INTEGRAL prin absența datelor Sackmann pentru Tubello. Robinhood (64% Jeanjean) și Elo (577 >> 326) sunt aliniate → market CONFIRMĂ Jeanjean ca favorită clară. Markov supraevaluează Tubello pe baza hold rates fără context de nivel și carieră.

**PASUL 1: PARȚIAL** — gap artificial (nu conflict real), Robinhood 64% trece pragul minim (≥60%). Continuăm cu flag explicit.

---

## TRIPLE FILTER — PASUL 2 (CoreTennis — Clay)

### Alice Tubello — Clay S2 TB Rate (CoreTennis ID 108523)

**Sample:** 89 meciuri lut total (2023-2026) — sample SOLID

**Meciuri lut 2-seturi cu S2 TB (relevant pentru U12.5 S2):**

| Data | Turneu | Adversară | Ranking apx. | Scor | S1 | S2 |
|---|---|---|---|---|---|---|
| 2026-06-29 | W50 Stuttgart-Vaihingen | Julie Struplova | ~350-500 | 7-6(3) **7-6(8)** | **TB** | **TB** ← cascade |
| 2025-09-22 | W35 Santa Margherita | Nellie Taraba Wallberg | ~500-700 | **7-6(1) 7-6(2)** | **TB** | **TB** ← cascade |
| 2026-03-16 | W35 San Gregorio | Eva Bennemann | ~600+ | 6-3 **7-6(4)** | no TB | **TB** |
| 2025-09-01 | W50 Saint-Palais | Sofia Shapatava | ~350 | 6-2 **7-6(1)** | no TB | **TB** |
| 2024-09-30 | W35 Reims | Olga Helmi | ~600+ | 6-3 **7-6(2)** | no TB | **TB** |

**S2 TB rate în 2-set clay matches: 5/~50 = ~10%** → ≤15% **+1pp confirmare** ✅

**Meciuri lut 2-seturi cu S1 TB fără cascade:**

| Data | Turneu | Adversară | Scor | Cascade? |
|---|---|---|---|---|
| 2026-06-29 | W50 Stuttgart QF | Tessa Brockmann | **7-6(7)** 6-2 | ❌ no cascade |
| 2025-09-01 | W50 Saint-Palais F | Julia Avdeeva | **7-6(5)** 6-2 | ❌ no cascade |
| 2026-04-27 | Saint-Malo R1 | Lucie Nguyen Tan | **7-6(4)** 6-3 | ❌ no cascade |
| 2024-08-12 | W35 Arequipa SF | Jazmin Ortenzi | **7-6(2)** 6-3 | ❌ no cascade |

**S1 TB → S2 cascade (lut, 2-seturi):**
- Total S1 TB în 2-set matches: 6 meciuri
- Din care S2 și TB: 2 (Struplova, Wallberg)
- **Cascade rate: 2/6 = 33.3%** → exact la pragul critic (>33% = max 6/10 per grid)

### Analiza contextuală cascadelor Tubello

**CRITIC — ambele cascade au fost contra adversarelor ITF W35/W50 nivel 350-700:**

| Cascade | Adversară | Ranking | Context |
|---|---|---|---|
| Struplova (W50 Stuttgart) | Julie Struplova | ~400 | W50 R16, meci echilibrat ITF-level |
| Wallberg (W35 Santa Margherita) | Nellie Taraba Wallberg | ~600 | W35 R16, adversar la nivel W35 |

**Meciuri NON-cascade (S1 TB dar S2 net):**
- Brockmann (W50 QF, ~350): **7-6(7)** 6-**2** → Tubello câștigă S2 6-2 net după S1 TB
- Avdeeva (W50 F, ~300): **7-6(5)** 6-**2** → Tubello câștigă S2 6-2 net după S1 TB
- Nguyen Tan (WTA 125 R1): **7-6(4)** 6-**3** → S2 net după S1 TB

**Pattern clar:** Cascade apare EXCLUSIV contra adversarelor de nivel W35/W50 (ranking 400-700+). Contra adversarelor de nivel mai bun (WTA 125, W50 F), Tubello câștigă S2 net după S1 TB.

**Jeanjean (WTA 132, Elo 577)** este cea mai bună adversară întâlnită de Tubello în 2026. Dinamica matchului va fi complet diferită față de meciurile W35/W50 unde cascadele au apărut.

---

### Leolia Jeanjean — Clay S2 TB Rate (CoreTennis ID 9983)

**Sample:** ~20 meciuri în 2 seturi pe lut (2024-2026)

**S2 TB confirmate pe lut:**

| Data | Turneu | Adversară | Ranking | Scor | S1 | S2 |
|---|---|---|---|---|---|---|
| Jun 2024 | Roland Garros R1 | Kaitlin Quevedo | ~200 | **7-6(5) 7-6(2)** | **TB** | **TB** ← cascade |

**S2 TB rate Jeanjean pe lut: 1/~20 = ~5%** → sub 15% **+1pp** ✅

**S1 TB → S2 cascade Jeanjean (lut):**

| Meci | S1 | S2 | Cascade? |
|---|---|---|---|
| vs Quevedo (RG 2024) | 7-6(5) | 7-6(2) | ✅ cascade |
| vs Haddad Maia (Roma 2025) | 7-6(6) | 6-4 | ❌ |
| vs Tomljanovic (Roma 2025) | 7-6(5) | 5-7 (Jeanjean pierde S2 net) | ❌ |
| vs Paolini (Roma 2025) | 6-7(4) | 6-2 | ❌ |
| vs Gibson (Saint-Malo 2025) | 6-7(5) | 6-2 | ❌ |

**Cascade rate Jeanjean: 1/5 = 20%** → 20-33% zonă → neutru ✅

**Analiza singurei cascade Jeanjean pe lut:**
- Adversară: **Kaitlin Quevedo** (columbiacă, WTA ~200 la acea vreme, jucătoare solidă de lut)
- Turneu: **Roland Garros** (Grand Slam, presiune maximă, cel mai lent lut)
- Context: RG = suprafața cea mai lentă, ambele jucătoare cu serviciiu solid → seturi lungi caracteristice RG
- Relevanță pentru azi: RG clay ≠ Contrexeville clay (Contrexeville = lut mai rapid, căldură 32°C = minge rapidă)

---

### PASUL 2 — Scor intermediar

| Metric | Tubello | Jeanjean | Verdict |
|---|---|---|---|
| S2 TB rate (lut, 2-set) | ~10% ≤15% | ~5% ≤15% | **+1pp** ambele bune |
| S1→S2 cascade | **33.3%** (la prag) | 20% (zonă OK) | ⚠️ Tubello exact la limita |
| Sample | ~50 2-set (solid) | ~20 (ok) | ✅ |
| Cascade context | EXCLUSIV vs W35/W50 (400-700) | 1 la RG (cel mai lent lut) | Mitigat de nivel adversare |

**PASUL 2: TRECE CU FLAG** ⚠️ (cascade Tubello 33.3% = exact la pragul >33%)

---

## TRIPLE FILTER — PASUL 3 (Context manual)

### Condiție fizică și path la turneu

| Factor | Jeanjean | Tubello |
|---|---|---|
| R1 (Jul 7) | W vs Monnot (FRA, local) 6-1 6-4 | W vs Ren Yufei **2-6 7-5 6-4** (3 seturi!) |
| R2 (Jul 9) | W vs Jones în **3 seturi** (date confirmate: had_3sets_7d_a=True) | W vs Werner **6-3 6-3** (dominator, 50 min) |
| Total seturi | **3 seturi ieri** (mai obosită) | ~2.5 seturi ieri (R1: 3S mai demult, R2: 2S ieri) |
| days_rest | 1 | 1 |
| fatigue_flag | **True** (activ) | **True** (activ) |
| had_3sets_7d | **True** (ieri!) | True (R1 acum câteva zile) |

**Nota fatigue:** Jeanjean a jucat 3 seturi IERI (meci dificil vs Jones), iar Tubello a jucat 3 seturi în R1 (acum 3-4 zile). Tubello vine mai fresh din perspectivă imediată — R2 vs Werner a fost rapid și dominat (6-3 6-3). Avantaj modest Tubello.

**Dar:** Tubello R1 vs Ren (2-6 7-5 6-4) a arătat că poate fi pusă sub presiune chiar și de adversare mai slabe. Și-a revenit bine.

### Motivație și miză

- **Jeanjean:** Prima QF WTA 125 din 2026 (sezon în reconstrucție, WTA 132). Un loc în SF reprezintă un progres de ranking important. Joacă **ACASĂ** (Franța) — avantaj psihologic semnificativ la Contrexeville, suport local masiv. Forma recentă: WWLWWWW (7 meciuri).
- **Tubello:** Prima QF WTA 125 din carieră? (WTA 229, nivel challenger). Are MOMENTUM excepțional în 2026 — 65.9% win rate pe an (27/41 meciuri). A câștigat titlul la W50 Stuttgart acum 2 săptămâni. La Contrexeville a dominat Werner 6-3 6-3. Formă: WWWWLWW.
- **Prize:** $4,608 + 49 puncte WTA → motivație maximă pentru ambele (mai ales Tubello care nu câștigă regulat sume WTA 125)
- **Jeanjean homecourt:** Factor critic pe lut francez — suportul publicului, condiții familiare, experiența supliniror pe lut francez (Roland Garros, Saint-Malo, Strasbourg)

### Stil de joc și hold rate analysis

**Date TennisStat (2026):**

| Metric | Jeanjean | Tubello | Match total |
|---|---|---|---|
| Ași/meci | 2.93 | 0.55 | 3.48 |
| DF/meci | 4.20 | 3.36 | 7.56 |
| TB/meci | **0.35** (32%) | **0.17** (15%) | 0.26 |
| Avg games/set | **9.85** | **8.80** | 9.33 |
| Over 12.5/set | **24%** | **7%** | **16%** |
| S2 Win% | 50% | **66%** | — |
| Breaks primite/meci | 3.17 | 3.64 | **6.81** |
| S1 Win% | 62% | 61% | — |

**Hold rate analysis din model:**

| Metric | Jeanjean (server) | Tubello (server) |
|---|---|---|
| p_hold | 0.4163 (42%) | 0.6220 (62%) |
| Break rate primită | **58%** | **38%** |
| Breaks/set (6 service games) | 6 × 0.58 = **3.48** | 6 × 0.38 = **2.28** |

**Combined breaks per set:** 3.48 + 2.28 = **5.76 breaks/set** → structural anti-TB  
**TennisStat confirmare:** 6.81 total breaks/meci combined ← consistent

La 5.76 breaks/set, ajungerea la 6-6 este structural dificilă — trebuie ca ambele jucătoare să "balanseze" break-urile perfect la 5-5 și să continue până la 6-6. Structurally improbabil.

**Paradoxul Markov vs Market:**
- Markov (p_markov=0.1324 Jeanjean): Tubello câștigă 86.76% pe baza hold rates (0.622 >> 0.416)
- Market (Robinhood 64% Jeanjean): Jeanjean câștigă prin Elo/skill nivel superior (577 >> 326)
- Explicație: Jeanjean "compensează" hold rate scăzut prin calitate în momentele cheie, returnuri agresive, mental superior în meciuri serioase. Tubello are hold rate bun dar la nivel W35/W50 — contra unui WTA 132, break rate-ul ei real va fi mai mare decât 38%.

### Stil tehnic și compatibilitate U12.5 S2

- **Jeanjean:** Jucătoare de topspin cu forehand puternic, servitoare cu 2.93 ași/meci (cel mai important serviciu din acest meci). Joacă cu lut french-style (saint-malo, roland garros background). **9.85 avg games/set — cel mai ridicat din ambele jucătoare** → tendință spre seturi mai lungi, competitive.
- **Tubello:** Compact player, serviciu modest (0.55 ași/meci!), solid de bază din fundal. Câștigă prin consistență, nu prin dominanță. **8.80 avg games/set** → seturi mai scurte în medie. Scor S2 66% → câștigă S2 des când ajunge acolo.
- **Compatibilitate U12.5:** Jeanjean are 24% over 12.5 per set (cel mai ridicat) — risc. Tubello 7% — excelent. Average 16% → 84% șanse per set să fie U12.5. Pe S2 specific (factor de fatigue + directional momentum), probabil mai bun.

### Context psihologic și mental

| Factor | Jeanjean | Tubello |
|---|---|---|
| Nivel meci QF WTA 125 | Experiată (WTA 132) | Prima QF WTA 125 posibil |
| Presiune | Favorită (64%), trebuie să livreze | Underdog (36%), nimic de pierdut |
| Homecourt | **DA** (FRA, suport public) | Nu |
| Recent big wins | Jones (3 seturi, revenire) | Werner (dominator, 6-3 6-3) |
| Momentum la turneu | Câștigat greu (Jones 3S) | Câștigat ușor (Werner 2S) |
| Mindset | Concentrată dar potentially obosită | Fresh, liberă, cu momentum clar |

**Risc psihologic Tubello:** Lipsa experienței QF WTA 125 poate fi un factor negativ la momente cheie (breakpoint-uri, 5-5 în set). Jeanjean a mai jucat la acest nivel.

### Condiții meteo — Contrexeville, 10.07.2026

| Parametru | Valoare |
|---|---|
| Temperatură | **32°C** (heatwave alert: Vigilance orange canicule) |
| Condiții | Cer senin, soare puternic (UV Index 7) |
| Vânt | 18 km/h, rafale 34 km/h (NE) |
| Precipitații | 0% |

**Impact 32°C pe lut:** Minge caldă = mai rapidă, rally-uri mai scurte = mai multe break-uri rapide → structural anti-TB. Jucătoarele obosesc mai repede în S2 → break-urile din S2 tind să fie mai rapide, setul se termină mai devreme.

---

## ANALIZA DETALIATĂ S2 TB-URI TUBELLO (calitate adversare)

| S2 TB meci | Adversară | Ranking est. | Context | Relevanță azi |
|---|---|---|---|---|
| Struplova (cascade) | Julie Struplova | ~400 | W50, meci echilibrat, dublu-TB total | SCĂZUTĂ — adversar W50 level, Jeanjean e mult mai bună |
| Wallberg (cascade) | Nellie Taraba Wallberg | ~600 | W35, adversar ITF low-level | SCĂZUTĂ — adversar W35 level |
| Bennemann | Eva Bennemann | ~500 | W35, set-ul a scăpat de sub control | SCĂZUTĂ |
| Shapatava | Sofia Shapatava | ~350 | W50, veterana dificilă | MEDIE — mai relevanță |
| Helmi | Olga Helmi | ~600 | W35 | SCĂZUTĂ |

**Concluzie:** Niciun S2 TB Tubello nu a apărut contra unei adversare de nivel WTA 125 sau WTA mainstream. Toate au fost la W35/W50 contra jucătoarelo ranked 350-700+. Contra Jeanjean (WTA 132, Elo 577), care o presează mai intens pe serviciu și în momente cheie, dinamica este fundamental diferită.

**Cele 2 cascade** (Struplova, Wallberg) au apărut în meciuri unde AMBELE jucătoare aveau hold rates apropiate la nivel W35/W50 → meciul era echilibrat → S1 a mers la TB → S2 similar. Contra Jeanjean (care rupe Tubello la 38%), setul nu va fi "echilibrat" la acel nivel — Jeanjean va domina serviciul Tubello mult mai mult.

---

## H2H

**Nicio întâlnire anterioară** — prima dată la nivel profesionist (confirmat TennisStat).

---

## ESTIMARE CÂȘTIGĂTOARE ȘI SCOR

| Factor | Jeanjean | Tubello |
|---|---|---|
| WTA Rank | 132 | 229 |
| Elo (TennisStat) | **577** | 326 |
| Robinhood | **64%** | 36% |
| Hold rate | 42% (mai breakabilă) | **62%** (mai solidă) |
| Form 2026 | 55.9% | **65.9%** |
| Homecourt | **DA** (Franța) | Nu |
| Fatigue | 3 seturi ieri ← | 2 seturi ieri → mai fresh |
| QF experience | Da | Primă dată WTA 125 level |

**Câștigătoare estimată: Leolia Jeanjean** (64% market, Elo 577 >> 326)

Cum câștigă Jeanjean vs un hold rate de 62% Tubello? Prin:
1. **Returnuri** agresive pe serviciu modest al lui Tubello (0.55 ași/meci) — poate sparge frecvent
2. **Serviciu puternic** al ei (2.93 ași/meci) — compensează parțial hold rate scăzut
3. **Experiența QF** și **homecourt** — momente cheie gestionate mai bine
4. La 32°C, ambele jucătoare sunt forțate — Tubello are mai puțin de mers înapoi la nivel WTA

| Scenariu | Probabilitate | S2 result |
|---|---|---|
| Jeanjean 6-4 6-3 | ~25% | U12.5 ✅ |
| Jeanjean 6-3 6-4 | ~20% | U12.5 ✅ |
| Jeanjean 6-4 6-2 | ~12% | U12.5 ✅ |
| Jeanjean 7-5 6-3 | ~10% | U12.5 ✅ |
| Jeanjean 6-4 7-5 | ~8% | U12.5 ✅ |
| Jeanjean 6-3 7-6 (S2 TB) | ~5% | ❌ |
| Tubello 6-3 6-4 (upset) | ~10% | U12.5 ✅ |
| Tubello 7-5 6-3 | ~6% | U12.5 ✅ |
| Tubello 7-5 7-6 | ~4% | ❌ |

**P(U12.5 S2 total estimat): ~90-91%** (consistent cu tb_p_cal=9.27%)  
**Estimare scor:** Jeanjean **W 6-4 6-3** sau **6-3 6-4**

---

## SCOR CoVe FINAL

| Criteriu | Valoare | Verdict |
|---|---|---|
| Model tb_p_raw | **0.0465** | ✅ |
| Model tb_p_cal | 0.0927 ≤ 0.10 | ✅ |
| premium_u125 | **YES** | ✅ |
| blowout_score | **8** | ✅ |
| expected_games | **21.82** | ✅ |
| Elo/Markov gap | **86.76pp** (artificial — Tubello fără Sackmann) | ⚠️ |
| Robinhood | **64%** (60-74% — class gap moderat) | ⚠️ < 75% |
| S2 TB Tubello (lut, 2-set) | ~10% ≤15% | **+1pp** ✅ |
| S2 TB Jeanjean (lut) | ~5% ≤15% | **+1pp** ✅ |
| Cascade Tubello (lut, 2-set) | **33.3%** (exact la prag >33%) | ⚠️ la limita grid cap |
| Cascade Jeanjean (lut) | 20% | ✅ |
| Hold asym | 0.2057 (ridicat) | anti-TB structural |
| Breaks combined/set | **5.76** | anti-TB structural |
| Fatigue Jeanjean | had_3sets_7d=True, ieri 3 seturi | ⚠️ |
| Weather | 32°C, canicule alert | anti-TB (minge rapidă) |

**SCOR FINAL: 6/10** (grid cap: cascade Tubello 33.3% > 33% → max 6/10 per reguli)

---

## ATENȚIONARE BACKTEST ȘI SURFACE MINIMUM

**Conform `reference_u125_s2_backtest_surfaces.md`:**
- **Clay 8/10 + Robinhood → HR 91.3%** ← nivel minim recomandat
- **Clay 9/10 + Robinhood → HR 93%** ← nivel target

**Scor 6/10 = SUB MINIMUL STANDARD PE LUT** → nu există date backtest separate pentru 6/10.

**Motivele care blochează 8/10:**
1. **Cascade Tubello 33.3% → grid cap 6/10** (strict rules: >33% = max 6/10)
2. **Robinhood 64% < 75%** → class gap neconfirmat complet (pentru 8/10+RH pe lut ai nevoie ≥75%)
3. **Elo/Markov gap artificial 86.76pp** → calitate datelor model compromisă pentru winner market

**Dacă cascade Tubello ar fi fost contextual ajustat la <33% (argument valabil — toate cascadele cu W35/W50):**
→ Scor ar fi fost **7/10** (S2 TB ~10% ≤15%, cascade ajustat <33%, Robinhood 64% < 75%)
→ **Tot sub minimul de 8/10+RH pe lut**

---

## VERDICT

```
MARKET:    WTA U12.5 Set 2
MECI:      Jeanjean vs Tubello — Contrexeville WTA 125, Clay, QF
DATA:      10.07.2026, ~17:30 CEST
MODEL:     tb_p_raw=0.0465 ✅ | premium=YES ✅ | blowout=8 ✅ | gap=86.76pp ⚠️
ROBINHOOD: Jeanjean 64% ⚠️ (moderat, < 75%)
CASCADE:   Tubello 33.3% (la prag) ⚠️ | Jeanjean 20% ✅
SCOR:      6/10 — SUB MINIMUL PE LUT (8/10+RH)
BET:       NU RECOMANDĂM — sub surface standard (clay min 8/10 + RH ≥75%)
```

**Concluzie analyst:** Model puternic (premium=YES, blowout=8, 9.27% cal) și structural anti-TB (5.76 breaks/set, 32°C). Dar combinația de 3 factori blochează recomandarea: cascade Tubello 33.3% (chiar dacă contextual mai mic), Robinhood sub 75% (64%), și gap Elo/Markov 86.76pp (artificial, dar care reflectă incertitudinea datelor). Jeanjean este favorita clară (64% market, Elo 577 >> 326) și va câștiga probabil 6-4 6-3 sau 6-3 6-4. Dar incertitudinea modelului + cascade borderline + Robinhood la nivel mediu = risc prea mare pentru a atinge pragul de calitate 8/10 impus pe lut.

**Concluzie estimare meci:** Jeanjean W **6-4 6-3** (cel mai probabil). Tubello joacă liber ca underdog cu momentum, dar Jeanjean are clasa, homecourt-ul și serviciul pentru a controla Q2.

---

*Surse: CoreTennis.net (Tubello ID 108523, Jeanjean ID 9983), TennisStat.com, Robinhood Markets, TennisTemple Contrexeville 2026 draw, Meteoblue Contrexeville 10.07.2026*
