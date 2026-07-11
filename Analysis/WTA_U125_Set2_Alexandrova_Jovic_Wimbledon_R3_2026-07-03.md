# CoVe Analysis: Alexandrova vs Jovic — U12.5 Set 2
## Wimbledon 2026 | Round 3 | Iarbă | 03.07.2026 | 12:30 UK

---

## TRIPLE FILTER WORKFLOW v1.1 — PASUL 1: Model + Market

### Date model (1.5_WTA_Under12_5.csv + 1.2_WTA_Set1_Over_7_5.csv)

| Parametru | Valoare | Status |
|---|---|---|
| tb_p_cal | **0.0000** | ✅ ≤ 0.10 — semnal primar |
| p_hold_a (Alexandrova, grass) | 0.7634 | — |
| p_hold_b (Jovic, grass) | 0.7234 | — |
| hold_asym | 4.0pp (Alexandrova ține mai bine) | ✅ |
| blowout_score | 0 | ✅ |
| fatigue_flag_a / b | False / False | ✅ |
| UNSTABLE flag | — | ✅ absent |
| p_elo (Alexandrova win%) | 0.4638 (46.4%) | ≠ 0 ✅ |
| p_markov (Alexandrova win%) | 0.6114 (61.1%) | — |
| Gap Elo vs Markov | **14.76pp** | ✅ < 35pp |

### Robinhood Prediction Markets

URL: https://robinhood.com/us/en/prediction-markets/tennis/events/alexandrova-vs-jovic-jul-03-2026/

- **Jovic: 66%** (favorita pieței)
- **Alexandrova: 35%**

**P(favorita) = 66% → 60-74% range → CONTINUĂ, notează divergența**

**Divergență market vs p_markov:**
- Market spune Jovic favorita la 66%
- p_markov spune Alexandrova favorita la 61.1% → Jovic 38.9%
- Divergență = |66% - 38.9%| = **27.1pp > 15pp → INVESTIGHEAZA**

**Explicație divergență (CLARĂ):**
- Alexandrova 2026: 40.7% win rate (11-16) — sezon slab, eliminată timpuriu în clay season
- Jovic 2026: 68.3% win rate (28-13) — progres exponential
- Jovic swing iarbă: Queen's Club SF (beat Anisimova top-5), Wimbledon R1-R2 cu 9 game-uri pierdute total
- p_markov e bazat pe hold rates istorice — nu reflectă forma 2026
- **Concluzie: explicație valabilă, divergența justificată, continuăm**

Alte surse odds pentru corroborare:
- Dimers: Jovic 59% — [dimers.com](https://www.dimers.com/news/ekaterina-alexandrova-vs-iva-jovic-tennis-prediction-wimbledon-2026-ac)
- Bleacher Nation: Jovic 63.6% — [bleachernation.com](https://www.bleachernation.com/picks/2026/07/01/alexandrova-vs-jovic-prediction-at-the-wimbledon-friday-july-3/)
- SI.com: Jovic ~67% — [si.com](https://www.si.com/betting/ekaterina-alexandrova-vs-iva-jovic-prediction-odds-for-wimbledon-round-3-01kwhtc1tadt)

**PASUL 1: ✅ TRECE (cu notă divergență explicată)**

---

## TRIPLE FILTER WORKFLOW v1.1 — PASUL 2: TennisAbstract (iarbă)

Sursa: Sackmann/wta_matches_combined.csv (date locale)

### Ekaterina Alexandrova — iarbă career

**Total meciuri analizate: 56** ✅ (≥ 10)

**S2 TB rate: 11/56 = 19.6%** → zona neutră 15-25% → -1pp

**Meciuri cu TB în Set 2 — analiză contextuală:**

| Data | Turneu | Adversară | Rang adv. | Scor | Relevanță vs Jovic |
|---|---|---|---|---|---|
| 2016 | Wimbledon | Friedsam | ~80 WTA | 6-4 **7-6(1)** | MICĂ — 2016, adversară mid-level |
| 2019 | Eastbourne | Tomljanovic | ~50 WTA | 6-2 **7-6(1)** | MEDIE — adversară similară ca nivel |
| 2022 | s'Hertogenbosch | Yastremska | ~30 WTA | 2-6 **7-6(1)** 6-2 | MEDIE — pierdut S1, forțat în S2 |
| 2023 | s'Hertogenbosch | Sasnovich | ~80 WTA | 6-1 **7-6(1)** | MICĂ — adversară inferioară |
| 2023 | s'Hertogenbosch | Birrell | ~100 WTA | 6-4 **7-6(3)** | MICĂ — adversară inferioară |
| 2023 | Wimbledon | **Brengle** | ~100 WTA | 6-7(4) **7-6(5)** 7-6(7) | MICĂ — Brengle = serve-and-volley specialist, triple TB match atipic |
| 2024 | s'Hertogenbosch | Samsonova | ~20 WTA | 6-3 **6-7(1)** 6-1 | MARE — adversară top-20, nivel similar Jovic |
| 2025 | s'Hertogenbosch | Mertens | ~70 WTA | 2-6 **7-6(7)** 6-4 | MEDIE — pierdut S1, revenire în S2 |
| 2025 | Bad Homburg | Swiatek | #1 WTA | 6-4 **7-6(5)** | MICĂ — vs Swiatek = meci excepțional |
| 2025 | Bad Homburg | Sakkari | ~15 WTA | 6-3 **6-7(2)** 6-3 | MARE — adversară top-20, cel mai relevant! |
| **2025** | **Wimbledon** | **Sonmez** | ~60 WTA | 6-3 **7-6(1)** | MEDIE — turneu identic, suprafață identică |

**Meciuri cu relevanță MARE pentru azi (Jovic ~16 WTA):** 2 din 11 TB-uri S2 (vs Samsonova top-20, vs Sakkari top-15).
Ambele sunt meciuri unde Alexandrova CÂȘTIGĂ S2 prin TB — semn că poate "supraviețui" în seturi strânse vs adversare similare ca nivel.
**Nu schimbă semnificativ probabilitatea U12.5 S2 vs Jovic.**

**S1 TB → S2 TB pattern:**
- Meciuri cu TB în Set 1: 6
- Din care TB și în S2: **1/6 = 16.7%** → sub 20% → +1pp

**Net Alexandrova: -1pp (S2 TB rate) + 1pp (S1→S2 pattern) = 0pp**

### Iva Jovic — iarbă career

**Total meciuri analizate: 9** ❌ **SUB PRAGUL DE 10 → PASS STRICT conform workflow**

Date existente (informative):

| Data | Turneu | Adversară | Rang adv. | Scor | Relevanță |
|---|---|---|---|---|---|
| 2025 | Ilkley 125 | Lepchenko | ~200 WTA | 6-3 **6-7(2)** 6-0 | ZERO — adversară 125-level, veteran |
| 2025 | Wimbledon Q3 | Kawa | ~300 WTA | 6-3 **7-6(2)** | ZERO — calificări, adversară de nivel scăzut |

**Concluzie Jovic S2 TBs:** Ambele TB-uri S2 sunt vs adversare de nivel radical inferior (125-level/qualifier). ZERO relevanță pentru un meci vs Alexandrova top-20. În meciuri WTA main draw vs adversare comparabile, Jovic nu a avut niciun TB în Set 2 pe iarbă.

**PASUL 2: ⚠️ PASS STRICT** — Jovic < 10 meciuri iarbă (9 în Sackmann)
**NOTĂ ANALIST:** Sample borderline (9 ≈ 10). Conform tabelului de scor: "Sample borderline (8-12) → max 7/10". Contextual, TB-urile Jovic sunt complet irelevante. Continuăm analiza cu cap maxim 7/10.

---

## TRIPLE FILTER WORKFLOW v1.1 — PASUL 3: Context

| Factor | Alexandrova | Jovic |
|---|---|---|
| Fatigue | ✅ 0 seturi pierdute, odihnită | ✅ 0 seturi pierdute, odihnită |
| Meciuri Wimbledon 2026 | R1: 6-4 6-2 vs Udvardy; R2: 7-5 7-5 vs Tararudee | R1: 7-6(1) 6-0 vs Cristian; R2: 6-1 6-2 vs Maria |
| Days rest | ✅ normal (zi de pauză) | ✅ normal |
| Motivație | RIDICATĂ — salvează sezonul | RIDICATĂ — momentum, record carieră |
| UNSTABLE flag | ✅ absent | ✅ absent |

**PASUL 3: ✅ Context favorabil pentru U12.5**

---

## SCOR FINAL TRIPLE FILTER

| Criteriu | Status | Impact scor |
|---|---|---|
| tb_p_cal = 0.000 | ✅ | Semnal maxim |
| Gap ≤ 35pp (14.76pp) | ✅ | — |
| Robinhood Jovic 66% | 60-74%, divergență explicată | ✅ |
| Alexandrova S2 TB rate 19.6% | Zona neutră | -1pp |
| Alexandrova S1→S2 16.7% | Sub 20% | +1pp |
| Jovic sample < 10 | ⚠️ borderline | **Max 7/10** |
| Fatigue | ✅ | — |
| UNSTABLE | ✅ absent | — |

**SCOR FINAL U12.5 Set 2: 7/10** (blocat de sample Jovic borderline)

---

## ANALIZĂ PROFESIONISTĂ EXTINSĂ

### Profiluri jucătoare

#### Ekaterina Alexandrova (#19, seed 18, 31 ani, RUS)

**Stil de joc:** Baseliner agresivă, forehand puternic cu viteză mare de rachetă. Dictează punctele de la linie, lovește prin teren cu putere. Servă solidă (~6 aces/meci la Wimbledon 2026). Ideal pe iarbă rapidă — suprafața ei statistic cea mai bună (65.1% win rate career pe iarbă).

**Puncte slabe:** Inconsistentă mental în meciuri mari (niciodată dincolo de R4 la un major). Sezon 2026 slab pe clay (5 eliminări timpurii consecutive). Double faults ridicat: 4.12/meci — risc în momente decisive.

**Antrenori:** Evgeny Alexandrova (tată) + Petr Kralert + **Igor Andreev** (adăugat 2024, fost ATP #18, specialist tactic).
Sursa: [sportskeeda.com](https://www.sportskeeda.com/tennis/ekaterina-alexandrova-coach)

**Grass career:** 11-7 la Wimbledon, R4 atins în 2023 și 2025. Iarba este suprafața ei de vârf.
**Wimbledon 2026:** R1 def. Udvardy 6-4, 6-2 | R2 def. Tararudee 7-5, 7-5 — meciuri controlate, 0 seturi pierdute.
Sursa: [wtatennis.com](https://www.wtatennis.com/players/319007/ekaterina-alexandrova), [rallyher.com](https://rallyher.com/wimbledon-2026-women-results-draw-scores-schedule/)

#### Iva Jovic (#16, seed 16, 18 ani, USA)

**Stil de joc:** Baseliner modernă, agresivă, maturitate tactică remarcabilă pentru vârsta ei. Stă jos pe minge, joacă agresiv de la linie, mișcări naturale pe iarbă. Controlul mental sub presiune = arma sa principală.

**Puncte forte:** Compozitie în momente decisive, joc agresiv, adaptabilitate tactică rapidă. "Stays low, very aggressive" — ideal pe iarba Wimbledon.

**Antrenor:** Thomas Guttheridge (din juniori). Influența tatălui în formarea mentalității competiționale.
Sursa: [sportskeeda.com](https://www.sportskeeda.com/tennis/news-iva-jovic-parents-coach-serbian-heritage-ucla-connection-need-know-naomi-osaka-blockbuster-french-open)

**Swing iarbă 2026 complet:**
- Queen's Club: R1 def. Ruzic 6-3 6-4 | R2 def. Eala 6-2 6-2 | QF def. **Anisimova (seed #2)** 6-2 3-6 6-3 | SF pierdut Raducanu 2-6 2-6
- Wimbledon R1: def. Cristian 7-6(1), 6-0 | R2: def. Maria 6-1, 6-2 (68 min, 22 winners)

**Grand Slam 2026:** AO = Sferturi de finală (cel mai bun result carieră), RG = R3, Wimbledon = R3.
Sursa: [olympics.com](https://www.olympics.com/en/news/wimbledon-2026-iva-jovic-ends-tatjana-maria-s-grass-court-run-to-reach-third-round), [lta.org.uk](https://www.lta.org.uk/fan-zone/international/hsbc-championships/news/2026/2026-results-updates/)

---

### H2H și context comparativ

**Singura întâlnire directă:** US Open 2024 (hard), R2 — Alexandrova def. Jovic 4-6, 6-4, 7-5.
La momentul respectiv, Jovic avea 16 ani și a forțat 3 seturi la primul GS major. Acum, la 18 ani, cu 2 ani mai multă experiență și un swing de iarbă dominant, raportul de forțe s-a schimbat fundamental.
Sursa: [sofascore.com](https://www.sofascore.com/tennis/match/jovic-alexandrova/jfCsHmtd)

**Pe iarbă:** NICIO întâlnire anterioară.

---

### Motivație și miză

| Aspect | Alexandrova | Jovic |
|---|---|---|
| Sezon 2026 | Salvare — 40.7% win rate | Confirmare top-15 — 68.3% win rate |
| Record Wimbledon | R4 în 2023 și 2025 (apără puncte) | Prima dată în R3 — record carieră |
| Motivație | RIDICATĂ — Wimbledon = singura salvare a sezonului | RIDICATĂ — momentum exponential |
| Presiune | Mai mare — trebuie să confirme forma de iarbă | Liberă — every game is bonus |
| Mindset | Veterana cu experiență Grand Slam | 18 ani, "rien à perdre", energie pură |

**Avantaj psihologic: Jovic.** Joacă fără povara așteptărilor istorice. A recunoscut că era "very nervous" în calificări acum un an — acum e seed #16 și joacă cu naturalețe. Alexandrova are presiunea apărării punctelor din R4 2025.

---

### Condiții fizice

Ambele jucătoare sunt odihnte — 2 meciuri straight sets fiecare, tempo normal de turneu. **Nicio accidentare semnalată** pentru niciuna.

Jovic: 5 ore jucate total în 2 meciuri (estimat). Alexandrova: 5-6 ore total. **Egalitate completă pe condiție fizică.**

---

### Condiții teren și meteo

- **Temperatura:** Max 25°C, fără ploaie, 12+ ore soare — condiții ideale
- **Vânt:** 13 mph — vânt moderat, poate afecta serva
- **Iarba (Ziua 5):** Uzată în zona baseline, mai alunecoasă — avantaj pentru jucătoarele cu mișcare joasă (Jovic confirmată ca "stays low")
- **Concluzie meteo:** Favorabil jocului agresiv, rapid. Iarba uzată din ziua 5 tinde să producă seturi mai rapide — favorizează U12.5.

Sursa: [theweatheroutlook.com](https://www.theweatheroutlook.com/forecast/uk/wimbledon), [espn.com](https://www.espn.com/tennis/story/_/id/49155718/wimbledon-2026-today-order-play-daily-schedule-results-weather-forecast)

---

### Statistici comparative relevante (TennisRatio 2026)

| Metric | Jovic | Alexandrova |
|---|---|---|
| Win % 2026 | **68.3%** | 40.7% |
| 1st serve win% Wimbledon | **77%** | 71% |
| 2nd serve win% Wimbledon | **61%** | 48% |
| Aces/meci | 3 | 6 |
| Under 12.5 per set | 88% | **92%** |
| TB rate per match | 22% | 28% |
| Avg games per set | 9.37 | 9.37 |
| Breaks per match | 3.51 | 2.40 |

**Alexandrova: mai multă putere de servă (6 aces/meci vs 3), dar Jovic câștigă mai mult pe 1st și 2nd serve.**
**Ambele: medie 9.37 games/set — structurală favorabilă pentru U12.5.**

Sursa: [TennisRatio H2H](https://tennisratio.com)

---

### Predicție structurală Set 2

**Scenariile posibile:**

**Scenariu A (cel mai probabil ~55%):** Jovic câștigă S1 rapid (6-4 sau 6-3) → continuă agresivă în S2, Alexandrova nu poate menține nivel → S2 = 6-3 sau 6-2. **U12.5 ✅**

**Scenariu B (~25%):** Alexandrova câștigă S1 prin bătălie (7-5 sau 6-4) → Jovic răspunde agresiv în S2, nu repetă pasivitatea → S2 = 6-3 sau 6-4. **U12.5 ✅**

**Scenariu C (~15%):** Meci echilibrat, S1 ajunge la 6-5 sau mai lung → S2 competitiv, tensiune mentală → S2 poate fi 7-5 sau 6-5. **U12.5 ✅ (7-5 = 12 games, sub 12.5)**

**Scenariu D — RISC (~5%):** S1 merge la tiebreak → ambele jucătoare intră în momentum de TB → S2 tensionat → TB posibil. **U12.5 ❌**

**P(U12.5 S2) contextuală: ~93-95%** — confirmă tb_p_cal = 0.000 din model.

---

### Cine câștigă meciul?

**Verdict:** Jovic favorita clară. Piața la 66% subevaluează ușor avantajul ei de formă — Dimers o pune la 59%, dar 2026 forma și dominanța pe iarbă sugerează 65-70%.

**Motivele:**
1. Jovic 2026 = 68.3% win rate vs Alexandrova 40.7% — diferență de 27pp în 40+ meciuri
2. Servă mai eficientă la Wimbledon (77% vs 71% on 1st serve)
3. Maturitate tactică crescută față de US Open 2024 (când a forțat deja 3 seturi la 16 ani)
4. Queen's Club SF vs Anisimova (top-5) = dovadă că poate bate adversare de nivel Alexandrova și mai mult
5. Iarba uzată din ziua 5 favorizează mișcarea joasă a lui Jovic

**Contra:** Alexandrova are servă cu mai multă putere (6 aces/meci), hold rate mai bun pe iarbă din model (76.3% vs 72.3%), experiență mai mare în situații de presiune la Grand Slam.

**Predicție:** Jovic def. Alexandrova 6-4, 6-3 sau 6-3, 6-4. Posibil și 7-5, 6-3 dacă Alexandrova rezistă în S1.

---

## VERDICT FINAL U12.5 SET 2

| Factor | Evaluare |
|---|---|
| Model (tb_p_cal = 0.000) | ✅ Semnal maxim |
| Robinhood market (Jovic 66%) | ✅ Trecut, divergență explicată |
| Alexandrova S2 TB rate (19.6%) | Neutru (-1pp + 1pp = 0) |
| Jovic sample (9 meciuri) | ⚠️ Borderline — max 7/10 |
| Jovic S2 TBs (2 vs qualifier/125) | ✅ Irelevante contextual |
| Formă Jovic (dominant, seturi rapide) | ✅ Favorabil U12.5 |
| Condiții meteo / teren | ✅ Favorabil joc rapid |
| Motivație / oboseală | ✅ Neutru |
| Predicție meci | Jovic domină → seturi decisive | ✅ |

**SCOR FINAL: 7/10**

**⚠️ Limitat la 7/10** din cauza sample-ului Jovic pe iarbă (9 meciuri < pragul de 10 din workflow). Dacă sample-ul ar fi complet, scorul ar fi 8/10.

**RECOMANDARE:** PASS conform triple filter strict (Jovic < 10 meciuri iarbă).

Dacă userul decide să continue cu 7/10:
- Probability U12.5 S2 contextuală: **~93-95%**
- Odds minime pentru recomandare: ≥ 1.10
- Tip pick: speculativ 7/10, nu standard de portofoliu

---

**Fișier generat:** 2026-07-03
**Workflow aplicat:** Triple Filter U12.5 Set 2 v1.1
**Surse principale:**
- [wtatennis.com — Alexandrova](https://www.wtatennis.com/players/319007/ekaterina-alexandrova)
- [olympics.com — Jovic R2 Wimbledon](https://www.olympics.com/en/news/wimbledon-2026-iva-jovic-ends-tatjana-maria-s-grass-court-run-to-reach-third-round)
- [lta.org.uk — Queen's Club results](https://www.lta.org.uk/fan-zone/international/hsbc-championships/news/2026/2026-results-updates/)
- [sofascore.com — H2H US Open 2024](https://www.sofascore.com/tennis/match/jovic-alexandrova/jfCsHmtd)
- [si.com — R3 preview](https://www.si.com/betting/ekaterina-alexandrova-vs-iva-jovic-prediction-odds-for-wimbledon-round-3-01kwhtc1tadt)
- [bleachernation.com — odds analysis](https://www.bleachernation.com/picks/2026/07/01/alexandrova-vs-jovic-prediction-at-the-wimbledon-friday-july-3/)
- [robinhood.com — prediction markets](https://robinhood.com/us/en/prediction-markets/tennis/events/alexandrova-vs-jovic-jul-03-2026/)
- TennisAbstract / Sackmann wta_matches_combined.csv (date locale)
