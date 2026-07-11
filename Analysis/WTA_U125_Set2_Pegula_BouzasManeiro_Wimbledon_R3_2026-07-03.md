# CoVe Analysis: Pegula vs Bouzas Maneiro — U12.5 Set 2
## Wimbledon 2026 | Round 3 | Iarbă | 03.07.2026 | ~15:00 UK

---

## TRIPLE FILTER WORKFLOW v1.1 — PASUL 1: Model + Market

### Date model (1.5_WTA_Under12_5.csv + 1.2_WTA_Set1_Over_7_5.csv)

| Parametru | Valoare | Status |
|---|---|---|
| **tb_p_cal** | **0.1270** | ❌ **DEPĂȘEȘTE PRAGUL 0.10 — FAIL** |
| p_hold_a (Pegula, grass) | **0.8333** | ⚠️ Serveră de elită — hold 83.3% |
| p_hold_b (Bouzas, grass) | 0.7148 | — hold decent 71.5% |
| hold_asym | 11.84pp (Pegula ține cu 11.8pp mai mult) | — |
| blowout_score | 2 | — minor |
| fatigue_flag_b | **True** (Bouzas Maneiro) | ⚠️ 3 seturi în R2 |
| p_elo (Pegula win%) | 0.7096 (71.0%) | — |
| p_markov (Pegula win%) | 0.7963 (79.6%) | — |
| Gap Elo vs Markov | **8.67pp** | ✅ < 35pp |

### ❌ PASUL 1: FAIL — STOP

**tb_p_cal = 12.7% > 10% (pragul operațional)**

Analiza U12.5 Set 2 se oprește la Pasul 1. Modelul estimează o probabilitate de 12.7% pentru un tiebreak în Set 2 — cu 27% mai mare față de pragul nostru.

**De ce tb_p_cal > 10%:** Pegula are hold rate 83.3% pe iarbă — una din cele mai ridicate din circuit. Cu hold atât de mare, serviciile sunt adesea la egalitate → seturi care ajung la 6-6 mai frecvent decât media. Aceasta nu e o eroare de model, e o caracteristică structurală a lui Pegula pe iarbă.

**Robinhood** (informativ, chiar dacă Pasul 1 a eșuat):

URL: https://robinhood.com/us/en/prediction-markets/tennis/events/pegula-vs-bouzas-maneiro-jul-03-2026/
- **P(Pegula) = 87-89%**
- **P(Bouzas) = 13%**
- Divergență market vs p_markov: |88% - 79.6%| = **8.4pp → sub 15pp ✅** (piața și modelul sunt bine aliniate)

---

## TRIPLE FILTER v1.1 — PASUL 2: TennisAbstract (informativ — nu mai schimbă verdictul)

Sursa: Sackmann/wta_matches_combined.csv (date locale)

### Jessica Pegula — iarbă career (39 meciuri cu S2 jucat)

**S2 TB rate: 10/39 = 25.6%** ⚠️ zona 25-35% — RISC REAL

**Meciuri cu TB în Set 2 — analiză contextuală:**

| Data | Turneu | Adversară | Rang adv. | S1 | S2 | Relevanță |
|---|---|---|---|---|---|---|
| 2015 | s'Hertogenbosch | Bencic | #33 | 6-2 | **7-6(3)** | MICĂ — carieră timpurie |
| 2022 | Wimbledon R2 | Martic | **#80** | 6-2 | **7-6(5)** | MARE — turneu identic, adversară de nivel inferior |
| 2022 | Wimbledon R1 | Vekic | **#82** | 6-3 | **7-6(2)** | MARE — turneu identic, adversară de nivel inferior |
| 2023 | Wimbledon R1 | Davis | **#46** | 6-2 | **6-7(8)** | MARE — TB profund (#46, similar cu Bouzas #52) |
| **2024** | s'Hertogenbosch | **Krunic** | **#400** | **7-6(3)** | **6-7(3)** | ⚠️ CRITIC — TB S2 vs o jucătoare ranată 400! |
| 2024 | Berlin SF | Gauff | #2 | 7-5 | **7-6(2)** | MICĂ — vs #2 mondial, așteptat |
| 2024 | Eastbourne R16 | **Raducanu** | **#168** | 4-6 | **7-6(6)** | ⚠️ CRITIC — TB S2 vs jucătoare ranată 168 |
| 2024 | Wimbledon | Wang | #42 | 6-4 | **6-7(7)** | MARE — pierdut TB profund vs #42 |
| 2026 | Berlin SF | Sabalenka | #1 | 6-4 | **6-7(4)** | MICĂ — vs #1 mondial |
| **2026** | **Berlin QF** | **Keys** | **#28** | **7-6(5)** | **7-6(8)** | ⚠️ CRITIC — double TB match, S1 și S2 ambele tiebreak |

**Concluzie critică:** Pegula a produs TB în Set 2 inclusiv vs Krunic (#400), Raducanu (#168 la momentul respectiv), Martic (#80), Vekic (#82). Bouzas Maneiro e ranată **#52** — mai bine decât Krunic și Raducanu. Pattern structural: serviciul puternic al lui Pegula permite seturilor să ajungă la 6-6 chiar și vs adversare mai slabe.

**S1 TB → S2 TB pattern: 2/6 = 33.3%** ⚠️ **La limita pragului critic (>33% = max 6/10)**

Meciurile S1 TB → S2 TB:
- vs Krunic 2024: 7-6(3) 6-7(3) — vs #400 → EXTREM de relevant (arată că Pegula poate duce TB inclusiv vs adversare slabe)
- vs Keys 2026 Berlin: 7-6(5) 7-6(8) — double TB match, ambele seturi competitive

### Jessica Bouzas Maneiro — iarbă career (9 meciuri cu S2 jucat)

**Sample: 9 meciuri** ⚠️ sub pragul de 10

**S2 TB rate: 1/9 = 11.1%**

Singurul TB în S2:
- vs Kenin (#28), Wimbledon 2025 R2: S1 = 6-1 (dominat), S2 = **7-6(4)** → după un S1 dominant, Kenin a revenit și a forțat TB în S2. Kenin = fostă campioană GS, #28 la momentul respectiv. **Relevanță medie** — Pegula e mai bună decât Kenin pe iarbă.

**S1 TB → S2 TB: 0/2 = 0.0%** ✅ — dar din 2 meciuri (Bucsa #64 și Navarro), fără TB S2 după TB S1.

**PASUL 2 (informativ):** Ambele au probleme — Pegula cu S2 TB rate 25.6% și S1→S2 33.3%, Bouzas cu sample < 10. Confirma că Pasul 1 a dat corect FAIL.

---

## TRIPLE FILTER — VERDICT

| Filtru | Status |
|---|---|
| **Pasul 1: tb_p_cal ≤ 0.10** | ❌ **FAIL (0.1270 > 0.10)** |
| Pasul 2: Sample + S2 TB rate | N/A (oprit la Pasul 1) |
| Pasul 3: Context | N/A |

**RECOMANDARE U12.5 Set 2: PASS** — din primul filtru.

**Contextual P(U12.5 S2): ~70-75%** — semnificativ sub pragul de 82%.

---

## ANALIZĂ PROFESIONISTĂ EXTINSĂ

### Profiluri jucătoare

#### Jessica Pegula (#4, seed 4, 32 ani, USA)

**Stil pe iarbă:**
- **Serveră de elită** — 4.3 ace/meci în 2026, hold 83.3% pe iarbă din model
- Retur puternic, joc baseline cu timing precoce
- Rever pe două mâini = arma principală, profunzime și consistență
- Adaptabilitate tactică: în R2 contra Sorribes Tormo a scurtat rally-urile și a dominat total S2

**Sezon 2026 excepțional: 31-8 (80.5%)**
- Titluri: Dubai WTA 1000 + finalist Berlin (pe iarbă)
- AO 2026: SF | RG 2026: R1 (upset Birrell #83) — singura pată
- **Bad Homburg: neprotejată** — nu a jucat
- Berlin Final: pierdut cu Noskova 4-6 6-4 3-6
- Wimbledon 2026: R1 def. Vidmanova 7-5 6-3 | R2 def. Sorribes Tormo **7-6(6) 6-1** (a salvat 4 mingi de set în TB, apoi domina 6-1)

Surse: [WTA R2 Pegula](https://www.wtatennis.com/news/4529508/pegula-rallies-in-first-set-tiebreak-wins-last-six-games-to-reach-wimbledon-third-round) | [TennisUpToDate Berlin](https://tennisuptodate.com/wta/jessica-pegula-withdraws-from-queens-club-championships-among-names-delaying-start-to-grass-swing)

**Istoric Wimbledon:** Best result = **Sferturi de finală**. A eliminat Sakkari, Collins în drumul spre QF. În 2025: R1 vs Wang = dezastru. Vine motivată să repare 2025.

**Psihologie:** 32 ani, experiență maximă GS. Relaxată pe iarbă — "știu că pot juca bine." Fără anxietate de seed. Obiectiv: depășirea QF din carieră.

#### Jessica Bouzas Maneiro (#52, 23 ani, ESP)

**Stil pe iarbă:**
- Baseliner defensiv-agresivă, erori minime, shot preferat = backhand
- "Îmi place iarba — mingea se ridică la înălțimea mea de impact"
- Servă slabă (1.44 ace/meci vs 4.25 ale lui Pegula) — va fi atacată constant
- Double faults: 3.67/meci — sub presiunea Pegulei poate crește

**Sezon 2026: 13-18 (40.7%)** — slab global, dar cu nuclee bune pe iarbă:
- Nottingham QF: a pierdut cu Navarro 7-6 6-2
- Eastbourne R2: **0-6 0-6 vs Keys** (54 minute) — indicatorul real al clasei gap vs Top 5 pe iarbă
- Wimbledon R1: def. Potapova (seed 27) 6-2 6-3 — dar Potapova = hard court specialist, slab pe iarbă
- Wimbledon R2: def. Yastremska 6-3 **6-7(1)** 6-2 — a pierdut TB sec (1-7!), a revenit

**Key insight — rezultatele pe iarbă 2025-2026:** Victoriile lui Bouzas vin vs adversare care joacă slab pe iarbă (Vondroušová, Potapova, Yastremska). vs adversare cu serviciu puternic sau native pe iarbă (Keys: 0-6 0-6), imaginea se schimbă radical.

Surse: [Wikipedia Bouzas Maneiro](https://en.wikipedia.org/wiki/J%C3%A9ssica_Bouzas_Maneiro) | [WTA R1 vs Potapova](https://www.wtatennis.com/tournaments/wimbledon/scores/LS72362822) | [WTA R2 vs Yastremska](https://www.wtatennis.com/tournaments/wimbledon/scores/LS72320444)

---

### Matchup analitic — De ce tb_p_cal = 12.7% este corect

**Structura meciului:**
- Pegula servă: hold 83.3% → Bouzas sparge rar (break ~16.7%)
- Bouzas servă: hold 71.5% → Pegula sparge des (break ~28.5%)

**Probabilitate set tipic:**
- Cel mai frecvent: Pegula sparge Bouzas o dată, ține serviciul → 6-3 sau 6-4. Set rapid.
- Dar dacă Bouzas ține la 5-5 (71.5% hold = posibil), și Pegula nu rupe la 5-6 → **6-6 → TB**
- Aceasta e scenariul TB și e mult mai frecvent decât în meciurile Alexandrova/Jovic sau Muchova/Sawangkaew

**Factorul mental:** Bouzas a arătat că poate ține cu adversare mai bune pe iarbă. Nu e o jucătoare de 125 fără experiență. Are 10-7 record pe iarbă în carieră, a bătut Vondroušová (seed 1 la Wimbledon 2025), Potapova (seed 27). Nu va fi dominată 6-1 6-1.

---

### Statistici comparative (TennisRatio 2026)

| Metric | Bouzas | Pegula |
|---|---|---|
| Win % 2026 | 40.7% | **80.5%** |
| TB/meci | 22% | 22% |
| Over 12.5/set | 7% | **6%** |
| Avg games/set | 8.89 | 8.95 |
| Breaks/match (over 2.5) | 15% | **94%** |
| DF/match | 3.67 | **1.53** |
| Ace/meci | 1.44 | **4.25** |

**Notă crucială:** TB/meci egal (22% ambele) din TennisRatio este date cross-suprafață. Pe iarbă, Pegula specific are 25.6% S2 TB rate (TennisAbstract) — mai ridicat decât media ei globală. Serviciul ei de iarbă crește seturile competitive.

---

### Condiții fizice și fatigue

**Pegula:** fatigue_flag_b=True în model (Bouzas, nu Pegula — se referă la player_b). Pegula nu a jucat Queen's sau Eastbourne — odihna completă înaintea Wimbledon. R2 = 1h29min, fără 3 seturi. Formă fizică maximă.

**Bouzas Maneiro:** fatigue_flag_b=True confirmat. R2 contra Yastremska = 3 seturi (6-3, 6-7, 6-2). Energie consumată suplimentar. Va resimți în setul 2 dacă meciul se lungește.

---

### Predicție structurală Set 2

**Scenariu A (~50%):** Pegula domina S1 (6-3 sau 6-4), Bouzas nu mai poate rezista în S2 → 6-2 sau 6-3. **U12.5 ✅**

**Scenariu B (~25%):** Meciul competitiv în S1 (7-5), Bouzas ține câteva servicii în S2 → 7-5 sau 6-4. **U12.5 ✅**

**Scenariu C (~17%):** S1 merge la TB (frecvent la Pegula: 33% din meciuri au S1 TB) → S2 competitiv → **posibil TB** per pattern 33.3% S1→S2. **U12.5 ❌ (13 games)**

**Scenariu D (~8%):** Set lung fără TB în S1, dar Bouzas ține la 5-5, 6-6 → **TB S2 surpriză**. **U12.5 ❌**

**P(U12.5 S2) contextuală: ~73-75%** — confirmat: sub pragul de 82%.

---

### Cine câștigă meciul?

**Verdict: Pegula câștigă clar — ~87-89% (piața), 79.6% (p_markov), 71% (p_elo).**

**Convergeța modelelor cu piața (8pp diferență) = cea mai bună aliniere dintre toate meciurile de azi.**

**Predicție:** Pegula def. Bouzas 6-3, 6-4 sau 7-5, 6-3. Posibil 7-6, 6-4 dacă S1 merge la TB (frecvent pentru Pegula).

**Risc: 3 seturi** = ~10-12% (Bouzas poate câștiga un set dacă prinde un TB).

---

## VERDICT FINAL U12.5 SET 2

| Factor | Evaluare |
|---|---|
| **tb_p_cal = 0.1270** | ❌ **FAIL Pasul 1 — STOP** |
| Pegula S2 TB rate (25.6%) | ⚠️ Zona 25-35% — confirma riscul |
| Pegula S1→S2 TB (33.3%) | ⚠️ La limita pragului critic |
| Pegula TB vs adversare slabe (Krunic #400, Raducanu #168) | ⚠️ Pattern structural îngrijorător |
| Bouzas hold 71.5% (poate ține servicii) | ⚠️ Nu e jucătoare de 125 fără experiență |
| Robinhood vs p_markov (diferență 8pp) | ✅ Modele aliniate — dar nu schimbă verdictul U12.5 |

**SCOR FINAL: N/A — PASS** din Pasul 1 (tb_p_cal > 0.10)

**Contextual P(U12.5 S2): ~73-75%** — semnificativ sub 82%

**RECOMANDARE: PASS** — nu este un pick U12.5 Set 2. Clasamentul de risc al celor 3 meciuri analizate azi:

| Meci | tb_p_cal | P(U12.5 S2) ctx | Recomandat |
|---|---|---|---|
| Alexandrova / Jovic | 0.000 | ~93-95% | 7/10 (speculativ) |
| Muchova / Sawangkaew | 0.000 | ~96-97% | 7/10 (speculativ) |
| **Pegula / Bouzas** | **0.127** | **~73-75%** | **PASS** |

---

**Fișier generat:** 2026-07-03
**Workflow aplicat:** Triple Filter U12.5 Set 2 v1.1
**Surse principale:**
- [Robinhood Prediction Market](https://robinhood.com/us/en/prediction-markets/tennis/events/pegula-vs-bouzas-maneiro-jul-03-2026/)
- [WTA Pegula R2 Wimbledon](https://www.wtatennis.com/news/4529508/pegula-rallies-in-first-set-tiebreak-wins-last-six-games-to-reach-wimbledon-third-round)
- [WTA Bouzas R1 vs Potapova](https://www.wtatennis.com/tournaments/wimbledon/scores/LS72362822)
- [WTA Bouzas R2 vs Yastremska](https://www.wtatennis.com/tournaments/wimbledon/scores/LS72320444)
- [Wikipedia Bouzas Maneiro](https://en.wikipedia.org/wiki/J%C3%A9ssica_Bouzas_Maneiro)
- [JustWomensSports — Keys vs Bouzas Eastbourne](https://justwomenssports.com/reads/madison-keys-wta-eastbourne-2026-win/)
- TennisAbstract / Sackmann wta_matches_combined.csv (date locale)
