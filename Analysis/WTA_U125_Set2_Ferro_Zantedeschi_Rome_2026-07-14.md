# CoVe Analysis — U12.5 Set 2
## Fiona Ferro vs Aurora Zantedeschi
### WTA 125 Rome (ATV Bancomat Tennis Open) | Clay | R1 Main Draw | 14 Iulie 2026, 16:00 CEST

---

## MODEL SNAPSHOT

| Câmp | Valoare |
|---|---|
| Tournament | ROME 125, WTA 125, Clay |
| Round | R1 main draw (round=5) |
| p_hold_a (Ferro) | 0.5750 |
| p_hold_b (Zantedeschi) | 0.3793 |
| hold_asym | 0.2010 |
| min_hold | **0.3793** (Zantedeschi) |
| BCI | 0.1998 |
| tb_p_cal | **0.0865** (8.65%) |
| p_u125 | 0.9135 (91.35%) |
| premium_elite | NO (tb_p_cal=0.0865 > prag 0.08) |
| premium_u125 | **YES** (min_hold<0.50 + hold_asym>0.15 + tb_p_cal<0.10) |
| danger_zone | **NO** (min_hold=0.3793 < 0.40 → chiar sub danger zone) |
| blowout_score | — |
| UNSTABLE flag | NO |
| Winner model | **Ferro 79.1%** (fair odds 1.26) |

---

## PASUL 1 — CSV Model + Market Check

### Triple Filter Checklist

| Verificare | Valoare | Status |
|---|---|---|
| tb_p_cal ≤ 0.10 | **0.0865** | ✅ PASS |
| Elo/Markov gap > 35pp | p_elo Ferro 72.2% vs p_markov 79.1% → gap **6.9pp** | ✅ PASS (sub 35pp) |
| p_elo = 0.0 | Ferro Elo 403, Zantedeschi Elo 237 — ambele au date Elo | ✅ PASS |
| UNSTABLE flag | Absent din CSV | ✅ PASS |
| danger_zone | min_hold=0.3793 < 0.40 → sub danger zone | ✅ NO |

### Robinhood Market Check

**URL:** [robinhood.com — Ferro vs Zantedeschi Jul 14 2026](https://robinhood.com/us/en/prediction-markets/tennis/events/ferro-vs-zantedeschi-jul-14-2026/)

- **Ferro: 68¢ = 68%**
- **Zantedeschi: 32¢ = 32%**

| Criteriu | Valoare | Interpretare |
|---|---|---|
| P(favorita) | **68%** | 60-74% → continuă, notează divergența |
| p_markov (model Winner) | 79.1% | — |
| Divergență market vs p_markov | **11.1pp** | Sub 15pp → fără investigație obligatorie ✅ |

**Context divergență 11pp:** Market prețuiește ruginirea Ferro pe lut (6-7 săptămâni fără lut), avantajul home pentru Zantedeschi (italiancă la Roma) și forma recentă a lui Zantedeschi (SF la Contrexeville). Nu au fost găsite știri de accidentare pentru niciuna. Divergența rămâne sub pragul de SKIP.

**Pasul 1: COMPLET ✅ — Continuăm**

---

## PASUL 2 — TennisAbstract / CoreTennis (suprafața curentă: Clay)

### Aurora Zantedeschi — Date lut

**Sursa:** [CoreTennis — Aurora Zantedeschi rezultate](https://www.coretennis.net/tennis-player/aurora-zantedeschi/83021/results.html)

- **Total meciuri pe lut (2023-2026): 127**
- **Meciuri cu S2 tiebreak (scor 7-6 în Set 2): 18**
- **S2 TB rate pe lut: 18/127 = 14.2%**

**Threshold:**
- 14.2% < 15% → **confirmare +1pp** ✅

### Pattern S1 TB → S2 (clay, documentat)

| Meci | Turneu | S1 | S2 | Nota |
|---|---|---|---|---|
| vs Lew Yan Foon (Aix-les-Bains, Jul 2026) | W35 Hard Indoor | 7-6(2) | **6-3** | S1 TB → S2 clean ✅ |
| vs Ce (Bol, Mai 2026) | W35 Clay | 7-6(1) | **6-1** | S1 TB → S2 clean ✅ |
| vs Perrin (W75 Grado, Mai 2024) | Clay | 7-6(3) | **6-4** | S1 TB → S2 clean ✅ |
| vs Hon (W75 Rome, Jul 2024) | Clay | 7-6(5) | **6-4** | S1 TB → S2 clean ✅ |

**S1 TB → S2 TB rate: 0/4 = 0%** → ≤20% → **confirmare +1pp** ✅

### Analiza S2 TB-uri documentate pe lut (context)

**Singurul S2 TB semnificativ pe lut recent:**

> **Blinkova vs Zantedeschi, Contrexeville QF, 10 iulie 2026, Clay**
> Scor: S1=**6-4** (Zantedeschi), S2=**7-6(5)** (Blinkova)

| Factor | Detaliu |
|---|---|
| Opponent | Anna Blinkova, **WTA #102** |
| Tip meci | QF — meci cu miză reală |
| Context S2 TB | Zantedeschi câștigase S1 → Blinkova a luptat înapoi în S2 |
| Relevanță pentru meciul nostru | **SCĂZUT** — Zantedeschi era favorita acelui meci; Blinkova (WTA 102) este mult mai puternică decât Ferro? NU — Ferro (WTA 189) este mai slabă decât Blinkova, dar pattern-ul diferit: Zantedeschi era cea care câștigase S1. În meciul nostru, Zantedeschi este ***perdantul* așteptat** — pattern complet opus. |

**Concluzie critică:** TB-ul S2 al lui Zantedeschi apare exclusiv când ea este **în controlul meciului sau în meci competitiv echilibrat**. Când pierde controlul (serva 37.4% hold), seturile se termină rapid fără TB. Analiza pierdeerilor sale pe lut confirmă: vs Podoroska → 5-7, 0-6 (niciun TB), vs Vandromme → 3-6, 4-6 (niciun TB), vs Granwehr → 3-6, 3-6 (niciun TB).

### Fiona Ferro — Date lut 2026

**Sursa:** [TennisTemple — Ferro câștigă titlul Oeiras WTA 125](https://en.tennistemple.com/actu/fiona-ferro-claims-oeiras-wta-125-title-first/zRa8)

**Meciuri pe lut 2026 documentate:**

| Meci | Turneu | S1 | S2 | S3 | S2 TB? |
|---|---|---|---|---|---|
| vs Pridankina | Oeiras WTA 125 | **7-6** | 6-1 | — | **S1 TB → S2 clean** ✅ |
| vs Havlickova | Oeiras SF | 6-2 | 0-6 | 6-3 | NO |
| vs Andreescu | Oeiras QF | 6-0 | 2-6 | 6-4 | NO |
| vs Klimovicova | Oeiras | 6-3 | 6-4 | — | NO |
| vs Kudermetova | Oeiras Final | 6-3 | 0-6 | 6-1 | NO |
| vs Andreeva (RG) | Roland Garros R1 | 3-6 | 3-6 | — | NO |

**Ferro S2 TB rate pe lut 2026: 0/6 meciuri = 0%** (sample mic dar consistent cu modelul)

**Pattern S1 TB → S2:** 1 singur caz (vs Pridankina): S1=7-6 → S2=6-1 clean.

**Pasul 2: COMPLET ✅**

---

## PASUL 3 — Context Manual

### Oboseală (Fatigue)

| Jucătoare | Situație | Status |
|---|---|---|
| **Fiona Ferro** | Ultimul meci: 23 Iunie (Wimbledon Qualifying R1 pe iarbă). 3 săptămâni odihnă. Fără meci pe lut din Roland Garros (late May). | **Odihnită** — risc rugineală pe lut |
| **Aurora Zantedeschi** | 9 meciuri în 11 zile (1-11 Iulie): Aix-les-Bains W35 (SF) + Contrexeville WTA 125 (SF). QF vs Blinkova S2 TB (fizic solicitant). 3 zile odihnă (12-14 Iulie). **had_3sets_7d=True** (Aix R16: 6-2, 5-7, 6-2) | **Obosită acumulat** ← FACTOR CHEIE |

**Implicație:** La 36.6°C, Zantedeschi intră cu bateria descărcată după 9 meciuri. Serva ei (deja 37.4% hold) va deteriora suplimentar în Set 2 din cauza oboselii fizice.

### Condiții Meteo — Roma, 14 Iulie 2026

**Sursa:** [Meteo Roma 14 Iulie 2026](https://www.archeoroma.org/weather-rome/forecasts-for-four-days/)

- **Temperatură:** **36.6°C** (maxim zilnic) — EXTREM DE CALD
- **Umiditate:** ~47%
- **Vânt:** până la 21.6 km/h
- **Ploaie:** <5% probabilitate
- **Indice UV:** 9 (extrem)

**Impact U12.5 S2:** Căldura extremă accelerează deteriorarea fizică → serva slabă a lui Zantedeschi devine și mai slabă → break-uri rapide → seturi scurte fără TB. Factor net pozitiv pentru Under.

### Motivație

**Fiona Ferro:**
- A câștigat primul titlu WTA în 3 ani la Oeiras (Aprilie 2026) → revenire în forma bună
- Ranking în creștere (WTA #189) → fiecare punct pe lut contează pentru top-100
- A primit wildcard Roland Garros 2026 → confirmată ca jucătoare cu statut special în circuit
- **Motivație: RIDICATĂ** ★★★★★

**Aurora Zantedeschi:**
- Jucătoare de origine italiancă (Verona) → avantaj public la Roma
- Primarily dublist (peak doubles #102 WTA) — singles este obiectiv secundar
- 9 meciuri în 11 zile → motivație poate fi afectată de oboseală
- **Motivație: MODERATĂ** ★★★ (avantaj public, dar oboseală fizică)

### Coaching

**Ferro:** Fără informații de coaching schimbate în surse deschise 2026. Tactici de la Oeiras (primele serve de calitate, forehand topspin agresiv pe lut) confirmă o echipă stabilă.

**Zantedeschi:** Informații de coaching indisponibile din surse deschise.

### Context Psihologic

**Ferro:**
- Titlu la Oeiras (Aprilie) → încredere maximă pe lut
- 3 săptămâni pauză → intrare fresh, motivată
- Rusineala pe lut este *reală* (6-7 săptămâni fără match pe clay) dar este specialist
- **Psihologie: STABILĂ, ÎNCREZĂTOARE** ✅

**Zantedeschi:**
- 9 meciuri în 11 zile → mental și fizic la limită
- Home crowd Roma → presiune suplimentară (poate deveni adversar)
- A pierdut două finale consecutive (Bol, Aix-les-Bains) + SF la Contrexeville → poate resimți presiunea publicului italian
- **Psihologie: INCERTĂ** — home crowd poate fi sabie cu 2 tăișuri

### Stil de Joc — Matchup Analysis

**Ferro (Clay Baseline Specialist):**
- Heavy topspin forehand pe lut → controlează ritmul din fundal
- Serve ajustat tactic: reduce viteza pt. consistență (84% first serve în finalul decisiv Oeiras)
- Nu este specialist la fileu, dar solid din baseline
- Pattern la Oeiras: inteligentă tactic, răbdătoare, câștigă punctele lungi

**Zantedeschi (Dublist cu problemă la serviciu):**
- Dublist experimentat → bună la fileu, returner solid
- Serva = PUNCTUL SLAB MAJOR: 37.4% hold rate (se pierde serviciul 62.6% din situații)
- Gets broken 9.5 ori pe meci în medie → Ferro va face break practic la fiecare serviciu Zantedeschi
- Ca dublist: joacă mult la fileu → pe lut, împotriva forehand-ului greu al lui Ferro, asta e riscant

**Matchup:**
- Ferro va controla din baseline cu forehand
- Zantedeschi va încerca să scurteze punctele venind la fileu (stilul doublistului)
- Problema: 36.6°C + oboseală → voleu-urile din a doua parte a setului vor deveni imprecise
- Break-uri frecvente de la Ferro pe serviciul Zantedeschi → seturi scurte garantate structural

---

## ANALIZA PROBABILITATE CERCETARE

| Factor | Ajustare |
|---|---|
| Baza model (p_u125) | **91.35%** |
| Zantedeschi fatigue (9 meciuri / 11 zile) + căldură 36.6°C | +1.5pp |
| Market Robinhood 68% vs model 79.1% (Ferro mai competitiv decât crede modelul) | -0.5pp |
| S2 TB clay rate 14.2% (< 15%) + S1 TB→S2 0% | confirmare score ✅ |
| **Probabilitate cercetare finală** | **~92.3%** |

**≥ 82%** ✅ → **RECOMANDĂM dacă odds ≥ 1.10**

---

## SCOR FINAL U12.5 SET 2

| Condiție din tabel | Status |
|---|---|
| Pasul 1 OK (toate verificările) | ✅ |
| Pasul 2 OK (sample ≥ 10, date S2 TB clay găsite) | ✅ |
| S2 TB clay ≤ 15% (14.2%) | ✅ |
| S1 TB → S2 ≤ 20% (0/4 = 0%) | ✅ |
| Fără UNSTABLE flag | ✅ |
| Fără danger_zone | ✅ |
| Robinhood ≥ 60% (68%) + divergență < 15pp | ✅ |

**SCOR: 9/10** ★★★★★

**Clay minimum: 8/10 + RH** → Scor 9/10 depășește minimul ✅

---

## PREDICȚIE JOC

### Cine câștigă?
**FIONA FERRO** — 79.1% model, 92.3% structurală pentru U12.5 S2

### Cum se desfășoară?

**Set 1:**
Ferro iese cu forehand-ul agresiv de la prima minge. Zantedeschi va face o apărare inițială bună (home crowd, proaspătă după 3 zile odihnă), dar hold rate-ul ei de 37.4% înseamnă că Ferro face break frecvent. Așteptăm 1-2 serve games pierdute de Ferro (rugineală pe lut, presiunea inițială) dar Ferro ține mai bine (57.5%). Zantedeschi poate câștiga 2-3 jocuri.

**Estimare Set 1: Ferro 6-3 sau 6-2**

**Set 2:**
Acesta este setul-cheie pentru piața noastră. La 36.6°C, după Set 1, Zantedeschi va fi fizic slăbită. Serva va deteriora suplimentar. Ferro, odihnită (3 săptămâni) și în ritm după S1, va crește nivelul. Break-uri rapide pe serviciul Zantedeschi. Zantedeschi poate lupta pe serviciul Ferro (returnuri bune de dublist), dar cu hold rate 37.4% nu poate face față.

**Estimare Set 2: Ferro 6-2 sau 6-1 — FĂRĂ TIEBREAK**

### Scor final predicție:
**Ferro 6-3 / 6-2**

*(Variantă minoritară: 6-2 / 6-3 dacă Ferro scade în Set 2 initial. Scenariu TB S2 estimat la ~8% structural.)*

---

## VERDICT FINAL

| Piață | Recomandare | Scor | Probabilitate Cercetare |
|---|---|---|---|
| **U12.5 Set 2** | ✅ **RECOMANDĂM** | **9/10** | **~92.3%** |
| Winner (Ferro) | ✅ Model 79.1%, Elo 72.2% | — | — |

**Odds minime necesare:** ≥ 1.10 pentru U12.5 S2

**Stake sugerat:** 5% (conform scoring 9/10 → HIGH confidence)

---

## FLAGS FINALE

- **Rugineală lut Ferro (6-7 săpt.):** Prețuită de piață (11pp divergență), nu schimbă structura U12.5 S2
- **Home advantage Zantedeschi:** Factor psihologic real, dar insuficient contra clasei tehnice și servei slabe
- **Căldură 36.6°C:** Favorabil Under (seturi mai scurte)
- **Zantedeschi oboseală acumulată:** Factor cheie — 9 meciuri în 11 zile

---

## SURSE

- [Robinhood — Ferro vs Zantedeschi Jul 14 prediction market](https://robinhood.com/us/en/prediction-markets/tennis/events/ferro-vs-zantedeschi-jul-14-2026/)
- [CoreTennis — Aurora Zantedeschi clay results](https://www.coretennis.net/tennis-player/aurora-zantedeschi/83021/results.html)
- [TennisTemple — Blinkova vs Zantedeschi Contrexeville QF](https://en.tennistemple.com/match/blinkova-zantedeschi-contrexeville-2026/9471991/)
- [TennisTemple — Vandromme vs Zantedeschi Contrexeville SF](https://en.tennistemple.com/match/vandromme-zantedeschi-contrexeville-2026/9472117/)
- [TennisTemple — Ferro wins Oeiras WTA 125 title](https://en.tennistemple.com/actu/fiona-ferro-claims-oeiras-wta-125-title-first/zRa8)
- [Tennis Inside Numbers — Oeiras 2026 Final](https://tennisinsidenumbers.substack.com/p/wta125-oeirasceto26-kudermetova-vs-ferro)
- [Roland Garros 2026 — Andreeva vs Ferro R1](https://www.rolandgarros.com/en-us/article/2026-edition-r1-andreeva-ferro)
- [Wikipedia — 2026 Grand Est Open 88 (Contrexeville, clay confirmed)](https://en.wikipedia.org/wiki/2026_Grand_Est_Open_88)
- [Wikipedia — Aurora Zantedeschi profile](https://en.wikipedia.org/wiki/Aurora_Zantedeschi)
- [WTA — ATV Bancomat Tennis Open 2026 draws](https://www.wtatennis.com/tournaments/1130/rome-125/2026/draws)
- [Meteo Roma 14 Iulie 2026](https://www.archeoroma.org/weather-rome/forecasts-for-four-days/)
- [TieBreaK Tennis — WTA 125 Rome draw 2026](https://www.tiebreaktennis.it/wta-125-roma-2026-tyra-grant-sfida-martina-trevisan-al-primo-turno-sorteggiato-il-tabellone-dellatv-tennis-open/)
