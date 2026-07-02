# CoVe Analysis — U12.5 Set 2 | Wimbledon 2026 R1
## Maja Chwalinska vs Mananchaya Sawangkaew
**Data:** 2026-06-29 | **Ora:** 11:00 BST (12:00 CEST)
**Turneu:** The Championships, Wimbledon — Round 1 (R128)
**Suprafață:** Iarbă (outdoor, All England Club)
**Analist:** AI Sports Analyst | **Metodologie:** Triple Filter + Contextual TB Analysis

---

## NOTE METODOLOGICE CRITICE

**MODELUL A SKIP-AT:** "[SKIP] Sawangkaew vs Maja Chwalinska (Grass) — no stats for B=Maja Chwalinska"

Chwalinska are **1 singur meci pe iarbă** în TennisAbstract (Q1 Wimbledon 2025 — pierdut). Modelul nu poate calcula hold rates sau tb_p_cal. **Triple filter = imposibil de aplicat complet.**

Analiza se face 100% din date manuale TennisAbstract + cercetare.

---

## PASUL 1 — TRIPLE FILTER

**IMPOSIBIL DE APLICAT** — lipsesc datele pentru Chwalinska:
- tb_p_cal: **NEDISPONIBIL** (no model compute)
- p_hold Chwalinska: **NEDISPONIBIL**
- p_markov, p_elo: **NEDISPONIBIL**

**Status: ❌ SKIP automat** (model nu poate analiza)

---

## PASUL 2 — TENNISABSTRACT (iarbă cu analiză contextuală)

### Maja Chwalinska — Iarbă 2023-2026

**Sample: 1 meci** 🔴🔴 — CATASTROFAL DE MIC

| Meci | Turneu | Oponent rang | S1 TB? | S2 | **S2 TB?** |
|---|---|---|---|---|---|
| vs Serban (L) Wimbledon Q1 2025 | Grand Slam Q | **~231** | ❌ | 3-6 | ❌ NO (3-set: 6-4, 3-6, 6-2) |

**Chwalinska S2 TB pe iarbă: 0/1 = 0%** — **NESEMNIFICATIV (1 meci!)**

**CONTEXT CHWALINSKA PE IARBĂ:**
- A PIERDUT în Q1 Wimbledon 2025 vs Serban (#231 = jucătoare extrem de slabă!)
- Nu a câștigat niciodată un meci WTA de iarbă în carieră
- 2026: **DIRECT DE LA Roland Garros (clay final!) → Wimbledon** fără niciun meci de pregătire pe iarbă
- Practic: **DEBUTANTĂ pe iarbă la nivel WTA main draw**

---

### Mananchaya Sawangkaew — Iarbă 2023-2026

**Sample: 6 meciuri** ✅ (2026, toate recente)

| Meci | Turneu | Oponent rang | S1 TB? | S2 | **S2 TB?** |
|---|---|---|---|---|---|
| vs Miyazaki (W) Birmingham Q1 | WTA 125 | ~292 | ❌ | 6-3 | ❌ NO |
| vs **Ito (W) Birmingham Q2** | WTA 125 | **~234** | **7-6(3)** ✅ | 6-3 | ❌ NO |
| vs Klimovicova (W) Birmingham R32 | WTA 125 | ~156 | ❌ | 7-5 | ❌ NO |
| vs Day (W) Birmingham R16 | WTA 125 | **~145** | ❌ | 6-1 | ❌ NO |
| vs Eala (L) Birmingham QF | WTA 125 | **#37** | ❌ | 6-2 | ❌ NO (Eala won) |
| vs Urhobo (L) Ilkley R32 | WTA 125 | ~180 | **7-6(3)** ✅ | 6-0 | ❌ NO ← S1 TB pentru Urhobo, S2 decisiv |

**CORECȚIE AGENT:** Ilkley vs Urhobo — TB-ul era în S1 (7-6(3) Urhobo), S2=6-0 DECISIV. Agent greșit identificat ca "S2 TB".

**Sawangkaew S2 TB pe iarbă: 0/6 = 0%** ✅✅✅

**S1 TB → S2 pattern:**
- vs Ito: S1 TB (Sawangkaew câștigă) → S2=6-3 decisiv ✅
- vs Urhobo: S1 TB (Urhobo câștigă) → S2=6-0 decisiv pentru Urhobo ✅

Pattern: indiferent cine câștigă S1 TB, S2 e decisiv pentru Sawangkaew.

**CONTEXT SAWANGKAEW:**
- A câștigat **calificările Wimbledon** (3 runde, inclusiv victorie vs Dodin)
- Birmingham 125: a ajuns QF (4 victorii) înainte de a pierde vs Eala (#37)
- Match-sharp pe iarbă: **6 meciuri recente pe iarbă!**

---

## 1. MATCH CONTEXT

**Wimbledon 2026 R1** — #20 seed vs qualificantă. Meci extrem de interesant:

**Chwalinska:** Roland Garros finalist (clay!) → **DIRECTĂ la Wimbledon** fără niciun meci de iarbă în 2026!

**Sawangkaew:** A jucat **6 meciuri pe iarbă** (Birmingham 125 + Wimbledon qualifying). Match-sharp pe suprafață.

---

## 2. PROFILURI JUCĂTOARE

### Maja Chwalinska (Polonia)
- **Rang:** **#21 WTA** | **Seed Wimbledon: #20 (wildcard seed!)** | **Vârstă:** 24 ani (n. 11 oct 2001)
- **Stil:** 5'5", stângace, **slice + topspin + drop shots** = stil de clay! NU o jucătoare naturală de iarbă
- **Roland Garros 2026:** Finalistă (prima qualifier în finală la Roland Garros!) — a bătut 4 top-50
- **Grass career:** **1 victorie pe iarbă în toată cariera WTA** (Wimbledon Q... pierdut chiar acela!)
- **2026 grass:** ZERO meciuri înainte de azi
- **History making:** A treia cu wildcard care este seedată la un major (Hingis 2002, Schnyder 2004)
- **Sursa ranking:** De la #114 → **#21** în 10 zile (post-Roland Garros)

### Mananchaya Sawangkaew (Thailanda)
- **Rang:** **#164 WTA** | **Career high: #100** (iunie 2025) | **Vârstă:** ~23 ani
- **Stil:** Baseliner solid, returnuri active, joc asiatic bazat pe consistență
- **Grass 2026:** 6 meciuri, ajuns QF Birmingham 125 (4-1 înainte de Eala)
- **Wimbledon:** A câștigat 3 runde de calificări!
- **2026 record:** 25W-10L (bun!)
- **Hold iarbă (estimat):** Bazat pe 0/6 S2 TB = ține bine serviciu pe iarbă

---

## 3. ANALIZA STILURILOR — IMPACT PE IARBĂ

**Chwalinska pe iarbă — major concern:**

"Stângace, 5'5", slice + topspin + drop shots" = **JOC DE CLAY** clasic. Pe iarbă:
- Mingea vine joasă și rapidă → ea preferă alta viteză
- Drop shots mai puțin eficiente pe iarbă
- Slice funcționează pe iarbă, dar nu este specialista acestui stil (spre deosebire de Hsieh sau Navarro)
- **Niciodată nu a câștigat consistent pe iarbă**

**Contextul post-RG:**
- Câștigat Roland Garros final cu mental maxim pe CLAY
- Tranziție directă clay → iarbă fără meciuri de pregătire
- Boston Globe: "swings into Wimbledon as an unqualified success" → presiunea renumelui
- WTA: "I'm just a human being" = încearcă să gestioneze noua realitate

**Sawangkaew pe iarbă — match-sharp:**
- 6 meciuri de iarbă în 3 săptămâni (Birmingham + Ilkley + Wimbledon Q)
- Știe ce face pe iarbă acum
- A câștigat vs Day (#145), Klimovicova (#156), Ito (#234), Miyazaki (#292) = adversare mai slabe dar experience acumulată

---

## 4. CONDIȚIE FIZICĂ

**Chwalinska:** A jucat Roland Garros final cu ~3-4 săptămâni în urmă. Odihnită fizic, DAR:
- Fără meci de iarbă = ritm de joc 0 pe suprafață
- WTA article: "adjusts to tennis' new reality" = adaptare mentală activă

**Sawangkaew:** Has played Birmingham (QF) + Ilkley (R32) + Wimbledon Q (3 runde) = ~8 meciuri în 4 săptămâni pe iarbă. Match-sharp dar potențial ușor obosită.

**Avantaj ritm iarbă: Sawangkaew masiv.**

---

## 5. MOTIVAȚIE & PSIHOLOGIC

### Chwalinska — ⬆️ ENORM + ⚠️ SUPRASOLICITATĂ
- A trăit cel mai mare moment din carieră (RG finalist la 24 ani, din qualifier!)
- Wimbledon = primul Grand Slam de iarbă ca jucătoare seeded
- Presiunea enormă: "Miracolul de la Roland Garros, acum la Wimbledon?"
- Fără experiență iarbă → anxietate posibilă la adaptare
- Are toate motivele să lupte, dar suprafața e complet nouă la nivel WTA

### Sawangkaew — ↔️ NIMIC DE PIERDUT
- #164 vs #21 seed = favorita clară este Chwalinska
- A câștigat calificările = deja și-a depășit așteptările
- Match-sharp, relaxată
- Joacă cu curaj = periculoasă

---

## 6. ANALIZA U12.5 SET 2 — CE PUTEM SPUNE FĂRĂ MODEL

**Fără date model**, ne bazăm exclusiv pe:

| Date disponibile | Valoare | Semnal |
|---|---|---|
| **Sawangkaew S2 TB iarbă** | **0/6 = 0%** | ✅✅✅ |
| **S1 TB → S2 Sawangkaew** | 0/2 = 0% | ✅✅ |
| **Chwalinska S2 TB iarbă** | 0/1 (nesemnificativ) | — |
| Chwalinska stil pe iarbă | Clay-orientat = imprevizibil | ⚠️ |
| Match-up | Incert (clay #21 vs grass #164) | ⚠️ |

**Factori structurali pentru U12.5:**
- Sawangkaew nu tinde să ajungă la S2 TB (0/6)
- DAR: fără date hold rate pentru Chwalinska → nu știm cât de bine ține serviciu pe iarbă
- Dacă Chwalinska ține bine serviciu (s-ar putea dat stilul versatil) → set competitiv → risc TB
- Dacă Chwalinska ține slab (mai probabil pe iarbă pentru o clay specialist) → seturi decisive → U12.5

**PROBLEMA FUNDAMENTALĂ:** Nu avem tb_p_cal. Fără el, nu putem da o probabilitate U12.5 fiabilă.

---

## 7. VERDICT U12.5 SET 2

**PASS** ❌ — din motive metodologice clare

**Motivul:** Modelul a SKIP-at pentru că Chwalinska nu are date pe iarbă. Fără hold rates pentru Chwalinska, nu putem calcula tb_p_cal. Triple filter nu poate fi aplicat. Orice predicție U12.5 ar fi speculație pură.

Chiar dacă Sawangkaew are 0/6 S2 TB (excelent), incertitudinea totală privind comportamentul Chwalinskiej pe iarbă face pick-ul nerecomandabil.

---

## 8. PREDICȚIE CÂȘTIGĂTOARE

**Cel mai dificil de prezis din toată lista de azi.**

| Factor | Pro Chwalinska | Pro Sawangkaew |
|---|---|---|
| Ranking | #21 vs #164 ✅ | — |
| Seeding | #20 ✅ | — |
| Grass experience 2026 | 0 meciuri ❌ | **6 meciuri** ✅ |
| Recent form | RG finalist (clay) | Wimbledon qualifier |
| Surface adaptation | Clay → grass 0 meciuri | 4+ săptămâni pe iarbă |
| Mental | Suprasolicitată? | Relaxată |
| Stil pe iarbă | Sub-optimal (clay game) | Adaptat recent |

**Estimare Chwalinska câștigă: ~55-60%** (superioritate calitativă generală, dar grass = minus)

**Sawangkaew câștigă: ~40-45%** (upset real — match-sharp, grass-ready)

Aceasta este una din cele mai incerte predicții ale zilei. Sawangkaew NU este un clear underdog pe suprafața asta.

---

## 9. VERDICT FINAL

| Market | Status | Scor | Decizie |
|---|---|---|---|
| U12.5 Set 2 | **PASS** | N/A | ❌ Fără date model Chwalinska |
| Match winner | Incert | — | Chwalinska ~57% |

**Acest meci nu este analizabil prin triple filter** — este singurul din lista de azi fără date model pentru una din jucătoare. Orice pick bazat exclusiv pe Sawangkaew's 0/6 S2 TB ar ignora incertitudinea completă privind Chwalinska pe iarbă.

---

## RANKING FINAL PICKS WIMBLEDON DAY 1 — ACTUALIZAT

| # | Market | Meci | Score | Decizie |
|---|---|---|---|---|
| **1** | **U12.5 Set 2** | **Pegula vs Vidmanova** | **8/10** | **✅✅ PRINCIPAL** |
| **2** | **U12.5 Set 2** | **Yastremska vs Ito** | **7/10** | **✅** |
| **3** | **O7.5 Set 1** | **Jacquemot vs Osaka** | **7/10** | **✅** |
| **4** | **O7.5 Set 1** | **Linette vs Andreeva** | **7/10** | **✅** |
| 5 | U12.5 Set 2 | Ann Li vs Sonmez | 6/10 | ✅ rezerve |
| — | U12.5 Set 2 | Chwalinska vs Sawangkaew | **PASS** | ❌ fără date |

---

## SURSE

- [TennisAbstract JS — Maja Chwalinska](https://www.tennisabstract.com/jsmatches/MajaChwalinska.js)
- [TennisAbstract JS — Mananchaya Sawangkaew](https://www.tennisabstract.com/jsmatches/MananchayaSawangkaew.js)
- [Wikipedia — Maja Chwalińska](https://en.wikipedia.org/wiki/Maja_Chwali%C5%84ska)
- [Boston Globe — Chwalinska at Wimbledon](https://www.bostonglobe.com/2026/06/28/sports/french-open-runner-up-maja-chwalinska-wimbledon/)
- [WTA — Chwalinska: 'I'm just a human being'](https://www.wtatennis.com/news/4527718/im-just-a-human-being-maja-chwalinska-adjusts-to-tennis-new-reality)
- [Just Women's Sports — Chwalinska Wimbledon WC](https://justwomenssports.com/reads/maja-chwalinska-wimbledon-2026-wildcard-entry-wta-rankings/)
- [WTA Official — Sawangkaew Profile](https://www.wtatennis.com/players/326929/mananchaya-sawangkaew)
- [Bangkok Post — Sawangkaew books Wimbledon](https://www.bangkokpost.com/sports/3277270/mananchaya-books-spot-in-wimbledon)
- Model Markov+WElo: SKIP — no stats for Maja Chwalinska (run 2026-06-29)
