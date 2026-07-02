# CoVe Analysis — U12.5 Set 2 | Wimbledon 2026 R1
## Elena-Gabriela Ruse vs Caty McNally
**Data:** 2026-06-30 | **Ora:** 17:00 BST
**Turneu:** The Championships, Wimbledon — Round 1 (R128)
**Suprafață:** Iarbă | **Date:** fresh fetch TennisAbstract + web search

---

## PASUL 1 — TRIPLE FILTER

| Parametru | Valoare | Status |
|---|---|---|
| tb_p_cal | **8.64%** | ✅ (la prag) |
| gap | **6.3pp** | ✅ |
| UNSTABLE | Nu | ✅ |
| **hold_asym** | **2.70pp** ← CEL MAI MIC AZI | 🔴🔴 |
| blowout | **2** | ⚠️ |
| competitive_set | True | ⚠️ |

**PASUL 1: ✅ TRECUT** — dar hold_asym = 2.70pp = cel mai mic din toată lista de azi

---

## PASUL 2 — TENNISABSTRACT GRASS

### Elena-Gabriela Ruse — Iarbă 2023-2026

**Sample: 11 meciuri** ✅✅

**S2 TBs:**

| Meci | Turneu | Oponent rang | S1 | S2 | S2 TB? |
|---|---|---|---|---|---|
| vs **Mertens (L) s'Herto F 2025** | WTA 500 F | **#25** | 6-3 (Ruse W) | **7-6(4)** | **✅ YES** |
| vs **Korpatsch (W) s'Herto R32 2026** | WTA 500 | **#79** | 6-4 (Ruse W) | **7-6(5)** | **✅ YES** |

**Ruse S2 TB: 2/11 = 18.2%** ⚠️

**Contextul critic:**
- Mertens #25 (WTA 500 final): nivel ridicat, normal că e tight
- **Korpatsch #79**: Ruse câștigă S1 6-4 → S2 **7-6(5) TB!** — Korpatsch #79 ≈ **McNally #50** nivel. **Cel mai relevant precedent!**

---

### Caty McNally — Iarbă 2023-2026

**Sample: 14 meciuri** ✅✅

**S2 TBs:**

| Meci | Turneu | Oponent rang | S1 | S2 | S2 TB? |
|---|---|---|---|---|---|
| vs **Pliskova (L) Nottingham R16 2026** | Int. WTA | **#87** | 6-4 (McNally W) | **7-6(3)** (Pliskova W) | **✅ YES** |
| vs **Tjen (W) Eastbourne R32 2026** | Int. WTA | **#41** | 7-5 (McNally W) | **6-7(5)** (Tjen W) | **✅ YES** |

**McNally S2 TB: 2/14 = 14.3%** ✅ (acceptabil, sub 15%)

**Contextul TBs McNally:**
- vs Pliskova #87: Pliskova câștigă S2 TB → McNally PIERDE matchul
- vs Tjen #41: McNally PIERDE S2 TB dar câștigă în S3 → meciul în 3 seturi

---

## VERDICT U12.5 SET 2

| Factor | Valoare | Semnal |
|---|---|---|
| **hold_asym** | **2.70pp** ← MINIM ABSOLUT | 🔴🔴🔴 |
| tb_p_cal | **8.64%** | ⚠️ |
| Ruse S2 TB | **18.2%** | ⚠️ |
| **Korpatsch #79 = McNally #50** | TB relevant! | 🔴 |
| McNally S2 TB | 14.3% ✅ | ✅ |
| blowout | 2 | ⚠️ |

**Scor: 5/10 → PASS** — cel mai slab structural din toată lista de azi.

hold_asym 2.70pp = practic zero diferență structurală între cele două. Ambele hold la ~70%. Seturile vor fi competitive natural → TB risc real. Ruse vs Korpatsch #79 confirmat cu S2 TB la același nivel de adversar.

---

## PROFIL

**Ruse** (#71, română, 28 ani): s'Herto finalist 2025, Bad Homburg SF 2026. Formă bună pe iarbă.

**McNally** (#50, americancă, 24 ani): Newport 125 winner 2025 (beat Maria!), career high #49 în iunie 2026. Joc agresiv, returnuri bune.

---

## RANKING FINAL COMPLET WIMBLEDON DAY 2

| # | Meci | Score | Decizie |
|---|---|---|---|
| **1** | **Kudermetova vs Samsonova** | **8/10** | **✅✅ PICK PRINCIPAL** |
| **2** | **Shnaider vs Lys** | **7/10** | **✅ PICK** |
| 3 | Selekhmeteva vs Kraus | neanalizat | hold_asym 1.16pp |
| — | Ruse vs McNally | **5/10 PASS** | hold_asym 2.70pp |
| — | Eala vs Zarazua | **6/10 PASS** | hold_asym 3.34pp |
| — | Gibson vs Bouzkova | **6/10 PASS** | hold_asym 2.75pp |
| — | Tomljanovic vs Bolkvadze | **PASS** | TB Bolkvadze ieri |
| — | Begu vs Swan | **PASS** | UNSTABLE + Swan 50% TB |
| — | [SKIP] Erjavec/Jeanjean | — | gap 44.8pp |
| — | [SKIP] Serena/Joint | — | gap 44.4pp + 1401 zile |
| — | [SKIP] Boisson/Rybakina | — | p_elo=0 |

---

## SURSE
- [TennisAbstract — Elena-Gabriela Ruse](https://www.tennisabstract.com/jsmatches/ElenaGabrielaRuse.js)
- [TennisAbstract — Caty McNally](https://www.tennisabstract.com/jsmatches/CatyMcnally.js)
- [Wikipedia — Elena-Gabriela Ruse](https://en.wikipedia.org/wiki/Elena-Gabriela_Ruse)
- [Wikipedia — Caty McNally](https://en.wikipedia.org/wiki/Caty_McNally)
- [WTA Official — Ruse](https://www.wtatennis.com/players/320408/elena-gabriela-ruse)
- [WTA Official — McNally](https://www.wtatennis.com/players/325725/caty-mcnally)
- Model: `simulations/WTA/evaluations/1.5_WTA_Under12_5.csv` (run 2026-06-30)
