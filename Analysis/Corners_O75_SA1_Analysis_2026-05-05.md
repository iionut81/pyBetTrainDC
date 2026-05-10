# CoVe Analysis — Corners Al Khaleej vs Al Hilal
**Date:** 2026-05-05 | **Liga:** SA1 — Round 28 | **Ora:** 20:00
**Piețe:** Over 7.5 Total Corners @ 1.25 + Over 4.5 Al Hilal Corners @ 1.31
**Template:** CoVe Corners Over (adapted) + footystats data user-provided

---

## STEP 0 — DATE VERIFICATE (footystats, user-provided)

| Metric | Al Khaleej (HOME) | Al Hilal (AWAY) |
|--------|-------------------|-----------------|
| Match Corners AVG | 9.93 acasă | 10.43 deplasare |
| Over 7.5 rate | **71%** acasă | **93%** deplasare |
| Corners Earned/match | 5.07 acasă | 6.93 deplasare |
| Corners conceded/match | 4.86 acasă | 3.50 deplasare |
| Model λ (NB) | — | **10.63** total |

---

## STEP 1 — EXPECTED TOTAL

**Al Hilal corners în meci:**
(earned away 6.93 + conceded by Al Khaleej at home 4.86) / 2 = **5.90**

**Al Khaleej corners acasă:**
(earned home 5.07 + conceded by Al Hilal away 3.50) / 2 = **4.29**

**Total așteptat: ~10.20** — concordant cu model λ=10.63 ✅

---

## STEP 2 — CONTEXT RESEARCH

### Standings & Motivație
| Echipă | Poziție | Puncte | Context |
|--------|---------|--------|---------|
| Al Khaleej | 11th | 37 pts | **SAFE** — fără presiune, 2 victorii consecutive sub Poyet (3-1, 2-0) |
| Al Hilal | **2nd** | 74 pts | **5 pts în urma Al Nassr** — MUST WIN pentru titlu |

### Injuries
| Echipă | Jucător | Status |
|--------|---------|--------|
| Al Hilal | Koulibaly | OUT (picior) — defender, nu afectează corners generare |
| Al Hilal | Al Yami | OUT (genunchi) — defender |
| Al Hilal | Bouabre | DOUBTFUL (hamstring) |
| Al Khaleej | — | Fără accidentați |

**Atacul Al Hilal intact:** Benzema ✅, Malcom ✅ (revenit), Al-Dawsari ✅, Milinkovic-Savic ✅, Neves ✅

### Lineup Al Hilal (confirmat)
`Bono — Al-Harbi, Akcicek, Tambakti, Hernandez — Milinkovic-Savic, Kanno, Neves — Al-Dawsari, Malcom — Benzema`

4-3-3: Malcom (stânga) + Al-Dawsari (dreapta) + Hernandez/Al-Harbi = fullback-uri offensive. Cross-heavy structural.

### H2H
- Al Hilal câștigă 15/17 meciuri directe
- 12/17 victorii cu minimum 2 goluri diferență
- 3/4 ultimele meciuri: Over 4.5 goluri

---

## STEP 3 — TABEL AJUSTĂRI (Over 7.5 Total)

| Factor | Constatare | pp |
|--------|------------|----|
| C2 — Tactical Al Hilal | Malcom + Al-Dawsari (wingeri) + Hernandez/Al-Harbi (fullback-uri offensive) = cross-heavy | +3pp |
| C4-B — Al Hilal must-win | 5 pts în urma Al Nassr, Round 28 → atac constant | +2pp |
| C4-B — Al Khaleej form | Poyet 2 victorii, King (18G) + Fortounis (10G+11A) → nu parchează | +1pp |
| H2H dominanță | 15/17 victorii Al Hilal → controlează jocul | +1pp |
| Blowout risk | 12/17 cu 2+ goluri diferență → risc scădere intensitate | −2pp |
| SA1 ligă | Mai puțin predictibilă decât E0/I1 | −1pp |
| **TOTAL** | | **+4pp** |

---

## STEP 4 — P(Over 7.5) CONVERGENȚĂ

| Sursă | Valoare |
|-------|---------|
| Al Khaleej home historical | 71% |
| Al Hilal away historical | **93%** |
| Media simplă | 82% |
| Model λ=10.63 (Poisson) | **~82.5%** |
| **p_research (82% + 4pp)** | **~86%** |

---

## STEP 5 — OVER 4.5 AL HILAL CORNERS

| Metric | Valoare |
|--------|---------|
| Al Hilal Over 4.5 Earned overall | 77% |
| Al Hilal Over 4.5 Earned away | **79%** |
| Al Hilal earned away avg | 6.93/meci |
| Expected Al Hilal corners azi | 5.90 |
| P(≥5 \| λ=6.93) Poisson | ~82% |

**Ajustări Over 4.5 Al Hilal:**
- Malcom revenit → winger stânga principal → +2pp
- Must-win → presiune constantă → +1pp
- Al Khaleej concedă 4.86 → nu parchează → +1pp
- **p_research: ~85%**

---

## STEP 6 — EV CALCULATION

| Piață | p_research | Fair odds | Offered | EV |
|-------|-----------|-----------|---------|-----|
| Over 7.5 Total | **86%** | 1.163 | **1.25** | **+7.5%** |
| Over 4.5 Al Hilal | **85%** | 1.176 | **1.31** | **+11.4%** |

---

## FINAL PICKS

### ✅ PICK 1 — Over 7.5 Total Corners @ 1.25 | Score: 8/10
- p_research: ~86% | EV: +7.5%
- Key: Al Hilal must-win (titlu) cu atac complet (Malcom + Benzema + Al-Dawsari) + Al Khaleej sub Poyet atacă. Expected 10.20 cornere.
- How I lose: Al Hilal marchează 3-0 la pauză, coboară ritmul, total scade sub 8.

### ✅ PICK 2 — Over 4.5 Al Hilal Corners @ 1.31 | Score: 8.5/10
- p_research: ~85% | EV: +11.4%
- Key: Al Hilal generează 6.93 cornere/meci deplasare. Malcom revenit. Must-win = presiune constantă pe flanc.
- How I lose: Al Hilal joacă central (Benzema + mid) fără să folosească flancurile → 3-4 cornere. Improbabil dat de stilul lor specific.

⚠️ **Cele 2 piețe sunt parțial corelate — nu combina în acumulator.**

---

*Analysis: 2026-05-05 | Data: footystats.org (user-provided) + model NB λ=10.63*

Sources:
- [Dailysports preview](https://dailysports.net/predictions/al-khaleej-vs-al-hilal-prediction-h2h-and-probable-lineups-05052026/)
- [Sports Mole](https://www.sportsmole.co.uk/football/al-khaleej/preview/al-khaleej-vs-al-hilal-prediction-team-news-lineups_596919.html)
- [The National — SA1 context](https://www.thenationalnews.com/sport/football/2026/05/05/al-hilal-hopes-revived-as-momentum-builds-for-ronaldo-and-al-nassr-showdown-saudi-pro-league-talking-points/)
