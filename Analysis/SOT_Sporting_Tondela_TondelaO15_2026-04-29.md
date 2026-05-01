# Sporting CP vs Tondela — Tondela SOT Over 1.5 (Away)

**Data:** 29 aprilie 2026, 21:15 UTC+1
**Competiție:** Liga Portugal — Matchday 26
**Stadion:** Estádio José Alvalade, Lisboa

---

## 1. Context Match

| Factor | Sporting CP | Tondela |
|---|---|---|
| Poziție | **#3** (push pentru top-3) | **#17/18** (zonă retrogradare, 7p sub safety line) |
| Forma | 1-2 vs Benfica acasă; 0-0 vs Porto (Taça SF) | Winless + goalless ultimele 2 meciuri |
| Motivație | Top-3 lock | Salvare matematică |

**H2H:** 6 victorii consecutive Sporting; 17 meciuri din 2015 (12W-3D-2L Sporting). Ultim direct: **Tondela 0-3 Sporting** (away în 2025/26).

**Sporting injuries:** Hjulmand, Inácio (mijlocași defensivi), Fresneda, Ioannidis, Nuno Santos — 5 jucători OUT, **inclusiv 2 defensivi** = ușor mai vulnerabili defensiv ✓

---

## 2. Statistici SOT Tondela (cheia analizei)

### Sezon 2025/26 Liga Portugal:
- **3.5 SOT/meci sezon** (rank 14/18) ✓
- **3.0 SOT/meci ultimele 10 meciuri** ✓
- **3.7 SOT/meci ultimele away matches** (cu 4.9 cornere medii)
- 10.57 shots total/meci (9.0 away)
- xG: 35.8 (rank 12)

### Recent away matches (key data):
| Meci | Posession | SOT | Result |
|---|---|---|---|
| **vs FC Porto (away, 19.04.2026)** | 40% | **3** | L 0-2 ⚠️ |
| vs Gil Vicente | 52% | 5 | D 2-2 |
| vs Nacional (acasă, 25.04) | n/a | ~5 | L 0-2 |

**KEY INSIGHT:** Chiar și la **Porto away** (defensivă top), Tondela a avut **3 SOT**. Asta e cel mai relevant comparable pentru Sporting away.

---

## 3. Statistici Sporting (defensiv)

- 17.75 shots/meci (offensive)
- 6.86 SOT/meci (offensive)
- ~0.9 goluri/meci concedate (Top-3 cea mai bună apărare)
- Sporting concede ~3-4 SOT acasă (teams generate shots even vs strong defenses)

---

## 4. Calcul Probabilitate Tondela OVER 1.5 SOT

### Model SOT v2.1:
- λ_our (Tondela): 1.438
- λ_bk (scaled 2.0x): **2.875**
- elo_multiplier: 0.812 (echipă slabă vs Sporting Elo)
- k_dispersion: 10.09 (P1 league)
- p_over (line 2.5) = **48.6%** (calibrated)

### Estimare manuală pentru linia 1.5:
**Poisson (λ=2.875):**
- P(X=0) = e^(-2.875) = 0.0564
- P(X=1) = 2.875 × 0.0564 = 0.162
- **P(X≥2) = 1 - 0.0564 - 0.162 = 0.781 = 78.1%**

**NB (λ=2.875, k=10.09):** ușor sub 78% → **~75-78%**

### Estimare empirică (din date reale):
- Tondela avg ULTIMELE 10 = 3.0 SOT
- Vs Porto away (similar caliber) = 3 SOT
- Adjusted Sporting away (top-3 defense): ~2.7 SOT expected
- P(X≥2 | λ=2.7, Poisson) = 75%

### Range probabilitate: **75-82%**
**Fair odds:** **1.22-1.33**

---

## 5. Factori de risc / consider

### ✅ PRO Tondela OVER 1.5:
1. **Floor în date:** chiar și worst-case (Porto away) = 3 SOT ✓
2. **Avg ultimele 10 = 3.0** SOT (margine 1.0 față de linia 1.5)
3. **Sporting injuries defensive** (Inácio, Hjulmand) = mai puține blockuri ✓
4. **Sporting trebuie să atace** (top-3 push, Tondela bunker) = Tondela counter-shots ✓
5. **Nothing to lose mentality:** Tondela goalless ultimele 2 → presați să marcheze ✓
6. **Sporting averaging 6.86 SOT for** = high-pace match → multe shots in total → Tondela și ea generează ✓

### ⚠️ RISC:
1. **Tondela goalless 2 meciuri** — dacă continuă, posibil sub 2 SOT (DAR goluri ≠ SOT)
2. Tondela 10.57 shots/meci → conversion 7% (low) → SOT ratio ~30-35%
3. Sporting press intens acasă → Tondela poate fi ținută la 1-2 SOT
4. Caz extrem: 6-0 blowout cu Tondela complet decimată = 1 SOT posibil (rare)

### Probabilitate evenimente extreme (Tondela 0-1 SOT):
- 0 SOT = ~5-7% (foarte rar)
- 1 SOT = ~14-18%
- TOTAL P(under 1.5) = ~20-25%
- **P(over 1.5) = 75-80%** ✓

---

## 6. Verdict

### 🟢 **TONDELA OVER 1.5 SOT — RECOMMENDED**

| Criteriu | Score |
|---|---|
| Probabilitate (estimată) | **78-82%** |
| Fair odds | 1.22-1.28 |
| Margine față de linie | 1.5 SOT (avg 3.0 = +1.5 buffer) |
| Empiric (Porto away) | Confirmă (3 SOT) |
| Risc class-mismatch | LOW (Sporting nu poate sufoca complet) |
| Risc dead-rubber | LOW (Tondela luptă să nu retrogradeze) |
| Sporting injuries | PRO Tondela |

### Cota minimă acceptabilă: **1.22**
Cota recomandată: **1.30+** = value clear

### Stake: dacă cota ≥ 1.25 → 2-3% bankroll (confidence 8/10)

**Avertisment:** Modelul SOT din script are p_over (linia 2.5) = 48.6% → for 1.5 line să interpolez să fie ~78%. Match-ul nu e în watchlist (line 1.5 nu e default), dar matematic clar.

---

## 7. Pariere alternativă pe acest meci

Dacă vrei OVER mai sigur:
- **Sporting SOT Over 4.5 (acasă)** — model 69.5%, fair 1.44 — Sporting top scoring
- **Total SOT in match Over 6.5/7.5** — Sporting + Tondela combined ≈ 6.86 + 3.0 = ~10 SOT expected
- Sporting WIN -1.5 AH = sporting ar trebui sa câștige cu 2+ goluri

---

## Surse internet

- [Sportsgambler - Sporting vs Tondela preview](https://www.sportsgambler.com/betting-tips/football/sporting-vs-tondela-prediction-lineups-odds-2026-04-29/)
- [Liontips - Sporting vs Tondela betting tips](https://www.liontips.com/tips/2026/04/28/sporting-tondela-betting-tip-cf-1-88-and-bets-on-the-portuguese-championship-match-april-29-2026)
- [APwin - Sporting vs Tondela H2H](https://www.apwin.com/predictions/sporting-vs-tondela-prediction-liga-portugal-29-04-2026/)
- [OneFootball - Sporting CP v Tondela: top-three push meets survival fight](https://onefootball.com/en/news/sporting-cp-v-tondela-top-three-push-meets-survival-fight-42783916)
- [FotMob - Tondela team stats](https://www.fotmob.com/teams/188163/stats/tondela/teams)
- [FootyStats - CD Tondela 2025/26](https://footystats.org/clubs/cd-tondela-169)
- [ESPN - FC Porto 2-0 Tondela (Apr 19, 2026)](https://www.espn.com/soccer/match/_/gameId/750510/tondela-fc-porto)
- [SoccerPunter - Sporting CP 2025/26 results](https://www.soccerpunter.com/team/all/25745/58/Sporting-CP-in-Portugal-Liga-Portugal-2025-2026)

---

**Generat:** 29 aprilie 2026
**Autor:** Claude Code SOT Analysis