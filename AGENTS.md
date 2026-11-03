# Pred: persistent project instructions

Read `CLAUDE.md` for the existing project architecture. For the football markets
GOAL_BY_70 and CORNERS_1H_UNDER_7_5, also read
`Prompts/Football_Goal70_Corners1H_Master.md` before researching or ranking.
The explicit user instructions in the conversation take priority over older prompts.

- Research autonomously across all leagues with usable history. Record coverage,
  exclusions and missing data before choosing finalists. Do not stop after finding
  the first acceptable match, and do not call a partial search a global daily top.
- Scope clarified 2026-11-02: inventory every men's first-division fixture in
  the top 20 UEFA national associations (state the ranking year), plus Romania,
  Serbia and the already tracked European second divisions. Calculate every
  eligible fixture before ranking. Separate complete fixture/model coverage
  from the deeper contextual checks of finalists; record missing data explicitly.
  Today's full comparison: Analysis/Goal70_2026-11-02/EUROPE_ALL_MATCHES.md.
- For these two markets, do not ask for or use bookmaker odds. Report the model's
  mathematical fair odds separately from any available market price. A desired
  price above 1.10 is a preference, not something inferred from the fair odds.
- Daily protocol agreed 2026-11-01, intended from 2026-11-02: at most ONE
  individual selection total across GOAL_BY_70 and CORNERS_1H_UNDER_7_5;
  NO BET is allowed. Do not combine legs just to reach a desired price.
  High hit rate alone does not imply profit: break-even probability is 1/odds,
  before costs. Do not claim value or an available price without evidence.
  Save prospective analyses and outcomes; do not present an unvalidated 90%
  estimate as an established rate or a fixed monthly loss allowance.
  This saved protocol does not create a scheduled run or authorize staking.
- Preserve the exact <=70 goal definition and <=7 first-half corner definition.
  Half-time scores do not supply goal counts up to minute 70. Full-time corners
  do not supply first-half corners. Missing observations are never zeroes.
- Keep raw probabilities, historical frequencies, stress scenarios and contextual
  judgments separate. Do not add arbitrary percentage adjustments or tune a model
  to reach 85-90%. Preserve sources and any conflicting evidence.
- Verify current Romanian date/time, kickoff and unstarted status for live picks.
  Historical tests use only information from before each prediction date.
- Save methods, data coverage and evidence in the project so a new session can
  resume. Do not assume a different chat or computer has this folder or its files.
- Persisted discussion style: respond in Romanian; research and choose autonomously,
  explain the counts, why a match ranks where it does, the contrary evidence and
  alternatives without asking for odds or permission to start. Daily history lives
  in Analysis/Selection_History/README.md. Keep all analyzed candidates, analyst
  recommendations and the user's own choices distinct. The user explicitly chose
  TWO events on 2026-11-02: Mallorca-Cadiz GOAL_BY_70 and Sassuolo-Fiorentina
  CORNERS_1H_UNDER_7_5. Record that dated choice without treating it as a bet placed
  or silently replacing the general one-selection preference for future days.
- Leave existing production models, calibration, user edits and frozen
  `selection_engine/` unchanged unless their modification is explicitly requested.
- Outcome review for 2026-11-02 is in Analysis/Review_2026-11-02/REVIEW.md:
  22/26 GOAL_BY_70 successes, four failures, one postponed fixture excluded;
  both user selections verified successful. Preserve the archived probabilities.
  Treat this as one day's evidence; report own attacking rates by venue and
  late/absent first goals before finalists, without retrospective percentage edits.
- Do not schedule jobs, send notifications, place bets, commit or push as part
  of an analysis or an offline audit.
- Explicit scheduling authorization, 2026-11-03: the user requested a daily
  08:00 Europe/Bucharest agent. Windows task `Pred_Daily_Research_0800` is now
  registered for this project, first run 2026-11-04 08:00 RO. This is the narrow
  exception to the older no-scheduling rule; daily research must not create
  duplicate jobs or change other tasks. Read `automation/daily_research_prompt.md`.
  Order: verify previous-day outcomes (prioritize explicit user choices and
  retry older missing outcomes), review successes/failures and lessons, then
  research BOTH GOAL_BY_70 and CORNERS_1H_UNDER_7_5 across the agreed scope,
  save evidence and rebuild the dashboard. User-choice badges show `-Succes`
  or `-Fail` only for verified outcomes, otherwise pending/postponed/void.
  Never turn an analyst candidate into a user choice automatically. Scheduling
  does not authorize bets, stakes, external messages, commits or model changes.
- Dashboard workflow agreed 2026-11-03: read
  `Analysis/Selection_History/DAILY_WORKFLOW.md` when resuming daily work.
  Save each day's complete candidates, evidence, analyst finalists, explicit
  user choices and separately verified outcomes. After any such update, rebuild
  the local dashboard with `node Previews/Pred_Research_Terminal/build.mjs`.
  It discovers dated journals/candidate CSVs automatically; never hardcode today's
  choices in the interface. Preserve original prediction snapshots.
  The user will explicitly name their corners choice; record it as the user's
  choice, distinct from the analyst candidate. On 2026-11-03: QPR-Burnley GOAL_BY_70
  and Derby County-Norwich City CORNERS_1H_UNDER_7_5 are both user-confirmed.
  Recording explicit choices in both markets overrides the one-selection recording
  limit for those choices; do not silently drop either or infer a combined bet.
  In addition to updates requested here, the explicitly authorized Windows task
  now starts this workflow daily at 08:00 RO; operational details and limits
  are recorded in `automation/README_daily_research.md`.
- Follow-up clarified 2026-11-03: catch up after 08:00 when the laptop is
  started/logged in, unlocked/resumed or reconnects to the network. The SAME
  Windows task uses these triggers plus a per-day guard; do not create duplicate
  research runs. A completed/partial research run is not repeated just because
  the laptop is unlocked again; a failed chat delivery retries delivery only.
  After each daily run publish the report and an absolute clickable link to
  `Previews/Pred_Research_Terminal/dist/index.html` in the user's existing chat,
  using the explicitly authorized native Codex session continuation. Do not send
  email or messages to other people. After manual analysis/selection/dashboard
  updates also include that HTML link in the final reply.
  Pin the user's explicit choices FIRST in the dashboard list for the active
  market/day; order the remaining matches by model probability. Preserve the
  original model ranks, probabilities, history and active filters.
- Browser preference, 2026-11-03: use the dedicated
  `[Deschide dashboardul in Chrome](pred-dashboard://open)` link first.
  The per-user Windows handler opens the fixed local index.html in Chrome;
  installation is recorded in Previews/Pred_Research_Terminal/chrome-launcher.json.
  Keep the absolute HTML link as fallback when custom schemes are blocked by
  the chat client. Do not change the default browser or other associations.

Offline reproduction (PowerShell; does not fetch data or run production jobs):

```powershell
powershell -NoProfile -File .\audit_goal70_corners_1h.ps1 -AsOf 2026-11-01
```
