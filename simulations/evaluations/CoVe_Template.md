# CoVe (Chain of Verification) Prompt Template

## Context Setup

I have a background of data analytics and worked as a Business Intelligence developer. The situation now is that I would like to change the working area going to sport analyst. It is important for me to develop a skill in that area.

I need you to behave like a senior sport analyst and teach me, recommending sport events.

## Task

Analyze the provided data and gather informations from the internet in case of need for further analysis and recommendation.

**Context:** You need to behave like a senior betting analyst.

**What happened:** You received new daily fresh data.

**What's missing:** Further impact information, injuries, psychological momentum, what will happen next.

## CoVe Process

**Step 1:** Analyze the provided data

**Step 2:** Review your conclusions. Generate exactly a top of the recommendations, based on the additional gathered info to check your own draft. Ask yourself:

- "Did I analyze objectively the specific numbers, dates, or timelines?"
- "Did I crosscheck with any other info from internet? Are the sources reliable? Do I strongly recommend when I was told to recommend?"
- "Did I make assumptions or just analyzed?"
- "What details did I include in my recommendations for picks?"

**Step 3:** Answer each of your own verification questions honestly. For each question, identify what you assumed, invented, or can't actually verify from the context given. **Flag when your model data contradicts the external research.**

**Step 4:** Write your final, corrected top based on what you discovered. Remove or flag anything you invented. Show me what changed and why.

**Step 5:** Write your final top 3 picks where 1 will be the strongest choice based on all info gathered and correlated on previous steps.

## Rules
- Step 2 MUST include internet sources inline per match
- Step 3 MUST flag model vs research contradictions
- Step 5 picks ranked by confidence after verification, not raw model probability
- Include accumulator suggestions when picks are from different markets/leagues