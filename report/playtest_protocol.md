# Playtest protocol (Assignment 6)

One session per subject, about 20 minutes. The facilitator follows the same script every time.

## Setup
- Open `<live URL>/?playtest=S<n>` in a fresh browser profile (or a private window) so the subject sees the first-visit
  "How to play" screen. The purple bar at the bottom shows the subject id; every click, screen change, photo, job,
  round result, error and an FPS sample every 5 s are logged to `run/studio.db`.
- Sound on, window maximised. Have a notes sheet ready (below).

## 1. Introduction (say only this)
"This is a 3D studio game: you describe things in words, an AI turns them into 3D models, and there is a challenge mode.
Please play for a while and think aloud: tell me what you are trying to do and why." Do **not** explain the controls.

## 2. Observation (think-aloud)
Remind them to keep talking if they go quiet. Do not guide them. Only if they are completely stuck for about a minute,
give the smallest possible hint and write it down as a **facilitator hint**.

## 3. Silent notes (one line per event, with the time)
- Where did they fail, hesitate or go back?
- Which prompts or UI did they miss (forbidden words, difficulty, photo shoot, leaderboard)?
- Frame-rate drops (the HUD shows FPS) and when; visual glitches; errors.
- How long they waited for the model and what they did meanwhile.

## 4. Post-play interview
Click **Finish & answer 5 questions** in the purple bar; the subject types the answers (stored with the session):
1. What was the most confusing or frustrating part of the experience?
2. On a scale of 1–10, how intuitive were the controls?
3. Did you notice any visual glitches or lag? If so, when?
4. On a scale of 1–10, how fun was the Challenge?
5. What one change would make it better?

## After all sessions
`python scripts/playtest_summary.py S1 S2 S3 S4 S5` → `evidence/playtest/telemetry_summary.csv`, `surveys.json`,
per-subject event logs. Put the notes sheets in `evidence/playtest/S<n>.md`.
