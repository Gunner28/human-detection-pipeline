# Roadmap: audience-adaptive screens for branches and stores

**Decided 2026-08-13.** Target sectors: **telecommunications retail and retail
banking.**

This document states what is being built, what it will and will not claim,
and the order of work. It is a plan, not a description of finished software —
Stage 0 is complete and everything after it is not.

---

## The goal, in one sentence

**Show the right service to the people actually in front of the screen, and
prove they watched.**

Not "increase sales by X%". That claim needs a client's revenue data attributed
to screen exposure, which is not obtainable here, and asserting it without the
data would be the one genuinely damaging thing this project could do. What is
demonstrable: the screen adapted to its audience, and here is the attention
data showing whether the adaptation mattered.

---

## Why telecom and banking, together

Both are **service businesses operating from physical branches where customers
wait**. That shared property is what makes one system serve both:

- Customers are stationary for meaningful periods — queueing, waiting for an
  advisor, waiting for a handset to be configured
- The products need explanation. A mortgage, a business account, a tariff
  change and a device upgrade are all considered purchases, not impulse buys
- Staff are occupied with the customer in front of them, so waiting customers
  receive no attention at all
- Both sectors already run screens in-branch, showing static loops that ignore
  the room entirely

**One important difference from mall or transit advertising:** in a branch,
dwell time is mostly *involuntary*. Someone standing for four minutes is
queueing, not expressing interest. This inverts the usual reading of dwell, and
the decision policy has to respect it — a captive audience justifies longer,
more explanatory content, but it must not be mistaken for engagement. Attention
is measured by whether people *orient toward the screen*, not by how long they
are stuck near it.

---

## The hard constraint

**No individual is identified. Ever.**

Faces are biometric data under UK and EU GDPR — special category, requiring
explicit consent from each person. In a walk-in branch that consent cannot be
obtained. Any design depending on recognising individuals does not ship in
these sectors, and both are heavily regulated besides.

What the system knows is limited to, and sufficient as:

| Signal | Where it comes from |
|---|---|
| How many people are in view | Detection |
| How long each has been present | Track lifetime |
| Approaching, passing, or stopped | Trajectory direction |
| Together, or separate individuals | Trajectory proximity over time |
| Someone returned after leaving | Appearance embedding (Stage 2) |
| Time of day, venue busyness | Clock, plus aggregate counts |

All of it comes from the **geometry of movement**, not from anyone's face. No
demographic inference, no gender or age estimation, no identity, nothing
retained between visits. Frames are processed and discarded; only counts and
durations persist.

This is a stronger position than a compromise. It is more defensible legally,
easier to sell to a compliance officer in either sector, and the signals it
yields are the ones that actually drive the decision.

---

## Stages

### Stage 0 — Detection, tracking, measurement · **complete**

Detection and tracking benchmarked on MOT17: **HOTA 41.99, IDF1 50.32, MOTA
43.87**, recall 52.8%, precision 86.8%. Scored by TrackEval, the implementation
behind the public leaderboard. See [FINDINGS.md](FINDINGS.md) #10–14.

Everything below depends on identity being stable across frames, which is why
this came first and why it was measured rather than assumed.

### Stage 1 — Audience attributes · ~1 day

Turn each track into: dwell seconds, direction of travel, stopped or passing,
group membership, distance band from the screen.

**Trajectory geometry, not a model.** Dwell is track lifetime. Group membership
is proximity sustained over time. Direction is the sign of displacement. There
is nothing to train and nothing to label, and a classifier for something
derivable from coordinates is a model you must train, validate and defend in
exchange for nothing.

*Unlocks:* every downstream stage.

### Stage 2 — Re-identification · ~1 day

A pretrained appearance network (OSNet via `torchreid`, weights download
automatically) gives each track an embedding — a numeric description of
clothing and build, not a face. Two purposes: identity survives someone walking
behind a pillar, and "this person came back" becomes detectable within a
session.

*Measurable:* IDF1 and AssA on MOT17. It works or it doesn't, and the benchmark
says which.

*Note:* embeddings are session-scoped and discarded. They are not stored, not
matched across visits, and not a person's identity.

### Stage 3 — Audience state · ~half a day

One object, refreshed each second, that the ad layer consumes:

```
{ present: 3, stopped: 2, mean_dwell_s: 12.4, group: true,
  approaching: 1, returning: 0, venue_busy: false }
```

Clean seam between the vision work and the advertising logic. Either side can
be replaced without touching the other, and the ad layer can be tested against
synthetic states with no video at all.

### Stage 4 — Decision policy · ~half a day

Audience state in, content category out. **Explicit rules, not a learned
model.**

That is a deliberate engineering choice, not a shortcut. A learned policy needs
outcome data — which content led to which sale — and that does not exist here.
An explicit policy is auditable, a compliance officer can read it and agree
with it, and in regulated sectors that matters more than sophistication.

Sketch, telecom:

| Audience state | Content |
|---|---|
| Nobody present | Brand loop, low brightness |
| One person passing | Single message, high contrast, no detail |
| One person stopped near handsets | Device comparison for that display |
| Group stopped, 10 s+ | Family and multi-line plans |
| Long dwell, queue forming | Upgrade eligibility, accessories, explainer |
| Returning visitor | Different creative — they have seen the first |

Sketch, banking:

| Audience state | Content |
|---|---|
| Nobody present | Brand loop |
| One person passing | Single message: rates, or opening hours |
| One person stopped, self-service zone | Digital banking, app features |
| Group stopped, 10 s+ | Mortgages, family savings, joint accounts |
| Long dwell, queue forming | Longer explainer — captive but not engaged |
| Business hours, weekday morning | Business banking, merchant services |

Content is categories, not creative. The client supplies the assets.

### Stage 5 — Attention measurement · ~half a day

The stage that makes this a product rather than a demo.

**Attention-seconds per content item**: with each piece of content, how many
people were present, how many were oriented toward the screen, and for how
long. This is the metric digital out-of-home genuinely trades on, it is
honestly derivable from what is already tracked, and it closes the loop —
policy changes can be evaluated against it.

Reported as aggregates per content item per hour. No individual-level records.

### Stage 6 — Scene context · optional

CLIP zero-shot classification for venue context: busy or quiet, queue formed or
not, daytime or evening. Pretrained; new categories are added by writing a
sentence rather than retraining. Worth doing only if Stage 4 turns out to need
context that counting cannot supply.

---

## What will not be claimed

Stated up front so no later work drifts into it:

- **No sales or conversion lift.** No outcome data exists. Attention is the
  honest ceiling on what can be measured here.
- **No demographic targeting.** No age, gender or ethnicity inference. Not
  attempted, not approximated.
- **No cross-visit tracking.** Nothing persists past a session.
- **No individual identification.** See the constraint above.

Each of these is a question an interviewer or a compliance officer will ask.
Having the answer ready, with the reason, is worth more than having the feature.

---

## Immediate next steps

Two cheap, single-variable experiments outstanding on Stage 0, both using code
already in the repository:

1. **Camera-motion compensation** (`--compensate`). MOT17-13 contributes 495 of
   1,079 identity switches from 10% of the data, and every benchmark run so far
   had compensation off. Largest known loss, smallest known fix.
2. **A larger detector** (`--backend yolo:yolov8s.pt`). Recall is 52.8%; half
   of all annotated pedestrians are never found, concentrated in small distant
   figures.

Then Stage 1, which needs no new models and unlocks everything after it.

---

## Open questions

- **Screen placement per sector.** Queue-facing and entrance-facing produce
  different audience states and want different policies. Worth deciding before
  Stage 4.
- **Camera position.** Above the screen looking outward is the natural mount
  and gives the cleanest approach/orientation signal, but it is the worst angle
  for detection — heavy foreshortening. Needs testing on real footage.
- **Content inventory.** Stage 4 outputs categories; someone has to map
  categories to actual assets, and that is a client conversation.
