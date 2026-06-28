# Next attack — research: training-free clean object removal (glasses) on FLUX.1-Kontext

**Question.** Can a *smarter* training-free intervention cleanly remove a localized object
(eyeglasses) from a FLUX.1-Kontext edit — matching the trained GSTLoRA slider (+0.061, no
collapse) — where rank-1 diff-of-means activation steering was inert / artifacting? Survey
steering advances + attention editing; deliver a ranked, box-feasible experiment shortlist.

Deep-research run: 5 angles, 21 sources fetched, 103 claims, 25 adversarially verified
(21 confirmed / 4 killed). Workflow `wf_5c54407a-c50`.

## Headline conclusion

The evidence points **away** from the steering family entirely and **toward attention-level
re-synthesis**. Two converging facts:

1. **The rank-1 failure is confirmed mechanistic, and the "advanced" steering methods don't
   escape it.** SAEdit and Concept Steerers are (a) effectively **rank-1 in their own feature
   space** (`z' = z + α·d`, single leading SVD direction) and (b) applied to the **T5 text-encoder
   residual stream**, not the FLUX image/MMDiT stream. They do *abstract concept / attribute*
   control (nudity, style, smile, age), **not** localized object removal or content re-synthesis.
   Same trap as our experiments. → **deprioritize.**
2. **Every method with published training-free *object-removal* evidence works by manipulating
   attention K/V and re-synthesizing the masked region conditioned on a preserved background** —
   not by shifting activations along a fixed direction. KV-Edit (FLUX, ICCV 2025) and ConsistEdit
   (MM-DiT/FLUX, SIGGRAPH Asia 2025) are the two FLUX-native winners; Attentive Eraser proves the
   mechanism (training-free removal *with* plausible re-synthesis) but is U-Net-only.

**So "beat training, training-free" is achievable — but the answer is to stop steering and do
attention-level region re-synthesis (KV caching / differential Q-K-V / self-attn suppression).**
That is a different paradigm from our activation-steering line; adopting it is the correct lesson
from the failures, stated plainly.

## Ranked shortlist (box-feasible: 23GB GPU, 15GB RAM/0 swap, 4-bit NF4 FLUX ~10.7GB peak)

### Tier 1 — FLUX-native, training-free, highest payoff

**#1 KV-Edit** — `arXiv:2502.17363` (ICCV 2025), code `github.com/Xilluill/KV-Edit`.
Caches background key-value pairs during inversion; reconstructs **only** the masked edit region,
attention decoupled so queries carry foreground-only while K/V carry foreground+background.
Explicitly supports **removal** (needs the mask-guided inversion + reinitialization enhancements —
not optional; residual object info otherwise leaks via attention). Best background-preservation
PSNR (35.87 vs FLUX-Fill 32.53, PIE-Bench). **Why it beats rank-1:** re-synthesizes edit-region
content conditioned on preserved background instead of shifting activations. **Feasibility:** demo
targets ~24GB but offload variants exist; 4-bit FLUX leaves headroom — **watch the 15GB host RAM
during the inversion pass** (caches ~double activation memory; 0 swap = OOM risk). *Confidence: high.*

**#2 ConsistEdit** — `arXiv:2510.17803` (SIGGRAPH Asia 2025), tested on SD3 / **FLUX.1-dev** /
CogVideoX, official code. Differential attention control: distinct treatment of **Q/K (structure)**
vs **V (content/appearance)**, restricted to **vision tokens only**, edit/non-edit regions fused
**before** the attention computation (not a post-hoc clamp), across **all** layers and steps.
**Why it beats rank-1:** attention-level, high-DOF, and directly fixes three of our specific failure
modes (text-stream hurts robustness → vision-only; post-hoc soft/hard masks weak → pre-attention
fusion; one global low-rank direction insufficient → every MM-DiT layer retains rich semantics).
**Key risk:** documented failure mode — **small/abstract objects fail when the attention map lacks
clear activation; glasses are small**, so this is exactly what to test. (Note: its "Delete Object as
a listed task" claim was **refuted 0-3** — removal is mechanistically applicable but *not* a
pre-validated task; the "all layers" universality is contested 2-1.) *Confidence: high.*

### Tier 2 — cheaper diagnostics / probes

**#3 HeadRouter** — `arXiv:2411.15034`, runs on **Flux-1.0[dev]** by default, training-free.
Different attention heads are sensitive to different image semantics; routes text guidance
head-by-head (sigmoid-weighted by per-head sensitivity). **Use as a cheap probe:** identify which
heads encode "eyeglasses," then suppress/reweight them. **Risk:** published eval is *attribute*
editing, **not** removal — transfer is an untested extrapolation. *Confidence: high (method);
removal untested.*

**#4 Conceptor / affine projection upgrade** — `arXiv:2410.16314` (conceptors),
`arXiv:2402.09631` (Representation Surgery). Replace rank-1 additive with a **soft-projection**
`h' = β·C·h` (C = PSD ellipsoid capturing principal directions *and* variances) or a least-squares
**affine** `f(h)=Wh+b` (higher-rank W≠I). Beats additive diff-of-means on every LLM task tested
(e.g. 81.62% vs 32.04%). **Why try:** strictly more expressive than the failed rank-1 shift, and
**reuses our existing cached-activation infra** — cheapest experiment to run. **Major risk:** ALL
evidence is autoregressive LLMs, **zero** diffusion/FLUX/removal transfer shown; still operates at
the activation level where re-synthesis is the unsolved problem. → treat as a **confirm-or-kill
diagnostic for the whole steering family**, not a likely solution. *Confidence: medium.*

### Supporting mechanism — port only if Tier 1 stalls

**Attentive Eraser** — `arXiv:2412.12974` (AAAI 2025 Oral). Training-free removal *with* plausible
background re-synthesis via self-attention re-engineering (set foreground→foreground self-attention
to **−∞**: `S_self* = S_self − M·inf`). Beats training-based baselines (local FID 5.835 vs 15.58;
44.9% user pref vs LaMa 19.7%). **Validates our core hypothesis** (attention edits re-synthesize;
rank-1 shifts cannot). **But U-Net-only (SD1.5/2.1/SDXL)** — the −∞ trick needs non-trivial porting
to FLUX's *joint* text+image MMDiT attention, plus a binary mask and ~2× inference. *Confidence: high.*

### Deprioritize
- **SAEdit / Concept Steerers** (`2510.05081`, `2501.19066`): text-encoder residual stream +
  effectively rank-1 → same trap. (SAE *on the FLUX image stream* remains theoretically open —
  per the withdrawn-but-backstopped DiT-SAE work — but no clean-removal result exists yet.)

## SHIFT lead — RESOLVED (verified, run wf_725118ee-115)

**"SHIFT: Steering Hidden Intermediates in Flow Transformers"** (`arXiv:2604.09213`, Konovalova,
Kuznetsov, Alanov; FusionBrain Lab / HSE; 10 Apr 2026) is **REAL** (4/4 locate routes, raw curl
HTTP 200, correct ID — not confabulated). Code `github.com/ControlGenAI/SHIFT` is **announced but
currently 404s** (not yet public). Adversarial verification (4 claims × 3 skeptics):

| claim | votes | survived? |
|---|---|---|
| training-free (frozen FLUX, inference-time only) | 0/3 refute | ✅ yes |
| directions via diff-of-means `v=(1/n)Σ(a⁺−a⁻)` | 0/3 refute | ✅ yes (same family as ours) |
| does **localized-object removal with region re-synthesis** | **3/3 refute** | ❌ NO |
| uses a **different locus** (velocity/flow) that explains success | **3/3 refute** | ❌ NO |

**It does NOT moot our question.** SHIFT is **text-to-image GENERATION control, not image editing.**
Its "glasses/hat/lipstick removal" (Fig 6) is *re-generating a different person without the object* —
two separate generations, not an edit of a given image. The paper explicitly disclaims our task:
*"it is not an image-editing method, and background consistency is not preserved."* No
FLUX-Kontext / image-conditioned eval anywhere.

Mechanistically it is **the same primitive as ours** (rank-1 diff-of-means *added* to double-stream
block activations, `Y_steered = Y_t + α·v`, time-invariant vector) — narrowed to the **text-token
slice `Y_t` only** (image tokens `Y_i` deliberately untouched) plus the **CLIP pooled embedding**,
in a *generation* pipeline. "velocity" never appears. So SHIFT **corroborates our hypothesis**: a
fixed rank-1 shift can re-bias a not-yet-realized generation but cannot re-synthesize content over
an object already committed in an input image — which is exactly why our edit-time rank-1 failed and
GSTLoRA succeeded.

**The one borrowable delta (a fresh experiment, not a fix):** steer **only `Y_t`** (never the image
stream — our image-stream steer is precisely what smeared/collapsed) **+ a CLIP pooled-embedding
steer** (a lever we never tried) **+ adaptive SVM-gated strength `α=γ·η_cls`** (vs our fixed coeff).
Honest expectation: the dominant gap is *generation-vs-editing task framing*, not locus — so this is
a hypothesis test, not a likely fix, and there's no off-the-shelf code (reimplement from Eq. 4/9/11–14).

## The single sharpest first move

Open question #4 from the run is the highest-information experiment: a **conceptor (#4) vs KV-Edit
(#1) head-to-head on the identical masked-glasses case (woman_7)**. It disambiguates the root cause:
- If the **conceptor** (still activation-level, just higher-rank) works → the failure was *low rank*,
  and we stay in the steering family (cheap, reuses infra).
- If only **KV-Edit** (re-synthesis) works → the failure is *shift-can't-resynthesize*, and the
  whole steering line is the wrong tool for object removal. Commit to attention K/V methods.

Run the cheap conceptor arm first (reuses our cached activations); stand up KV-Edit in parallel.

## Caveats (from the verification pass)
1. **No source shows clean glasses removal end-to-end on FLUX.1-*Kontext* specifically** — KV-Edit
   & ConsistEdit are on FLUX.1-*dev*; Kontext's instruction/reference-image conditioning may interact
   with attention injection differently. Transfer is plausible, unverified.
2. The winners are **mask/inversion/inpainting re-synthesis** — a different paradigm from steering.
   Adopting them = partly abandoning the steering framing (the correct lesson, made explicit).
3. ConsistEdit "Delete Object" claim **refuted 0-3**; "all layers" contested 2-1 — removal not
   pre-validated; small-object (glasses) is the specific risk.
4. Conceptor/affine evidence is **LLM-only**, zero diffusion transfer — cheapest but most speculative.
5. Field is <1 yr old (sources Nov 2024–Oct 2025); FLUX attention-editing code maturity varies.
6. **GPU/RAM:** all feasibility assumes 4-bit NF4 FLUX (~10.7GB). Inversion methods (KV-Edit) roughly
   double cache memory and stress the 0-swap 15GB host RAM — monitor for OOM during inversion.

## Verified sources
- KV-Edit: arXiv:2502.17363 · github.com/Xilluill/KV-Edit
- ConsistEdit: arXiv:2510.17803
- HeadRouter: arXiv:2411.15034
- Conceptors: arXiv:2410.16314 · Representation Surgery: arXiv:2402.09631
- Attentive Eraser: arXiv:2412.12974
- SAEdit: arXiv:2510.05081 · Concept Steerers: arXiv:2501.19066 · DiT-SAE: openreview J48XM0au4u
- Lead (unverified): SHIFT arXiv:2604.09213
