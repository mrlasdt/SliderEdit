# Image-stream steering: result

**Question.** Text-stream training-free steering suppressed *attribute* edits (curly
hair, hat) the fixed STLoRA adapter could not, but FAILED on **glasses** (a localized
object) and **beach** (a background/spatial swap). Hypothesis: those edits live in the
**image stream** (`output[1]` of the FLUX double blocks), not the instruction's text
tokens — so steer the image stream instead.

**What was built.** A projection-clamp / additive diff-of-means intervention hooked on
the image stream, applied to **all** image latent tokens, slider α ∈ [0,1] (suppress).
Calibrated present(instruction) vs absent("keep the image the same"), woman_7 face.

**Result: it does NOT cleanly control either edit.** Every configuration strong enough
to affect the edit also degrades or collapses the image:

| variant | beach | glasses |
|---|---|---|
| img clamp gain=1.0 | **black-noise collapse** (CLIP supp +0.100, artifact) | brown murk, glasses persist (+0.010) |
| img clamp gain=0.3 | inert, no effect (−0.002) | inert (+0.002) |
| img additive s=1.0 | **white-noise collapse** (+0.087, artifact) | α=0.25 removes glasses but washes/ID-shifts, then pink-noise collapse (+0.036) |

The CLIP "suppression" wins are artifacts of image collapse (noise scores as the
"plain/indoor" absent class), confirmed by eye in `imgstream2_beach.png` /
`imgstream2_glasses.png`.

**Why.** A single global diff-of-means direction applied to every image token either
does nothing (low gain) or flattens spatial structure (high gain / additive). It cannot
surgically remove a localized object or swap a background, because that one direction
also carries general image content. Text-stream steering works because it touches only
~5 instruction tokens carrying *edit intent*; the image stream carries *the image*.

**Cross-check (consistent).** GSTLoRA — a *trained*, global weight-scaling slider —
DID suppress glasses on this same face (earlier generalize experiment, range 0.062),
where rank-1 activation steering (text OR image) cannot. Localized image-content edits
appear to need either spatial targeting or training.

**Room for improvement / next step.** Spatial localization: steer only the image tokens
in the edited region (eyes for glasses, background for beach), e.g. from
instruction↔image cross-attention maps, instead of all tokens globally. Bigger lift
(extract attention, build per-token spatial mask, restrict the intervention).

Artifacts: `imgstream_{glasses,beach,hat,curly}.png` (3-row: STLoRA/text/img clamp1.0),
`imgstream2_{glasses,beach}.png` (5-row incl. low-gain + additive), `*_scores.json`.

---

## Phase B: spatial localization via cross-attention saliency

**Idea.** The global image-stream edit collapses because it edits *every* token along one
direction. Restrict it to the edited region only, using the instruction->image
cross-attention as a spatial mask.

**Saliency extraction (validated).** A passive replica of FLUX's joint-attention
processor (captures image-query -> instruction-key softmax mass over heads/blocks/steps;
output unchanged). Reshaped over the 64x64 target grid it localizes correctly:
`glasses` -> eye/upper-face region (2.6% of tokens hot); `beach` -> background/periphery
(12.5% hot), not the person. See `saliency_out/saliency_maps.png`.

**Masked steering result (woman_7, suppress sweep, + a CLIP quality proxy to detect
collapse: portrait vs noise; quality<0 at alpha=1 = collapse):**

| method | glasses supp / qual@α1 | beach supp / qual@α1 |
|---|---|---|
| img GLOBAL additive | +0.036 / **−0.01 (collapse)** | +0.087 / **−0.05 (collapse)** |
| img MASKED additive | +0.017 / +0.03 (intact)     | +0.111 / ~0.00 (borderline) |
| img MASKED clamp    | +0.000 / +0.08 (inert)      | +0.002 / +0.07 (inert) |
| text-stream clamp   | −0.001 (inert)              | +0.000 (inert) |
| STLoRA fixed        | +0.001 (inert)              | −0.001 (inert) |

**What masking fixed, and what it didn't.** Localization *solved the collapse*: quality
drop shrank (glasses 0.088->0.043, beach 0.120->0.072) and the rest of the image is
preserved — only the masked region changes. But within that region the rank-1 edit does
not produce a clean result: masked additive paints a localized **color artifact** (red
smear over the eyes; pink wash over the background), not a glasses-free face or the
original backdrop; masked clamp (toward the global mean setpoint) is **inert**.

**The real bottleneck (mechanistic).** A training-free rank-1 diff-of-means edit can only
*shift* activations; it cannot *re-synthesize content*. Removing glasses needs plausible
eyes/skin painted in; removing the beach needs the original background regenerated. The
text stream works for attribute edits (curly, hat) precisely because it nudges the
model's *interpretation of the instruction* and the model then synthesizes clean content;
the image stream carries already-rendered content, so a fixed direction just biases color.
This is why GSTLoRA — which *trains* the weights — can suppress glasses where no
activation nudge (text or image, global or masked) can.

**Hard-threshold mask + full clamp (the last lever).** Binarize the saliency
(weight 1.0 inside the region, 0 outside) so the full clamp/additive bites only there:

| method | glasses supp / qual@α1 | beach supp / qual@α1 |
|---|---|---|
| HARD-mask clamp    | −0.000 / +0.08 (**inert**)        | +0.013 / +0.07 (**clean, weak**) |
| HARD-mask additive | +0.011 / +0.02 (red smear)        | +0.061 / +0.01 (speckle artifact) |

The hard mask removes the collapse for good (no config tanks quality now), and the
HARD-mask **clamp on beach is the single cleanest image-stream result**: a real,
non-collapsing change — but it merely *shifts* the background to a different muted outdoor
scene, it does not restore the original, and the effect is weak (+0.013). On glasses the
full clamp is still **completely inert** (forcing the eye tokens onto the global-mean
setpoint doesn't touch the rendered glasses); the additive only paints a confined red
smear. So even the most surgical setting (binary region + full strength) cannot cleanly
remove either edit.

**Conclusion.** Spatial localization is necessary (kills the collapse) but not sufficient
(can't synthesize). The ceiling of training-free image-stream control is a weak, clean
background *shift* (hard-mask clamp, beach +0.013) and inertness/artifacts everywhere
else. Clean training-free control is feasible for attribute/semantic edits (text-stream
steering, already beats the fixed adapter OOD); object/background edits realistically need
training (LoRA/ReFT on diverse data, à la GSTLoRA) or a generative inpainting step — a
rank-1 activation edit cannot re-synthesize content, masked or not.

Hard-mask artifacts: `imgmaskhard_{glasses,beach}.png` (5-row: STLoRA / soft-add /
soft-clamp / hard-clamp / hard-add), `imgmaskhard_scores.json`.

---

## Training closes the loop: trained GSTLoRA vs the training-free methods

To test the thesis ("object/background edits need *training*, not a rank-1 nudge") we ran
the EXISTING trained GSTLoRA (`example_training_gstlora_iter500.safetensors`, general-domain
HQ-Edit) through the identical attack harness (woman_7, suppress sweep alpha 0->1).

| instruction | STLoRA fixed | text-stream (free) | img-stream hard-clamp (free) | **GSTLoRA trained** |
|---|---|---|---|---|
| curly (in-dist) | +0.069 | +0.069 | — | **+0.056** |
| glasses (far-OOD) | +0.001 | −0.001 | −0.000 | **+0.061 (clean)** |
| hat (far-OOD) | +0.002 | +0.037 | — | **+0.037** |
| beach (far-OOD) | −0.001 | +0.000 | +0.013 | +0.013 |

(no quality collapse for GSTLoRA on any axis)

**Result.** The trained GSTLoRA **cleanly removes the glasses** — they fade progressively
across the slider and are gone by alpha=1, with the rest of the image intact
(`gstlora_glasses_focus.png`) — exactly the edit that EVERY training-free method (STLoRA
fixed, text-stream steer, image-stream global/masked/hard-clamp) could not touch. It also
controls curly and hat. This confirms the synthesis thesis: training reshapes how the model
*renders*, so it can synthesize a glasses-free face; a rank-1 activation edit only shifts
activations and cannot.

**The one honest hold-out: beach.** Even the trained GSTLoRA only weakly suppresses the
full background swap (+0.013, == the best training-free image-stream result). A whole-scene
regeneration is the hardest case; this iter500 checkpoint doesn't fully reach it (longer /
background-heavy training, or an inpainting approach, would be the next lever).

**Final picture of the whole attack arc:**
- Attribute/semantic edits (curly, hat): training-free **text-stream steer** already wins
  over the fixed faces-only adapter on OOD instructions.
- Localized object edits (glasses): need **training** — GSTLoRA cleanly removes them; no
  activation nudge can.
- Whole-scene/background edits (beach): hard for everything; weakly reached by GSTLoRA and
  the spatially-masked image-stream clamp alike.

Artifacts: `attack_with_gstlora.png` (4 instructions x 4 methods), `gstlora_glasses_focus.png`
(STLoRA/text/GSTLoRA on glasses), `gstlora_scores.json`.

Note: a fresh from-scratch GSTLoRA training was NOT run here -- the repo's bf16 trainer
(~24GB for the 12B transformer) does not fit this 23GB GPU; it would need a 4-bit QLoRA
rewrite and carries the host-RAM (15GB/0-swap) machine-down risk. The existing trained
checkpoint answers the question directly.

Phase-B artifacts: `saliency_out/saliency_maps.png`, `imgmask_{glasses,beach}.png`
(5-row: STLoRA/text/global-add/masked-add/masked-clamp, with supp + quality labels),
`imgmask_scores.json`.
