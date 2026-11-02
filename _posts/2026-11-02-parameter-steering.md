---
layout: post
lang: en
title: "Turning a Dial on a Model's Dangerous Direction"
subtitle: "On reversibility, hotspots, and why safety does not live in a single layer"
---

I have been thinking about safety fine-tuning in terms of directions. When you fine-tune a model, you add an update to its weights. If that update makes the model less safe, then in some loose sense it points along a "dangerous direction" in parameter space. Directions have a nice property: you can scale them. So instead of asking whether a fine-tuned model is safe or unsafe, we can ask what happens as we continuously dial that direction up and down, and where in the network it actually bites.

This is neither a very precise statement nor is it meant to be fully rigorous. I just find it to be a useful way to turn "fine-tuning erodes safety" from a binary observation into something with a shape. This post is about measuring that shape on Qwen3-8B.

## A Dial on a Dangerous Direction

The starting point is a LoRA that encodes a dangerous direction (I will call it Beaver-Danger). A LoRA is a low-rank update $$BA$$ added to a linear layer's weight $$W$$. At inference time I simply multiply it by a scalar $$s$$:

$$
W(s) = W + s \cdot BA
$$

At $$s=0$$ the LoRA is off and we have the base model. At $$s=2$$ we recover the strength it was trained at. Negative $$s$$ moves the opposite way, and $$s>2$$ extrapolates past the training strength. The nice thing about this setup is that nothing needs to be retrained or saved: load the base model and the LoRA once, and change $$s$$ on the fly. It is the same trick as [Task Arithmetic](https://openreview.net/forum?id=6t0Kwf8-jrj), where the difference between two sets of weights is treated as a vector that can be scaled and negated.

The second knob is a mask. I can choose to apply the update only in certain layers or certain modules: attention only, MLP only, a single projection like `o_proj`, a single layer, or a contiguous range of layers. Every experiment below is the same dial with a different mask.

<figure>
  <img src="{{ '/assets/images/param-steering/dial_setup.svg' | relative_url }}" alt="The steering setup">
  <figcaption>The setup. A LoRA update BA is scaled by a single number s and added to the base weights W, optionally through a mask that restricts it to chosen layers or modules. The resulting model answers 550 harmful questions, and two judges score the answers. Nothing is retrained when s or the mask changes.</figcaption>
</figure>

To measure safety, Qwen3-8B answers 550 harmful questions from CatQA-English with deterministic decoding. I score the answers with two judges that work very differently. LlamaGuard-3 gives a discrete safe or unsafe verdict, from which I compute the attack success rate (ASR, lower is safer). The Beaver cost model gives a continuous score, which I report as a Safety Score (higher is safer). I use two judges so that they can check each other. The base model at $$s=0$$ has a Safety Score of 4.836 and an ASR of 0.73%.

## The Dial Is Reversible, but the Response Is Not Linear

The first experiment is the simplest one: turn on the full update everywhere and sweep $$s$$ from $$-1$$ to $$3$$.

<figure>
  <img src="{{ '/assets/images/param-steering/module_overview.png' | relative_url }}" alt="Module-level response curves">
  <figcaption>Safety response as the steering scale s changes. Blue: update applied to all modules. Orange: attention only. Green: MLP only. Left: Beaver Safety Score (higher is safer). Right: LlamaGuard ASR (lower is safer). The shaded region marks 0 ≤ s ≤ 2, the range the LoRA was trained over; larger dots mark s=2. The all-modules curve is flat for small s, falls steeply in the middle, and saturates past s≈2.2.</figcaption>
</figure>

Moving along the direction does what you would expect. At $$s=2$$, ASR goes from 0.73% to 68.55%, and extrapolating to $$s=3$$ gets it to 87.64%. What I found more interesting is the other side. At $$s=-1$$, ASR drops to 0.18% and the Safety Score rises to 5.669, slightly above the base model. So the direction is continuous and reversible: pushing one way makes the model more dangerous, and pushing the other way makes it a bit safer than where it started.

But the response is not linear. The curve barely moves at small positive $$s$$, starts falling quickly around $$s=0.8$$, and flattens out after $$s≈2.2$$. It is tempting to say that doubling the update doubles the behavior change, since the weights themselves change linearly. That is not what happens, and it should not be surprising. The map from weights to a generated sequence, and then to a judge's verdict, is nonlinear at every step. A direction can be linear to manipulate and nonlinear to read out.

The steep middle section looks a lot like a critical band, and I am tempted to call it a phase transition. I do not want to overstate this. Without denser sampling, confidence intervals, and some evidence from internal representations, it is just a steep part of a curve.

## Where Does the Dial Bite?

Now we can ask where in the network the update matters. The simplest split is attention versus MLP, which are the orange and green lines in the figure above.

| Scope | ASR @ s=2 | ASR @ s=3 |
|---|---:|---:|
| All modules | **68.55%** | **87.64%** |
| Attention only | 14.00% | 28.91% |
| MLP only | 25.45% | 56.91% |

MLP is clearly more sensitive than attention. But notice that all modules together are stronger than either branch alone. Neither one explains the behavior on its own.

Going one level finer, I enable one projection matrix at a time.

<figure>
  <img src="{{ '/assets/images/param-steering/projection_overview.png' | relative_url }}" alt="Projection-level response curves">
  <figcaption>Same sweep, but the update is applied to only one projection type at a time (Q, K, V, O, Gate, Up, Down) across all layers. Left: Beaver Safety Score. Right: LlamaGuard ASR. The Q, K, V and Gate curves stay close to the base model. O, Up and Down separate clearly from the rest as s grows.</figcaption>
</figure>

| Single projection, s=3 | ASR |
|---|---:|
| `q_proj` | 2.18% |
| `k_proj` | 1.45% |
| `v_proj` | 1.82% |
| `o_proj` | **15.82%** |
| `gate_proj` | 2.91% |
| `up_proj` | 12.18% |
| `down_proj` | 9.27% |

Q, K and V barely react. `o_proj`, `up_proj` and `down_proj` are the ones that move. A natural story is that these are exactly the matrices that write their results back into the residual stream: `o_proj` writes the attention output, and `down_proj` writes the MLP output. Perturbing the part of the computation that writes to the stream has a more direct effect downstream. I find this plausible, but it is a hypothesis and these experiments do not test it.

There is also a confound I cannot rule out. At the same $$s$$, different matrices receive updates of different magnitude, so "this matrix is more sensitive" could mean the location matters, or it could just mean the update there is bigger. Separating the two needs a norm-matched comparison, which I have not done yet.

Also notice the scale. Even the best single projection gets to 15.82%, far below the 87.64% of the full update. The behavior is not sitting in one isolated matrix.

## A Hotspot in the Middle Layers

The other axis is depth. Qwen3-8B has 36 layers, so I enable the update in one layer at a time and measure how much safety degrades when going from $$s=-1$$ to $$s=3$$.

<figure>
  <img src="{{ '/assets/images/param-steering/layer_sensitivity.png' | relative_url }}" alt="Per-layer sensitivity">
  <figcaption>Per-layer sensitivity: the change in safety between s=-1 and s=3 when the update is applied to a single layer. Left: change in LlamaGuard ASR. Right: change in Beaver Safety Score. Both curves peak around layers 14 to 18 with a maximum at layer 16, while layers 30 to 35 are essentially flat.</figcaption>
</figure>

The profile is far from uniform. There is a sharp peak around layers 14 to 18, with layer 16 the highest on both judges: ASR changes by 3.45 percentage points and Safety Score by 0.993. The last six layers are close to silent. The two curves have a Pearson correlation of 0.894, which I take as reasonable evidence that the hotspot is not an artifact of one particular judge. Two measurements with different mechanisms agree on which layers matter, and the peak is formed by several neighboring layers rather than a single spike, which also makes it less likely to be noise.

It is worth being careful about what this does and does not say. What I modify is a LoRA update that has already been learned, not the base model. So the claim is that these layers give this dangerous direction more behavioral leverage. It is not that the model's safety knowledge is stored in these layers. It is tempting to connect this with [ROME](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html), which found that factual recall depends on mid-layer MLPs. Both results point to the middle of the network, but they ask different questions about different objects, so I would not treat them as the same finding.

## One Layer Is Weak, a Stretch of Layers Is Strong

Layer 16 is the most sensitive layer, yet on its own it moves ASR by only 3.45 points. What happens if we switch on a whole range at once?

<figure>
  <img src="{{ '/assets/images/param-steering/layer_range_overview.png' | relative_url }}" alt="Layer-range response curves">
  <figcaption>The 36 layers are split into three non-overlapping ranges, and the update is applied to one range at a time. Blue: layers 0 to 13. Orange: layers 14 to 29. Green: layers 30 to 35. The middle range carries most of the effect, the early range a smaller part, and the final range almost none.</figcaption>
</figure>

| Layers enabled | ASR @ s=2 | ASR @ s=3 |
|---|---:|---:|
| 0–13 | 8.73% | 19.82% |
| 14–29 | **32.55%** | **65.27%** |
| 30–35 | 1.09% | 1.45% |

<figure>
  <img src="{{ '/assets/images/param-steering/layer_map.svg' | relative_url }}" alt="Layer ranges of Qwen3-8B">
  <figcaption>Where the update matters, laid out over the depth of Qwen3-8B. Each cell is a layer. Darker shading means a larger ASR when only that range is enabled at s=3 (19.8%, 65.3%, 1.5%). The orange outline marks layer 16, the most sensitive single layer, and the bracket marks the 14 to 18 hotspot from the per-layer sweep.</figcaption>
</figure>

Turning on layers 14 to 29 alone gives 65.27% ASR at $$s=3$$. Measured from the base model, that recovers about 74% of the ASR increase of the full update. The early layers contribute something real (19.82% at $$s=3$$), so the direction is not confined to the middle. The final layers do almost nothing even when extrapolated.

The comparison I find most striking is between the two experiments. The most sensitive single layer produces a 3.45 point change, while a contiguous block of 16 layers produces 65.27. Because the judges are nonlinear, I cannot read the gap as a clean interaction term. But the order of magnitude is hard to explain away: the dangerous behavior is not controlled by one safety layer. It needs updates spread over a stretch of depth to enter the computation together.

<figure>
  <img src="{{ '/assets/images/param-steering/single_vs_range.svg' | relative_url }}" alt="Single layer versus layer ranges">
  <figcaption>The same quantity at different granularities: how much ASR rises when the scale goes from -1 to 3. The best single layer (orange) moves ASR by 3.45 points, while a contiguous block of 16 layers moves it by 65.27 and the full update by 87.46. Because the judges are nonlinear, the gap should not be read as an exact interaction term.</figcaption>
</figure>

## What I Think This Means

Putting these together, I think the most useful way to read the results is that two things which sound contradictory are both true. The update is a single direction, in the sense that it is one low-rank object you can scale with one number. But its implementation is distributed: it leaves traces in many layers and several matrices, and it only produces the full behavior when enough of them act together. LoRA and Task Arithmetic describe the geometry of the parameter change. Nothing in that geometry says that the change has to be implemented at one place in the network.

This also changes how I think about localization. "Which layer is responsible for safety" is not really a binary question. What the data supports is a sensitivity profile: layer 16 is the sharpest single point, layers 14 to 18 form a local hotspot, layers 14 to 29 are the main pathway, the early layers matter a bit, and the last layers are nearly silent. Different granularities give different answers, so any statement of the form "layer X handles safety" should come with the scale at which it was measured. If the direction were spread evenly through depth, we would not see a consistent peak at layers 14 to 18 on two independent judges, and we would not see such a large gap between the middle and the end of the network. I would call the result an anisotropic, distributed implementation: clearly biased toward the middle layers and toward MLP, but not collapsed onto any single coordinate.

I also like the control-system framing that this suggests. We usually evaluate safety fine-tuning as a discrete model version, such as before and after alignment. A continuous sweep makes the object of interest the transfer function instead. The flat region tells you how robust the model is to small perturbations, the steep region tells you how close the safety boundary is to a small parameter drift, and the saturated region tells you the behavior has been taken over by the direction. That is one way to understand why [a few benign fine-tuning steps can cross the safety boundary](https://openreview.net/forum?id=hTEGyKf0dZ), and it suggests that methods that repair safety by adding a safety vector, like [RESTA](https://aclanthology.org/2024.acl-long.762/), face a calibration problem: how much to add, and where.

## What I Do Not Know

The scope of these results is narrow, and I want to be explicit about it.

Everything here is Qwen3-8B, one Beaver-Danger direction, and CatQA-English. Whether the mid-layer hotspot is a stable property of how transformers turn a safety direction into behavior, or something specific to this adapter, is the most important open question. The natural next test is a different direction, such as one trained on PKU-Safety data, where I would expect the peak layer to shift but the normalized depth of the sensitive region to stay roughly in the middle. If the ordering of layers turned out to be close to random across directions, or if the hotspot were fully explained by update norm, I would have to retract the "stable structure" reading.

The measurements also have limits. Both judges are automatic and do not capture over-refusal or helpfulness, and I have only measured the safety side. I have not looked at MMLU, math or code, so none of this can be used to choose a deployment strength. With 550 deterministic samples and no bootstrap intervals or multiple seeds, small differences between neighboring values of $$s$$ may be within noise.

Finally, the causal claim has a limited range. A mask is an intervention, so I am fairly comfortable saying that these positions contribute to the behavior of this LoRA. I am not saying the base model's safety knowledge lives there, and I cannot rule out that the network reaches the same behavior through other paths.

The next steps are the boring and necessary ones: reproduce the sweep on another direction, add confidence intervals, run norm-matched comparisons to separate update size from position, and record residual-stream representations to see whether the steep part of the curve has a counterpart inside the model. If the curve is a sharp rotation of some internal representation, that would be a much stronger statement than anything I can make from behavior alone. If it only shows up in the judge's output, then the "critical band" is mostly a feature of how we measure.

## References

- Hu et al. [LoRA: Low-Rank Adaptation of Large Language Models](https://openreview.net/forum?id=nZeVKeeFYf9). ICLR 2022.
- Ilharco et al. [Editing Models with Task Arithmetic](https://openreview.net/forum?id=6t0Kwf8-jrj). ICLR 2023.
- Meng et al. [Locating and Editing Factual Associations in GPT](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html). NeurIPS 2022.
- Arditi et al. [Refusal in Language Models Is Mediated by a Single Direction](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f545448535dfde4f9786555403ab7c49-Abstract-Conference.html). NeurIPS 2024.
- Qi et al. [Fine-Tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!](https://openreview.net/forum?id=hTEGyKf0dZ). ICLR 2024.
- Bhardwaj et al. [Language Models are Homer Simpson! Safety Re-Alignment of Fine-Tuned Language Models through Task Arithmetic](https://aclanthology.org/2024.acl-long.762/). ACL 2024.
