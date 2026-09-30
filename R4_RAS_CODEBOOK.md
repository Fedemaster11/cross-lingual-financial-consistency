# R4 Cross-Lingual Rationale Alignment Score (RAS)

## Per-rationale coding
direction:
- positive
- negative
- non_directional

magnitude:
- strong
- mild
- neutral

causal_structure:
- 0 = no substantive reason
- 1 = one inferential link
- 2 = event -> intermediate business/financial mechanism -> stock/market implication

uncertainty (descriptive; excluded from RAS):
- 0 = no hedge
- 1 = modal/mild hedge
- 2 = explicit uncertainty / unclear / depends on context

label_alignment (within-language; excluded from RAS):
- aligned
- partial
- contradicted

prompt_leakage:
- yes / no

## Pairwise RAS
For one statement i and languages a,b:

S_D = 1 if direction matches else 0
S_M = 1 if direction AND magnitude match else 0
S_C = 1 if causal_structure matches else 0

RAS(i,a,b) = (S_D + S_M + S_C) / 3

Possible values:
0, 1/3, 2/3, 1

Language-pair RAS is the mean over the 32 selected statements.

## 2x2 interpretation
same label + high RAS:
output and rationale aligned

same label + low RAS:
same output, different rationale pattern

different label + high RAS:
similar rationale, different output calibration

different label + low RAS:
output and rationale divergence

This is exploratory analysis of generated explanations, not access to hidden chain-of-thought.
