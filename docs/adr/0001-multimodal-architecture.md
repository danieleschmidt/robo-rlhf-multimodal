# ADR-0001: Multimodal Architecture Design

## Status
Accepted

## Context
The system needs to process multiple types of sensory data (RGB images, depth images, proprioceptive state, force/torque) for both preference learning and policy execution. We need to decide on the architecture for combining these modalities effectively.

## Decision
We will use a modular encoder architecture where:
1. Each modality has a dedicated encoder (vision encoder for RGB/depth, MLP for proprioception)
2. Encoders produce fixed-size feature vectors that are concatenated
3. A fusion network processes the combined features
4. This architecture is shared between reward models and policy networks

Vision encoding will use pre-trained models (CLIP, ResNet) with optional fine-tuning.

## Consequences
**Positive:**
- Modular design allows easy addition of new modalities
- Pre-trained vision encoders provide strong feature representations
- Shared architecture reduces implementation complexity
- Easy to ablate individual modalities for analysis

**Negative:**
- Fixed concatenation may not capture complex cross-modal interactions
- Requires careful balancing of feature dimensions across modalities
- May need modality-specific normalization strategies

## Alternatives Considered
1. **Cross-attention mechanisms**: More complex but potentially better cross-modal fusion
2. **Early fusion**: Concatenating raw inputs - rejected due to dimensionality mismatch
3. **Late fusion**: Separate networks per modality with late combination - rejected due to increased complexity

## Date
2025-01-15