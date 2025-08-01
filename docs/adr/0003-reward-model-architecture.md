# ADR-0003: Reward Model Architecture

## Status
Accepted

## Context
The reward model is critical for RLHF training quality. It must:
1. Process multimodal observations effectively
2. Be trainable from preference data
3. Provide meaningful reward signals for policy training
4. Be computationally efficient for large-scale training

## Decision
We will use a shared backbone architecture for reward models:

1. **Input Processing**: Same multimodal encoder as policy networks
2. **Reward Head**: 2-layer MLP that maps encoded features to scalar reward
3. **Training**: Pairwise ranking loss using Bradley-Terry model
4. **Ensemble**: Train multiple reward models and average predictions for robustness

Key design choices:
- Shared encoder weights with policy (optional, configurable)
- Batch normalization in reward head for training stability
- Dropout for regularization
- Option for uncertainty estimation via ensemble disagreement

## Consequences
**Positive:**
- Shared encoder reduces computational overhead
- Ensemble improves robustness and provides uncertainty estimates
- Modular design allows easy experimentation
- Standard architecture is well-understood and debuggable

**Negative:**
- Shared encoder may create optimization conflicts
- Ensemble increases computational cost
- May require careful hyperparameter tuning for stability

## Alternatives Considered
1. **Separate encoder**: Better separation of concerns but higher computational cost
2. **Transformer architecture**: More expressive but requires more data and computation
3. **Value-based rewards**: Using value function estimates - rejected due to bootstrap bias

## Date
2025-01-15