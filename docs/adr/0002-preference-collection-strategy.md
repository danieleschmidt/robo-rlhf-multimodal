# ADR-0002: Human Preference Collection Strategy

## Status
Accepted

## Context
Human preference collection is the bottleneck for RLHF training. We need an efficient strategy for:
1. Generating meaningful preference pairs from demonstrations
2. Collecting high-quality human annotations
3. Handling disagreement between annotators
4. Scaling to large datasets

## Decision
We will implement a multi-stage preference collection strategy:

1. **Pair Generation**: Use diversity sampling to select trajectory segments that are:
   - Sufficiently different in outcome
   - Similar in initial conditions
   - Representative of the task distribution

2. **Annotation Interface**: Web-based interface with:
   - Side-by-side video playback
   - Keyboard shortcuts for quick annotation
   - Option to mark pairs as "incomparable"
   - Progress tracking and quality metrics

3. **Quality Control**: 
   - Multiple annotators per pair (minimum 3)
   - Inter-rater reliability tracking
   - Expert validation for disputed pairs
   - Automatic filtering of low-agreement pairs

4. **Aggregation**: Use Bradley-Terry model to convert pairwise preferences to scalar rewards

## Consequences
**Positive:**
- Systematic approach ensures high-quality preference data
- Web interface scales to many annotators
- Quality control mechanisms reduce noise
- Bradley-Terry model handles intransitive preferences

**Negative:**
- Requires significant human annotation time
- Web interface development overhead
- Multiple annotators increase cost
- May introduce bias from annotator selection

## Alternatives Considered
1. **Active learning**: Smart pair selection - deferred to future work due to complexity
2. **Single annotator**: Faster but lower quality - rejected due to reliability concerns
3. **Crowdsourcing platforms**: Considered but requires extensive quality control

## Date
2025-01-15