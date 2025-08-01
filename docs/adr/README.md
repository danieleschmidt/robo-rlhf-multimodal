# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the Robo-RLHF-Multimodal project.

## What is an ADR?

An Architecture Decision Record (ADR) is a document that captures an important architectural decision made along with its context and consequences.

## ADR Format

We use the following format for our ADRs:

```markdown
# ADR-XXXX: [Title]

## Status
[Proposed | Accepted | Rejected | Deprecated | Superseded by ADR-YYYY]

## Context
[What is the issue that we're seeing that is motivating this decision or change?]

## Decision
[What is the change that we're proposing or have agreed to implement?]

## Consequences
[What becomes easier or more difficult to do and any risks introduced by this change?]

## Alternatives Considered
[What other alternatives were considered and why were they not chosen?]

## Date
[When was this decision made?]
```

## ADR Index

| ID | Title | Status | Date |
|----|-------|--------|------|
| [0001](0001-multimodal-architecture.md) | Multimodal Architecture Design | Accepted | 2025-01-15 |
| [0002](0002-preference-collection-strategy.md) | Human Preference Collection Strategy | Accepted | 2025-01-15 |
| [0003](0003-reward-model-architecture.md) | Reward Model Architecture | Accepted | 2025-01-15 |

## Creating a New ADR

1. Copy the template from `template.md`
2. Rename it with the next sequential number: `XXXX-descriptive-title.md`
3. Fill in all sections
4. Submit a pull request for review
5. Update this index when the ADR is approved

## ADR Lifecycle

- **Proposed**: The ADR is under discussion
- **Accepted**: The ADR has been approved and should be implemented
- **Rejected**: The ADR has been rejected and will not be implemented
- **Deprecated**: The ADR was previously accepted but is no longer relevant
- **Superseded**: The ADR has been replaced by a newer ADR