# Prompt Evolution Results

This document tracks the results of prompt optimization experiments for voyage-4-large.

## Target Datasets

| Dataset | Baseline | Domain |
|---------|----------|--------|
| AILACasedocs | 47.49% | Legal |
| FreshStackRetrieval | 50.79% | Code |
| AILAStatutes | 54.35% | Legal |
| DS1000Retrieval | 71.19% | Code |
| LegalQuAD | 74.63% | Legal |
| HC3FinanceRetrieval | 76.83% | Finance |
| ChatDoctorRetrieval | 77.20% | Medical |
| LegalSummarization | 78.09% | Legal |
| FinQARetrieval | 88.99% | Finance |

---


## Generation 0 Results (2026-02-20 15:54)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|

### Key Learnings

### Winning Prompts

```python
WINNING_PROMPTS = {
}
```

## Generation 1 Results (2026-02-20 15:54)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|

### Key Learnings

### Winning Prompts

```python
WINNING_PROMPTS = {
}
```

## Generation 2 Results (2026-02-20 15:54)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|

### Key Learnings

### Winning Prompts

```python
WINNING_PROMPTS = {
}
```

## Generation 3 Results (2026-02-20 15:54)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|

### Key Learnings

### Winning Prompts

```python
WINNING_PROMPTS = {
}
```

## Generation 4 Results (2026-02-20 15:54)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|

### Key Learnings

### Winning Prompts

```python
WINNING_PROMPTS = {
}
```

## Generation 5 Results (2026-02-20 15:54)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|

### Key Learnings

### Winning Prompts

```python
WINNING_PROMPTS = {
}
```

## Generation 0 Results (2026-02-20 16:02)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| LegalQuAD | 69.67% | 70.11% | +0.44% (+0.6%) | {text} |

### Key Learnings

**Improved:** LegalQuAD

### Winning Prompts

```python
WINNING_PROMPTS = {
    "LegalQuAD": {"query": "{text}", "corpus": "Legal reference: {text}"},
}
```

## Generation 1 Results (2026-02-20 16:13)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| LegalQuAD | 69.67% | 70.68% | +1.01% (+1.5%) | {text} |

### Key Learnings

**Improved:** LegalQuAD

### Winning Prompts

```python
WINNING_PROMPTS = {
    "LegalQuAD": {"query": "{text}", "corpus": "Canonical legal passage: {text}"},
}
```

## Generation 2 Results (2026-02-20 16:24)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| LegalQuAD | 69.67% | 70.68% | +1.01% (+1.5%) | {text} |

### Key Learnings

**Improved:** LegalQuAD

### Winning Prompts

```python
WINNING_PROMPTS = {
    "LegalQuAD": {"query": "{text}", "corpus": "Canonical legal passage: {text}"},
}
```

## Generation 3 Results (2026-02-20 16:35)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| LegalQuAD | 69.67% | 70.68% | +1.01% (+1.5%) | {text} |

### Key Learnings

**Improved:** LegalQuAD

### Winning Prompts

```python
WINNING_PROMPTS = {
    "LegalQuAD": {"query": "{text}", "corpus": "Canonical legal passage: {text}"},
}
```

## Generation 4 Results (2026-02-20 16:47)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| LegalQuAD | 69.67% | 70.68% | +1.01% (+1.5%) | {text} |

### Key Learnings

**Improved:** LegalQuAD

### Winning Prompts

```python
WINNING_PROMPTS = {
    "LegalQuAD": {"query": "{text}", "corpus": "Canonical legal passage: {text}"},
}
```

## Generation 5 Results (2026-02-20 16:58)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| LegalQuAD | 69.67% | 70.68% | +1.01% (+1.5%) | {text} |

### Key Learnings

**Improved:** LegalQuAD

### Winning Prompts

```python
WINNING_PROMPTS = {
    "LegalQuAD": {"query": "{text}", "corpus": "Canonical legal passage: {text}"},
}
```

## Generation 0 Results (2026-02-20 18:07)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| AILACasedocs | 42.20% | 43.24% | +1.04% (+2.5%) | Discover legal cases that provide contex... |

### Key Learnings

**Improved:** AILACasedocs

### Winning Prompts

```python
WINNING_PROMPTS = {
    "AILACasedocs": {"query": "Discover legal cases that provide contextually relevant insights for: {text}", "corpus": "Case law excerpt: {text}"},
}
```

## Generation 1 Results (2026-02-20 18:17)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| AILACasedocs | 42.20% | 43.35% | +1.15% (+2.7%) | Query law: {text} |

### Key Learnings

**Improved:** AILACasedocs

### Winning Prompts

```python
WINNING_PROMPTS = {
    "AILACasedocs": {"query": "Query law: {text}", "corpus": "Legal precedent document: {text}"},
}
```

## Generation 2 Results (2026-02-20 18:26)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| AILACasedocs | 42.20% | 43.51% | +1.31% (+3.1%) | Legal ask: {text} |

### Key Learnings

**Improved:** AILACasedocs

### Winning Prompts

```python
WINNING_PROMPTS = {
    "AILACasedocs": {"query": "Legal ask: {text}", "corpus": "Case law excerpt: {text}"},
}
```

## Generation 3 Results (2026-02-20 18:36)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| AILACasedocs | 42.20% | 43.51% | +1.31% (+3.1%) | Legal ask: {text} |

### Key Learnings

**Improved:** AILACasedocs

### Winning Prompts

```python
WINNING_PROMPTS = {
    "AILACasedocs": {"query": "Legal ask: {text}", "corpus": "Case law excerpt: {text}"},
}
```

## Generation 4 Results (2026-02-20 18:46)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| AILACasedocs | 42.20% | 43.64% | +1.44% (+3.4%) | Doc query: {text} |

### Key Learnings

**Improved:** AILACasedocs

### Winning Prompts

```python
WINNING_PROMPTS = {
    "AILACasedocs": {"query": "Doc query: {text}", "corpus": "Law excerpt: {text}"},
}
```

## Generation 5 Results (2026-02-20 18:56)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| AILACasedocs | 42.20% | 43.64% | +1.44% (+3.4%) | Doc query: {text} |

### Key Learnings

**Improved:** AILACasedocs

### Winning Prompts

```python
WINNING_PROMPTS = {
    "AILACasedocs": {"query": "Doc query: {text}", "corpus": "Law excerpt: {text}"},
}
```
