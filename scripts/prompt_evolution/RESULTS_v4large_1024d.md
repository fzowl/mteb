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


## Generation 0 Results (2026-01-16 16:45)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| AILAStatutes | 54.35% | 55.48% | +1.13% (+2.1%) | Based on scenario details, find exact ma... |

### Key Learnings

**Improved:** AILAStatutes

### Winning Prompts

```python
WINNING_PROMPTS = {
    "AILAStatutes": "Based on scenario details, find exact match legal statutes",
}
```

## Generation 1 Results (2026-01-16 16:46)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| AILAStatutes | 54.35% | 55.97% | +1.62% (+3.0%) | Instruct: Match the provided context wit... |

### Key Learnings

**Improved:** AILAStatutes

### Winning Prompts

```python
WINNING_PROMPTS = {
    "AILAStatutes": "Instruct: Match the provided context with accurate legal provisions
Query: ",
}
```

## Generation 2 Results (2026-01-16 16:47)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| AILAStatutes | 54.35% | 55.97% | +1.62% (+3.0%) | Instruct: Match the provided context wit... |

### Key Learnings

**Improved:** AILAStatutes

### Winning Prompts

```python
WINNING_PROMPTS = {
    "AILAStatutes": "Instruct: Match the provided context with accurate legal provisions
Query: ",
}
```

## Generation 0 Results (2026-01-16 17:02)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| AILACasedocs | 47.49% | 47.49% | -0.00% (-0.0%) | nan |

### Key Learnings

**Hurt by prompts:** AILACasedocs (baseline is better)

### Winning Prompts

```python
WINNING_PROMPTS = {
    "AILACasedocs": "",  # baseline is better
}
```

## Generation 1 Results (2026-01-16 17:10)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| AILACasedocs | 47.49% | 47.49% | -0.00% (-0.0%) | nan |

### Key Learnings

**Hurt by prompts:** AILACasedocs (baseline is better)

### Winning Prompts

```python
WINNING_PROMPTS = {
    "AILACasedocs": "",  # baseline is better
}
```

## Generation 2 Results (2026-01-16 17:16)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| AILACasedocs | 47.49% | 47.49% | -0.00% (-0.0%) | nan |

### Key Learnings

**Hurt by prompts:** AILACasedocs (baseline is better)

### Winning Prompts

```python
WINNING_PROMPTS = {
    "AILACasedocs": "",  # baseline is better
}
```

## Generation 0 Results (2026-01-16 17:29)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| DS1000Retrieval | 71.19% | 71.23% | +0.05% (+0.1%) | For computational tasks, retrieve script... |

### Key Learnings

**Improved:** DS1000Retrieval

### Winning Prompts

```python
WINNING_PROMPTS = {
    "DS1000Retrieval": "For computational tasks, retrieve scripts using scipy and numpy",
}
```

## Generation 1 Results (2026-01-16 17:29)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| DS1000Retrieval | 71.19% | 71.23% | +0.05% (+0.1%) | For computational tasks, retrieve script... |

### Key Learnings

**Improved:** DS1000Retrieval

### Winning Prompts

```python
WINNING_PROMPTS = {
    "DS1000Retrieval": "For computational tasks, retrieve scripts using scipy and numpy",
}
```

## Generation 2 Results (2026-01-16 17:34)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| DS1000Retrieval | 71.19% | 71.23% | +0.05% (+0.1%) | For computational tasks, retrieve script... |

### Key Learnings

**Improved:** DS1000Retrieval

### Winning Prompts

```python
WINNING_PROMPTS = {
    "DS1000Retrieval": "For computational tasks, retrieve scripts using scipy and numpy",
}
```

## Generation 0 Results (2026-01-16 17:44)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| LegalQuAD | 74.63% | 74.60% | -0.03% (-0.0%) | nan |

### Key Learnings

**Hurt by prompts:** LegalQuAD (baseline is better)

### Winning Prompts

```python
WINNING_PROMPTS = {
    "LegalQuAD": "",  # baseline is better
}
```

## Generation 1 Results (2026-01-16 17:50)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| LegalQuAD | 74.63% | 74.60% | -0.03% (-0.0%) | nan |

### Key Learnings

**Hurt by prompts:** LegalQuAD (baseline is better)

### Winning Prompts

```python
WINNING_PROMPTS = {
    "LegalQuAD": "",  # baseline is better
}
```

## Generation 2 Results (2026-01-16 17:55)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| LegalQuAD | 74.63% | 74.60% | -0.03% (-0.0%) | nan |

### Key Learnings

**Hurt by prompts:** LegalQuAD (baseline is better)

### Winning Prompts

```python
WINNING_PROMPTS = {
    "LegalQuAD": "",  # baseline is better
}
```

## Generation 0 Results (2026-01-16 18:05)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| HC3FinanceRetrieval | 76.83% | 77.61% | +0.78% (+1.0%) | Find comprehensive finance documents tha... |

### Key Learnings

**Improved:** HC3FinanceRetrieval

### Winning Prompts

```python
WINNING_PROMPTS = {
    "HC3FinanceRetrieval": "Find comprehensive finance documents that answer the question accurately",
}
```

## Generation 1 Results (2026-01-16 18:05)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| HC3FinanceRetrieval | 76.83% | 77.61% | +0.78% (+1.0%) | Find comprehensive finance documents tha... |

### Key Learnings

**Improved:** HC3FinanceRetrieval

### Winning Prompts

```python
WINNING_PROMPTS = {
    "HC3FinanceRetrieval": "Find comprehensive finance documents that answer the question accurately",
}
```

## Generation 2 Results (2026-01-16 18:06)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| HC3FinanceRetrieval | 76.83% | 77.61% | +0.78% (+1.0%) | Find comprehensive finance documents tha... |

### Key Learnings

**Improved:** HC3FinanceRetrieval

### Winning Prompts

```python
WINNING_PROMPTS = {
    "HC3FinanceRetrieval": "Find comprehensive finance documents that answer the question accurately",
}
```

## Generation 0 Results (2026-01-16 18:18)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| ChatDoctorRetrieval | 77.20% | 77.20% | -0.00% (-0.0%) | nan |

### Key Learnings

**Hurt by prompts:** ChatDoctorRetrieval (baseline is better)

### Winning Prompts

```python
WINNING_PROMPTS = {
    "ChatDoctorRetrieval": "",  # baseline is better
}
```

## Generation 1 Results (2026-01-16 18:24)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| ChatDoctorRetrieval | 77.20% | 77.20% | -0.00% (-0.0%) | nan |

### Key Learnings

**Hurt by prompts:** ChatDoctorRetrieval (baseline is better)

### Winning Prompts

```python
WINNING_PROMPTS = {
    "ChatDoctorRetrieval": "",  # baseline is better
}
```

## Generation 2 Results (2026-01-16 18:31)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| ChatDoctorRetrieval | 77.20% | 77.20% | -0.00% (-0.0%) | nan |

### Key Learnings

**Hurt by prompts:** ChatDoctorRetrieval (baseline is better)

### Winning Prompts

```python
WINNING_PROMPTS = {
    "ChatDoctorRetrieval": "",  # baseline is better
}
```

## Generation 0 Results (2026-01-16 18:41)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| LegalSummarization | 78.09% | 78.85% | +0.76% (+1.0%) | Search using domain-specific language to... |

### Key Learnings

**Improved:** LegalSummarization

### Winning Prompts

```python
WINNING_PROMPTS = {
    "LegalSummarization": "Search using domain-specific language to retrieve accurate documents",
}
```

## Generation 1 Results (2026-01-16 18:42)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| LegalSummarization | 78.09% | 78.85% | +0.76% (+1.0%) | Search using domain-specific language to... |

### Key Learnings

**Improved:** LegalSummarization

### Winning Prompts

```python
WINNING_PROMPTS = {
    "LegalSummarization": "Search using domain-specific language to retrieve accurate documents",
}
```

## Generation 2 Results (2026-01-16 18:43)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| LegalSummarization | 78.09% | 78.85% | +0.76% (+1.0%) | Search using domain-specific language to... |

### Key Learnings

**Improved:** LegalSummarization

### Winning Prompts

```python
WINNING_PROMPTS = {
    "LegalSummarization": "Search using domain-specific language to retrieve accurate documents",
}
```

## Generation 0 Results (2026-01-16 18:46)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| FinQARetrieval | 88.99% | 88.99% | -0.00% (-0.0%) | nan |

### Key Learnings

**Hurt by prompts:** FinQARetrieval (baseline is better)

### Winning Prompts

```python
WINNING_PROMPTS = {
    "FinQARetrieval": "",  # baseline is better
}
```

## Generation 1 Results (2026-01-16 18:48)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| FinQARetrieval | 88.99% | 88.99% | -0.00% (-0.0%) | nan |

### Key Learnings

**Hurt by prompts:** FinQARetrieval (baseline is better)

### Winning Prompts

```python
WINNING_PROMPTS = {
    "FinQARetrieval": "",  # baseline is better
}
```

## Generation 2 Results (2026-01-16 18:50)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| FinQARetrieval | 88.99% | 88.99% | -0.00% (-0.0%) | nan |

### Key Learnings

**Hurt by prompts:** FinQARetrieval (baseline is better)

### Winning Prompts

```python
WINNING_PROMPTS = {
    "FinQARetrieval": "",  # baseline is better
}
```

## Generation 0 Results (2026-01-16 19:12)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| FreshStackRetrieval | 50.79% | 50.79% | -0.00% (-0.0%) | nan |

### Key Learnings

**Hurt by prompts:** FreshStackRetrieval (baseline is better)

### Winning Prompts

```python
WINNING_PROMPTS = {
    "FreshStackRetrieval": "",  # baseline is better
}
```

## Generation 1 Results (2026-01-16 19:28)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| FreshStackRetrieval | 50.79% | 50.79% | -0.00% (-0.0%) | nan |

### Key Learnings

**Hurt by prompts:** FreshStackRetrieval (baseline is better)

### Winning Prompts

```python
WINNING_PROMPTS = {
    "FreshStackRetrieval": "",  # baseline is better
}
```

## Generation 2 Results (2026-01-16 19:42)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| FreshStackRetrieval | 50.79% | 50.79% | -0.00% (-0.0%) | nan |

### Key Learnings

**Hurt by prompts:** FreshStackRetrieval (baseline is better)

### Winning Prompts

```python
WINNING_PROMPTS = {
    "FreshStackRetrieval": "",  # baseline is better
}
```

## Generation 0 Results (2026-01-18 15:43)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|

### Key Learnings

### Winning Prompts

```python
WINNING_PROMPTS = {
}
```

## Generation 1 Results (2026-01-18 15:43)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|

### Key Learnings

### Winning Prompts

```python
WINNING_PROMPTS = {
}
```

## Generation 2 Results (2026-01-18 15:43)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|

### Key Learnings

### Winning Prompts

```python
WINNING_PROMPTS = {
}
```

## Generation 0 Results (2026-01-18 16:01)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| FinanceBenchRetrieval | 93.16% | 93.80% | +0.64% (+0.7%) | Surface relevant financial information f... |
| HumanEvalRetrieval | 99.36% | 99.68% | +0.32% (+0.3%) | Locate and return the Python function so... |
| WikiSQLRetrieval | 96.70% | 98.87% | +2.17% (+2.2%) | For the database question, retrieve the ... |

### Key Learnings

**Improved:** FinanceBenchRetrieval, HumanEvalRetrieval, WikiSQLRetrieval

### Winning Prompts

```python
WINNING_PROMPTS = {
    "FinanceBenchRetrieval": "Surface relevant financial information from annual and quarterly reports",
    "HumanEvalRetrieval": "Locate and return the Python function solving the problem",
    "WikiSQLRetrieval": "For the database question, retrieve the correct SQL query",
}
```

## Generation 1 Results (2026-01-18 16:09)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| FinanceBenchRetrieval | 93.16% | 93.89% | +0.73% (+0.8%) | Convey the essential financial data usin... |
| HumanEvalRetrieval | 99.36% | 99.68% | +0.32% (+0.3%) | Locate and return the Python function so... |
| WikiSQLRetrieval | 96.70% | 98.87% | +2.17% (+2.2%) | For the database question, retrieve the ... |

### Key Learnings

**Improved:** FinanceBenchRetrieval, HumanEvalRetrieval, WikiSQLRetrieval

### Winning Prompts

```python
WINNING_PROMPTS = {
    "FinanceBenchRetrieval": "Convey the essential financial data using filing-based analysis",
    "HumanEvalRetrieval": "Locate and return the Python function solving the problem",
    "WikiSQLRetrieval": "For the database question, retrieve the correct SQL query",
}
```

## Generation 2 Results (2026-01-18 16:17)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| FinanceBenchRetrieval | 93.16% | 93.89% | +0.73% (+0.8%) | Convey the essential financial data usin... |
| HumanEvalRetrieval | 99.36% | 99.77% | +0.41% (+0.4%) | Identify the appropriate Python function... |
| WikiSQLRetrieval | 96.70% | 99.02% | +2.32% (+2.4%) | For the database inquiry, surface the SQ... |

### Key Learnings

**Improved:** FinanceBenchRetrieval, HumanEvalRetrieval, WikiSQLRetrieval

### Winning Prompts

```python
WINNING_PROMPTS = {
    "FinanceBenchRetrieval": "Convey the essential financial data using filing-based analysis",
    "HumanEvalRetrieval": "Identify the appropriate Python function for the problem",
    "WikiSQLRetrieval": "For the database inquiry, surface the SQL query that matches",
}
```

## Generation 3 Results (2026-01-19 20:19)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| AILACasedocs | 47.49% | 47.49% | -0.00% (-0.0%) | nan |
| AILAStatutes | 54.35% | 55.97% | +1.62% (+3.0%) | Instruct: Match the provided context wit... |
| ChatDoctorRetrieval | 77.20% | 77.20% | -0.00% (-0.0%) | nan |
| DS1000Retrieval | 71.19% | 71.23% | +0.05% (+0.1%) | For computational tasks, retrieve script... |
| FinQARetrieval | 88.99% | 88.99% | -0.00% (-0.0%) | nan |
| FinanceBenchRetrieval | 93.16% | 93.89% | +0.73% (+0.8%) | Convey the essential financial data usin... |
| FreshStackRetrieval | 50.79% | 50.79% | -0.00% (-0.0%) | nan |
| HC3FinanceRetrieval | 76.83% | 77.61% | +0.78% (+1.0%) | Find comprehensive finance documents tha... |
| HumanEvalRetrieval | 99.36% | 99.77% | +0.41% (+0.4%) | Identify the appropriate Python function... |
| LegalQuAD | 74.63% | 74.60% | -0.03% (-0.0%) | nan |
| LegalSummarization | 78.09% | 78.85% | +0.76% (+1.0%) | Search using domain-specific language to... |
| WikiSQLRetrieval | 96.70% | 99.02% | +2.32% (+2.4%) | For the database inquiry, surface the SQ... |

### Key Learnings

**Improved:** AILAStatutes, DS1000Retrieval, HC3FinanceRetrieval, LegalSummarization, FinanceBenchRetrieval, WikiSQLRetrieval, HumanEvalRetrieval

**Hurt by prompts:** AILACasedocs, FreshStackRetrieval, LegalQuAD, ChatDoctorRetrieval, FinQARetrieval (baseline is better)

### Winning Prompts

```python
WINNING_PROMPTS = {
    "AILACasedocs": "",  # baseline is better
    "AILAStatutes": "Instruct: Match the provided context with accurate legal provisions
Query: ",
    "ChatDoctorRetrieval": "",  # baseline is better
    "DS1000Retrieval": "For computational tasks, retrieve scripts using scipy and numpy",
    "FinQARetrieval": "",  # baseline is better
    "FinanceBenchRetrieval": "Convey the essential financial data using filing-based analysis",
    "FreshStackRetrieval": "",  # baseline is better
    "HC3FinanceRetrieval": "Find comprehensive finance documents that answer the question accurately",
    "HumanEvalRetrieval": "Identify the appropriate Python function for the problem",
    "LegalQuAD": "",  # baseline is better
    "LegalSummarization": "Search using domain-specific language to retrieve accurate documents",
    "WikiSQLRetrieval": "For the database inquiry, surface the SQL query that matches",
}
```

## Generation 4 Results (2026-01-19 21:51)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| AILACasedocs | 47.49% | 47.49% | -0.00% (-0.0%) | nan |
| AILAStatutes | 54.35% | 56.26% | +1.91% (+3.5%) | Instruct: Match context with the best-fi... |
| ChatDoctorRetrieval | 77.20% | 77.20% | -0.00% (-0.0%) | nan |
| DS1000Retrieval | 71.19% | 71.23% | +0.05% (+0.1%) | For computational tasks, retrieve script... |
| FinQARetrieval | 88.99% | 88.99% | -0.00% (-0.0%) | nan |
| FinanceBenchRetrieval | 93.16% | 93.89% | +0.73% (+0.8%) | Convey the essential financial data usin... |
| FreshStackRetrieval | 50.79% | 50.79% | +0.00% (+0.0%) | Tap into Stack Overflow for resolution p... |
| HC3FinanceRetrieval | 76.83% | 78.50% | +1.67% (+2.2%) | Seek out finance Q&A documents answering... |
| HumanEvalRetrieval | 99.36% | 99.77% | +0.41% (+0.4%) | Identify the appropriate Python function... |
| LegalQuAD | 74.63% | 75.80% | +1.17% (+1.6%) | Surface directly applicable legal conten... |
| LegalSummarization | 78.09% | 78.96% | +0.87% (+1.1%) | Search for comprehensive legal documents... |
| WikiSQLRetrieval | 96.70% | 99.02% | +2.32% (+2.4%) | For the database inquiry, surface the SQ... |

### Key Learnings

**Improved:** AILAStatutes, DS1000Retrieval, LegalQuAD, HC3FinanceRetrieval, LegalSummarization, FinanceBenchRetrieval, WikiSQLRetrieval, HumanEvalRetrieval

**Hurt by prompts:** AILACasedocs, ChatDoctorRetrieval, FinQARetrieval (baseline is better)

### Winning Prompts

```python
WINNING_PROMPTS = {
    "AILACasedocs": "",  # baseline is better
    "AILAStatutes": "Instruct: Match context with the best-fitting legal statute provisions
Query: ",
    "ChatDoctorRetrieval": "",  # baseline is better
    "DS1000Retrieval": "For computational tasks, retrieve scripts using scipy and numpy",
    "FinQARetrieval": "",  # baseline is better
    "FinanceBenchRetrieval": "Convey the essential financial data using filing-based analysis",
    "FreshStackRetrieval": "",  # baseline is better
    "HC3FinanceRetrieval": "Seek out finance Q&A documents answering the question completely",
    "HumanEvalRetrieval": "Identify the appropriate Python function for the problem",
    "LegalQuAD": "Surface directly applicable legal content",
    "LegalSummarization": "Search for comprehensive legal documents to aid summarization",
    "WikiSQLRetrieval": "For the database inquiry, surface the SQL query that matches",
}
```

## Generation 5 Results (2026-01-19 23:15)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| AILACasedocs | 47.49% | 47.49% | -0.00% (-0.0%) | nan |
| AILAStatutes | 54.35% | 56.26% | +1.91% (+3.5%) | Instruct: Match context with the best-fi... |
| ChatDoctorRetrieval | 77.20% | 77.20% | -0.00% (-0.0%) | nan |
| DS1000Retrieval | 71.19% | 71.23% | +0.05% (+0.1%) | For computational tasks, retrieve script... |
| FinQARetrieval | 88.99% | 88.99% | -0.00% (-0.0%) | nan |
| FinanceBenchRetrieval | 93.16% | 93.90% | +0.74% (+0.8%) | Derive pertinent financial data using pr... |
| FreshStackRetrieval | 50.79% | 50.79% | +0.00% (+0.0%) | Tap into Stack Overflow for resolution p... |
| HC3FinanceRetrieval | 76.83% | 78.50% | +1.67% (+2.2%) | Seek out finance Q&A documents answering... |
| HumanEvalRetrieval | 99.36% | 100.00% | +0.64% (+0.6%) | Return the easiest Python function solvi... |
| LegalQuAD | 74.63% | 75.80% | +1.17% (+1.6%) | Surface directly applicable legal conten... |
| LegalSummarization | 78.09% | 79.20% | +1.11% (+1.4%) | Search using domain-specific language to... |
| WikiSQLRetrieval | 96.70% | 99.02% | +2.32% (+2.4%) | For the database inquiry, surface the SQ... |

### Key Learnings

**Improved:** AILAStatutes, DS1000Retrieval, LegalQuAD, HC3FinanceRetrieval, LegalSummarization, FinanceBenchRetrieval, WikiSQLRetrieval, HumanEvalRetrieval

**Hurt by prompts:** AILACasedocs, ChatDoctorRetrieval, FinQARetrieval (baseline is better)

### Winning Prompts

```python
WINNING_PROMPTS = {
    "AILACasedocs": "",  # baseline is better
    "AILAStatutes": "Instruct: Match context with the best-fitting legal statute provisions
Query: ",
    "ChatDoctorRetrieval": "",  # baseline is better
    "DS1000Retrieval": "For computational tasks, retrieve scripts using scipy and numpy",
    "FinQARetrieval": "",  # baseline is better
    "FinanceBenchRetrieval": "Derive pertinent financial data using primary SEC filings",
    "FreshStackRetrieval": "",  # baseline is better
    "HC3FinanceRetrieval": "Seek out finance Q&A documents answering the question completely",
    "HumanEvalRetrieval": "Return the easiest Python function solving the problem",
    "LegalQuAD": "Surface directly applicable legal content",
    "LegalSummarization": "Search using domain-specific language to extract comprehensive summaries",
    "WikiSQLRetrieval": "For the database inquiry, surface the SQL query that matches",
}
```

## Generation 6 Results (2026-01-20 00:36)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| AILACasedocs | 47.49% | 47.49% | -0.00% (-0.0%) | nan |
| AILAStatutes | 54.35% | 56.26% | +1.91% (+3.5%) | Instruct: Match context with the best-fi... |
| ChatDoctorRetrieval | 77.20% | 77.39% | +0.19% (+0.2%) | Clarify medically with comprehensive, ef... |
| DS1000Retrieval | 71.19% | 71.23% | +0.05% (+0.1%) | For computational tasks, retrieve script... |
| FinQARetrieval | 88.99% | 88.99% | -0.00% (-0.0%) | nan |
| FinanceBenchRetrieval | 93.16% | 93.90% | +0.74% (+0.8%) | Derive pertinent financial data using pr... |
| FreshStackRetrieval | 50.79% | 50.79% | +0.00% (+0.0%) | Tap into Stack Overflow for resolution p... |
| HC3FinanceRetrieval | 76.83% | 78.50% | +1.67% (+2.2%) | Seek out finance Q&A documents answering... |
| HumanEvalRetrieval | 99.36% | 100.00% | +0.64% (+0.6%) | Return the easiest Python function solvi... |
| LegalQuAD | 74.63% | 75.80% | +1.17% (+1.6%) | Surface directly applicable legal conten... |
| LegalSummarization | 78.09% | 79.20% | +1.11% (+1.4%) | Search using domain-specific language to... |
| WikiSQLRetrieval | 96.70% | 99.02% | +2.32% (+2.4%) | For the database inquiry, surface the SQ... |

### Key Learnings

**Improved:** AILAStatutes, DS1000Retrieval, LegalQuAD, HC3FinanceRetrieval, ChatDoctorRetrieval, LegalSummarization, FinanceBenchRetrieval, WikiSQLRetrieval, HumanEvalRetrieval

**Hurt by prompts:** AILACasedocs, FinQARetrieval (baseline is better)

### Winning Prompts

```python
WINNING_PROMPTS = {
    "AILACasedocs": "",  # baseline is better
    "AILAStatutes": "Instruct: Match context with the best-fitting legal statute provisions
Query: ",
    "ChatDoctorRetrieval": "Clarify medically with comprehensive, effective responses",
    "DS1000Retrieval": "For computational tasks, retrieve scripts using scipy and numpy",
    "FinQARetrieval": "",  # baseline is better
    "FinanceBenchRetrieval": "Derive pertinent financial data using primary SEC filings",
    "FreshStackRetrieval": "",  # baseline is better
    "HC3FinanceRetrieval": "Seek out finance Q&A documents answering the question completely",
    "HumanEvalRetrieval": "Return the easiest Python function solving the problem",
    "LegalQuAD": "Surface directly applicable legal content",
    "LegalSummarization": "Search using domain-specific language to extract comprehensive summaries",
    "WikiSQLRetrieval": "For the database inquiry, surface the SQL query that matches",
}
```

## Generation 7 Results (2026-01-20 01:45)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| AILACasedocs | 47.49% | 47.53% | +0.04% (+0.1%) | Law query: |
| AILAStatutes | 54.35% | 56.26% | +1.91% (+3.5%) | Instruct: Match context with the best-fi... |
| ChatDoctorRetrieval | 77.20% | 77.39% | +0.19% (+0.2%) | Clarify medically with comprehensive, ef... |
| DS1000Retrieval | 71.19% | 71.23% | +0.05% (+0.1%) | For computational tasks, retrieve script... |
| FinQARetrieval | 88.99% | 88.99% | -0.00% (-0.0%) | nan |
| FinanceBenchRetrieval | 93.16% | 93.90% | +0.74% (+0.8%) | Derive pertinent financial data using pr... |
| FreshStackRetrieval | 50.79% | 50.79% | +0.00% (+0.0%) | Tap into Stack Overflow for resolution p... |
| HC3FinanceRetrieval | 76.83% | 78.50% | +1.67% (+2.2%) | Seek out finance Q&A documents answering... |
| HumanEvalRetrieval | 99.36% | 100.00% | +0.64% (+0.6%) | Return the easiest Python function solvi... |
| LegalQuAD | 74.63% | 75.80% | +1.17% (+1.6%) | Surface directly applicable legal conten... |
| LegalSummarization | 78.09% | 79.20% | +1.11% (+1.4%) | Search using domain-specific language to... |
| WikiSQLRetrieval | 96.70% | 99.02% | +2.32% (+2.4%) | For the database inquiry, surface the SQ... |

### Key Learnings

**Improved:** AILACasedocs, AILAStatutes, DS1000Retrieval, LegalQuAD, HC3FinanceRetrieval, ChatDoctorRetrieval, LegalSummarization, FinanceBenchRetrieval, WikiSQLRetrieval, HumanEvalRetrieval

**Hurt by prompts:** FinQARetrieval (baseline is better)

### Winning Prompts

```python
WINNING_PROMPTS = {
    "AILACasedocs": "Law query:",
    "AILAStatutes": "Instruct: Match context with the best-fitting legal statute provisions
Query: ",
    "ChatDoctorRetrieval": "Clarify medically with comprehensive, effective responses",
    "DS1000Retrieval": "For computational tasks, retrieve scripts using scipy and numpy",
    "FinQARetrieval": "",  # baseline is better
    "FinanceBenchRetrieval": "Derive pertinent financial data using primary SEC filings",
    "FreshStackRetrieval": "",  # baseline is better
    "HC3FinanceRetrieval": "Seek out finance Q&A documents answering the question completely",
    "HumanEvalRetrieval": "Return the easiest Python function solving the problem",
    "LegalQuAD": "Surface directly applicable legal content",
    "LegalSummarization": "Search using domain-specific language to extract comprehensive summaries",
    "WikiSQLRetrieval": "For the database inquiry, surface the SQL query that matches",
}
```

## Generation 8 Results (2026-01-20 03:06)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| AILACasedocs | 47.49% | 47.79% | +0.30% (+0.6%) | Court request: |
| AILAStatutes | 54.35% | 56.26% | +1.91% (+3.5%) | Instruct: Match context with the best-fi... |
| ChatDoctorRetrieval | 77.20% | 77.39% | +0.19% (+0.2%) | Clarify medically with comprehensive, ef... |
| DS1000Retrieval | 71.19% | 71.23% | +0.05% (+0.1%) | For computational tasks, retrieve script... |
| FinQARetrieval | 88.99% | 88.99% | -0.00% (-0.0%) | nan |
| FinanceBenchRetrieval | 93.16% | 93.91% | +0.75% (+0.8%) | Instruct: Gather pertinent financial ins... |
| FreshStackRetrieval | 50.79% | 50.79% | +0.00% (+0.0%) | Tap into Stack Overflow for resolution p... |
| HC3FinanceRetrieval | 76.83% | 78.50% | +1.67% (+2.2%) | Seek out finance Q&A documents answering... |
| HumanEvalRetrieval | 99.36% | 100.00% | +0.64% (+0.6%) | Return the easiest Python function solvi... |
| LegalQuAD | 74.63% | 75.80% | +1.17% (+1.6%) | Surface directly applicable legal conten... |
| LegalSummarization | 78.09% | 79.29% | +1.20% (+1.5%) | Collect legal documents ensuring broad s... |
| WikiSQLRetrieval | 96.70% | 99.04% | +2.34% (+2.4%) | Uncover SQL query precisely tailored to ... |

### Key Learnings

**Improved:** AILACasedocs, AILAStatutes, DS1000Retrieval, LegalQuAD, HC3FinanceRetrieval, ChatDoctorRetrieval, LegalSummarization, FinanceBenchRetrieval, WikiSQLRetrieval, HumanEvalRetrieval

**Hurt by prompts:** FinQARetrieval (baseline is better)

### Winning Prompts

```python
WINNING_PROMPTS = {
    "AILACasedocs": "Court request:",
    "AILAStatutes": "Instruct: Match context with the best-fitting legal statute provisions
Query: ",
    "ChatDoctorRetrieval": "Clarify medically with comprehensive, effective responses",
    "DS1000Retrieval": "For computational tasks, retrieve scripts using scipy and numpy",
    "FinQARetrieval": "",  # baseline is better
    "FinanceBenchRetrieval": "Instruct: Gather pertinent financial insights using primary reports
Query: ",
    "FreshStackRetrieval": "",  # baseline is better
    "HC3FinanceRetrieval": "Seek out finance Q&A documents answering the question completely",
    "HumanEvalRetrieval": "Return the easiest Python function solving the problem",
    "LegalQuAD": "Surface directly applicable legal content",
    "LegalSummarization": "Collect legal documents ensuring broad summarization coverage
Query: ",
    "WikiSQLRetrieval": "Uncover SQL query precisely tailored to inquiry",
}
```

## Generation 9 Results (2026-01-20 04:26)

| Dataset | Baseline | Best Score | Improvement | Best Prompt |
|---------|----------|------------|-------------|-------------|
| AILACasedocs | 47.49% | 47.79% | +0.30% (+0.6%) | Court request: |
| AILAStatutes | 54.35% | 56.26% | +1.91% (+3.5%) | Instruct: Match context with the best-fi... |
| ChatDoctorRetrieval | 77.20% | 77.39% | +0.19% (+0.2%) | Clarify medically with comprehensive, ef... |
| DS1000Retrieval | 71.19% | 71.28% | +0.09% (+0.1%) | Instruct: Efficiently fetch Python data ... |
| FinQARetrieval | 88.99% | 88.99% | -0.00% (-0.0%) | nan |
| FinanceBenchRetrieval | 93.16% | 93.95% | +0.79% (+0.8%) | Gather directly applicable financial dat... |
| FreshStackRetrieval | 50.79% | 50.79% | +0.00% (+0.0%) | Tap into Stack Overflow for resolution p... |
| HC3FinanceRetrieval | 76.83% | 78.73% | +1.90% (+2.5%) | Find finance Q&A providing complete and ... |
| HumanEvalRetrieval | 99.36% | 100.00% | +0.64% (+0.6%) | Return the easiest Python function solvi... |
| LegalQuAD | 74.63% | 75.80% | +1.17% (+1.6%) | Surface directly applicable legal conten... |
| LegalSummarization | 78.09% | 79.29% | +1.20% (+1.5%) | Collect legal documents ensuring broad s... |
| WikiSQLRetrieval | 96.70% | 99.04% | +2.34% (+2.4%) | Uncover SQL query precisely tailored to ... |

### Key Learnings

**Improved:** AILACasedocs, AILAStatutes, DS1000Retrieval, LegalQuAD, HC3FinanceRetrieval, ChatDoctorRetrieval, LegalSummarization, FinanceBenchRetrieval, WikiSQLRetrieval, HumanEvalRetrieval

**Hurt by prompts:** FinQARetrieval (baseline is better)

### Winning Prompts

```python
WINNING_PROMPTS = {
    "AILACasedocs": "Court request:",
    "AILAStatutes": "Instruct: Match context with the best-fitting legal statute provisions
Query: ",
    "ChatDoctorRetrieval": "Clarify medically with comprehensive, effective responses",
    "DS1000Retrieval": "Instruct: Efficiently fetch Python data scripts using numpy
Query: ",
    "FinQARetrieval": "",  # baseline is better
    "FinanceBenchRetrieval": "Gather directly applicable financial data from key SEC documents
Query: ",
    "FreshStackRetrieval": "",  # baseline is better
    "HC3FinanceRetrieval": "Find finance Q&A providing complete and relevant answers
Query: ",
    "HumanEvalRetrieval": "Return the easiest Python function solving the problem",
    "LegalQuAD": "Surface directly applicable legal content",
    "LegalSummarization": "Collect legal documents ensuring broad summarization coverage
Query: ",
    "WikiSQLRetrieval": "Uncover SQL query precisely tailored to inquiry",
}
```
