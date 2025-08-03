# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This directory is for creating Anki flash cards for the Foundations of Machine Learning course. The parent repository contains completed assignments covering:
- PCA and preprocessing
- K-means and EM clustering
- Linear regression and feature analysis
- Ridge regression and classification
- Bayesian methods and breast cancer classification
- Logistic regression and Bayes error

## Common Commands

```bash
# Install dependencies
pip install -r ../requirements.txt

# Run Jupyter notebooks
jupyter notebook
```

## Structure and Context

The parent directory contains completed ML assignments that serve as source material for creating flash cards. Key topics covered include:

1. **Preprocessing & PCA** (assignment_1)
2. **Clustering Methods** (assignment_2): PCA evaluation, K-means, EM algorithms
3. **Linear Regression** (assignment_3): Correlation and feature analysis
4. **Ridge Regression** (assignment_4): Regularization and classification
5. **Bayesian Methods** (assignment_5): Naive Bayes on breast cancer data
6. **Logistic Regression** (assignment_6): Multi-class classification and Bayes error

## Development Notes

- The project uses Jupyter notebooks (.ipynb) extensively
- Minimal dependencies: numpy and jupyter
- Flash cards should extract key concepts, formulas, and implementation details from the completed assignments

## Anki Flash Card Creation Research

### Recommended Approach: CSV + genanki Library

**Primary Method**: Create CSV files for easy editing and use Python genanki library for .apkg generation

**CSV Structure for Anki Import**:
```csv
Front,Back,Tags
"What is PCA?","Principal Component Analysis - dimensionality reduction technique","ML PCA"
"Linear regression formula","[latex]y = \\beta_0 + \\beta_1 x + \\epsilon[/latex]","ML Linear-Regression"
```

**Dependencies to Add**:
```bash
pip install genanki
pip install nbformat  # For parsing Jupyter notebooks
```

### Card Types for ML Content

1. **Concept Cards**: Definition-based questions
2. **Formula Cards**: Mathematical expressions with LaTeX
3. **Code Cards**: Implementation snippets
4. **Algorithm Cards**: Step-by-step procedures
5. **Visualization Cards**: Chart/graph interpretations

### LaTeX Support

- Use `[latex][/latex]` tags for mathematical formulas
- Enable "Generate LaTeX images" in Anki preferences
- Example: `[latex]\\sigma^2 = \\frac{1}{n}\\sum_{i=1}^{n}(x_i - \\mu)^2[/latex]`

### Educational Best Practices

- **Spaced Repetition**: Cards reviewed at increasing intervals
- **Active Recall**: Question-answer format forces retrieval
- **Difficulty-Based**: Hard concepts appear more frequently
- **Atomic Cards**: One concept per card for better retention
- **Context**: Include enough context without over-explaining

### Technical Implementation Research

**Recommended Technology Stack:**
- **genanki**: Primary library for .apkg generation
- **nbformat**: Parse Jupyter notebooks programmatically  
- **PyMuPDF**: Extract content from PDF assignments
- **spaCy**: NLP for concept identification
- **AnkiConnect**: Real-time integration with running Anki

**CSV Structure with Validation:**
```python
# Production-ready CSV format
Front,Back,Source,Tags,Formula
"What is gradient descent?","Optimization algorithm minimizing cost function","Assignment 3","optimization algorithms","θ = θ - α∇J(θ)"
```

### Educational Content Analysis

**ML Content Patterns Identified:**
1. **Theory Components**: Mathematical formulas, concept definitions, algorithm descriptions
2. **Implementation**: Code examples, library usage, data processing patterns
3. **Evaluation**: Performance metrics, comparison studies, reflection questions

**Extraction Strategies:**
- **Pattern Recognition**: Use regex for task headers, formulas, algorithm descriptions
- **NLP Entity Recognition**: Identify ML terms, statistical concepts, performance metrics
- **Contextual Mapping**: Build prerequisite chains and cross-reference networks

### Pedagogical Best Practices (Evidence-Based)

**Spaced Repetition Optimization:**
- 1st review: 1 day, 2nd: 7 days, 3rd: 16 days, 4th: 35 days
- Personalize intervals based on concept difficulty
- Use active recall over passive recognition

**Effective Card Design for ML:**
- **Hierarchical structure**: Prerequisites → basic concepts → algorithms → implementation
- **Multiple representations**: Combine formulas, visualizations, and code
- **Cognitive load management**: One concept per card, progressive disclosure

### Automation Tools & Workflows

**AI-Powered Generation:**
- **T5/BART models**: Question-answer pair generation
- **Sentence-BERT**: Similarity detection for duplicates
- **Quality scoring**: Automated relevance assessment

**Complete Pipeline:**
1. **Source Processing**: nbformat (notebooks) + PyMuPDF (PDFs)
2. **Content Analysis**: spaCy NLP + regex patterns
3. **Card Generation**: AI models for Q&A pairs
4. **Quality Assurance**: Duplicate detection + validation
5. **Deck Creation**: genanki or AnkiConnect API

### Production Workflow

1. Parse Jupyter notebooks with `nbformat`
2. Extract key concepts, formulas, and code snippets using NLP
3. Generate validated CSV files with quality checks
4. Use genanki to create .apkg files for import
5. Tag cards by topic/assignment for spaced repetition optimization

## Implementation Requirements

### Dependencies
```bash
pip install genanki
pip install nbformat
```

### Card Structure Specification

**Optimized Model Fields:**
```python
fields = [
    'Front',           # Question/concept
    'Back',            # Answer/definition  
    'Formula',         # Mathematical expression (MathJax format)
    'Source',          # Assignment reference (e.g., "Assignment 3")
    'Tags',            # Topic tags (e.g., "PCA dimensionality-reduction")
    'Extra'            # Context/examples/code snippets
]
```

**Card Templates:**
1. **Basic Card**: Front → Back + Formula (if present)
2. **Reverse Card**: Back → Front (for bidirectional learning)
3. **Formula Card**: Concept → Formula + explanation

### MathJax Formula Format

**Use MathJax syntax (recommended for 2025):**
- Inline math: `\(formula\)`
- Display math: `\[formula\]`
- Example: `\[PCA: Cov(X) = \frac{1}{n}X^TX\]`

### Content Extraction Patterns

**Priority Topics (from TOPICS.md analysis):**
1. **Neural Networks** (36 pts) - Architecture, backprop, activation functions
2. **PCA** (New topic) - Covariance matrix, eigenvectors, explained variance
3. **Clustering** (New topic) - K-means, EM algorithm, GMM
4. **Regression** (27 pts) - Linear, Ridge, Logistic formulas and concepts

**Card Generation Rules:**
- **Atomic principle**: One concept per card
- **Context inclusion**: Enough background without over-explaining
- **Progressive difficulty**: Basic → intermediate → advanced
- **Cross-references**: Link related concepts via tags

### File Organization

```
flashCards/
├── generate_cards.py          # Main generator script
├── card_templates.py          # Anki model definitions
├── content_extractor.py       # Notebook parsing logic
├── generated_decks/           # Output .apkg files
│   ├── ML_Foundations.apkg
│   ├── PCA_Clustering.apkg
│   └── Neural_Networks.apkg
└── temp_data/                 # Intermediate CSV files
    ├── concepts.csv
    ├── formulas.csv
    └── algorithms.csv
```

### Quality Assurance

**Validation Checks:**
- Duplicate detection using content similarity
- Formula syntax validation for MathJax
- Source attribution verification
- Tag consistency enforcement
- Difficulty distribution analysis