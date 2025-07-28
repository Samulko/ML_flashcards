# ML Foundations Anki Flashcards

Generate 80+ curated Anki flashcards for ML exam preparation covering neural networks, PCA, clustering, regression, and evaluation metrics.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate flashcards
python generate_cards.py

# Import generated_decks/ML_Foundations_Exam_Prep.apkg into Anki
```

## Coverage

**High Priority Topics:**
- Neural Networks (36 pts): CNN layers, activation functions, backpropagation
- PCA & Clustering (30+ pts): Eigendecomposition, K-means, EM algorithm
- Regression (27 pts): Linear, Ridge, Logistic with evaluation metrics
- Decision Trees & SVM (31 pts): Entropy, information gain, margins
- Evaluation: ROC/AUC, confusion matrix, statistical significance

**Card Types:**
- Concept definitions with context
- Mathematical formulas (MathJax rendered)
- Algorithm steps and procedures
- Practical applications and limitations

## Project Structure

```
├── generate_cards.py          # Main generator
├── card_templates.py          # Anki styling
├── generated_decks/           # Output .apkg files
├── temp_data/                 # CSV export for review
└── requirements.txt
```

## Adding More Cards

### Method 1: Direct Addition

Edit `generate_cards.py` in the `_generate_comprehensive_cards()` method:

```python
# Add to the cards list around line 460
{
    'front': 'What is your new concept?',
    'back': 'Definition or explanation here',
    'formula': '\\[Mathematical\\_formula = here\\]',  # Optional
    'source': 'Your Source',
    'tags': 'topic-name subtopic',
    'extra': 'Additional context, examples, or mnemonics'
}
```

### Method 2: Topic-Based Expansion

Add new topic sections following existing patterns:

```python
# New topic section (e.g., around line 500)
# Reinforcement Learning (New Topic)
{
    'front': 'What is Q-learning?',
    'back': 'Model-free reinforcement learning algorithm that learns optimal action-value function',
    'formula': '\\[Q(s,a) = Q(s,a) + \\alpha[r + \\gamma \\max Q(s\',a\') - Q(s,a)]\\]',
    'source': 'RL Fundamentals',
    'tags': 'reinforcement-learning q-learning',
    'extra': 'Uses temporal difference learning to update Q-values'
},
```

### Method 3: CSV Import

1. Export current cards: Run generator to create `temp_data/all_cards.csv`
2. Edit CSV file with new cards following the format
3. Modify `generate_cards.py` to read from CSV if desired

### Card Design Guidelines

**Good Front Questions:**
- Start with "What is...", "How do...", "When should..."
- Be specific and unambiguous
- Focus on one concept per card

**Effective Back Answers:**
- Concise but complete
- Include key distinguishing features
- Mention practical applications

**Formula Formatting:**
- Use MathJax syntax: `\\[display\\]` or `\\(inline\\)`
- Escape underscores: `\\_` instead of `_`
- Common symbols: `\\alpha`, `\\beta`, `\\sum`, `\\frac{num}{den}`

**Tag Strategy:**
- Use consistent naming: `topic-subtopic`
- Include difficulty levels: `basic`, `intermediate`, `advanced`
- Add exam-specific tags: `high-priority`, `formula-heavy`

## Requirements

- Python 3.7+
- Anki Desktop 2.1+
- Dependencies: `genanki`, `nbformat`

## Troubleshooting

**Math not rendering:** Enable MathJax in Anki Preferences → Review

**Import errors:** Verify .apkg file exists in `generated_decks/`

**Missing content:** Check `temp_data/all_cards.csv` for generated cards