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

## Customization

Add new cards in `generate_cards.py`:

```python
{
    'front': 'Question here',
    'back': 'Answer here', 
    'formula': '\\[Math formula\\]',
    'tags': 'topic-name',
    'extra': 'Additional context'
}
```

## Requirements

- Python 3.7+
- Anki Desktop 2.1+
- Dependencies: `genanki`, `nbformat`

## Troubleshooting

**Math not rendering:** Enable MathJax in Anki Preferences → Review

**Import errors:** Verify .apkg file exists in `generated_decks/`

**Missing content:** Check `temp_data/all_cards.csv` for generated cards