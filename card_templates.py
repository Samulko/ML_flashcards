"""
Anki card model templates for ML flashcards
Optimized for mathematical content with MathJax support
"""

import genanki

# Unique model ID (generated once, hardcoded for consistency)
ML_MODEL_ID = 1607392819

# CSS styling for cards
ML_CARD_CSS = """
.card {
    font-family: Arial, sans-serif;
    font-size: 16px;
    line-height: 1.5;
    color: #333 !important;
    background-color: #fff !important;
    padding: 20px;
}

/* Force all text to be dark */
.card * {
    color: #333 !important;
}

/* Override any Anki theme that might interfere */
body {
    color: #333 !important;
    background-color: #fff !important;
}

.front {
    text-align: center;
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 15px;
}

.back {
    margin-bottom: 15px;
    color: #333 !important;
}

.formula {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 4px;
    padding: 10px;
    margin: 10px 0;
    text-align: center;
    font-family: 'Times New Roman', serif;
    color: #000 !important;
}

.formula .MathJax {
    color: #000 !important;
}

.formula .MathJax_Display {
    color: #000 !important;
}

.source {
    font-size: 12px;
    color: #666;
    font-style: italic;
    margin-top: 15px;
}

.tags {
    font-size: 11px;
    color: #888;
    margin-top: 10px;
}

.extra {
    background-color: #f1f3f4;
    border-left: 4px solid #1976d2;
    padding: 10px;
    margin-top: 15px;
    font-size: 14px;
    color: #333 !important;
}

.cloze {
    font-weight: bold;
    color: #1976d2;
}
"""

# Main ML card model with multiple templates
ML_CARD_MODEL = genanki.Model(
    ML_MODEL_ID,
    'ML Foundations Card',
    fields=[
        {'name': 'Front'},     # Question/concept
        {'name': 'Back'},      # Answer/definition
        {'name': 'Formula'},   # Mathematical expression
        {'name': 'Source'},    # Assignment reference
        {'name': 'Tags'},      # Topic tags
        {'name': 'Extra'}      # Context/examples/code
    ],
    templates=[
        {
            'name': 'Basic Card',
            'qfmt': '''
                <div class="front">{{Front}}</div>
                {{#Formula}}
                <div class="formula">{{Formula}}</div>
                {{/Formula}}
            ''',
            'afmt': '''
                <div class="front">{{Front}}</div>
                {{#Formula}}
                <div class="formula">{{Formula}}</div>
                {{/Formula}}
                <hr id="answer">
                <div class="back">{{Back}}</div>
                {{#Extra}}
                <div class="extra">{{Extra}}</div>
                {{/Extra}}
                <div class="source">{{Source}}</div>
                <div class="tags">{{Tags}}</div>
            '''
        },
        {
            'name': 'Reverse Card',
            'qfmt': '''
                <div class="front">{{Back}}</div>
                {{#Extra}}
                <div class="extra">{{Extra}}</div>
                {{/Extra}}
            ''',
            'afmt': '''
                <div class="front">{{Back}}</div>
                {{#Extra}}
                <div class="extra">{{Extra}}</div>
                {{/Extra}}
                <hr id="answer">
                <div class="back">{{Front}}</div>
                {{#Formula}}
                <div class="formula">{{Formula}}</div>
                {{/Formula}}
                <div class="source">{{Source}}</div>
                <div class="tags">{{Tags}}</div>
            '''
        }
    ],
    css=ML_CARD_CSS
)

# Formula-focused model for mathematical concepts
FORMULA_MODEL_ID = 1607392820

FORMULA_CARD_MODEL = genanki.Model(
    FORMULA_MODEL_ID,
    'ML Formula Card',
    fields=[
        {'name': 'Concept'},   # Mathematical concept name
        {'name': 'Formula'},   # The formula itself
        {'name': 'Variables'}, # Variable definitions
        {'name': 'Source'},    # Assignment reference
        {'name': 'Tags'},      # Topic tags
        {'name': 'Example'}    # Worked example
    ],
    templates=[
        {
            'name': 'Formula Card',
            'qfmt': '''
                <div class="front">What is the formula for: {{Concept}}?</div>
                {{#Variables}}
                <div class="extra">Variables: {{Variables}}</div>
                {{/Variables}}
            ''',
            'afmt': '''
                <div class="front">{{Concept}}</div>
                <hr id="answer">
                <div class="formula">{{Formula}}</div>
                {{#Variables}}
                <div class="extra">Variables: {{Variables}}</div>
                {{/Variables}}
                {{#Example}}
                <div class="extra">Example: {{Example}}</div>
                {{/Example}}
                <div class="source">{{Source}}</div>
                <div class="tags">{{Tags}}</div>
            '''
        }
    ],
    css=ML_CARD_CSS
)

def create_ml_note(front, back, formula="", source="", tags="", extra=""):
    """Create a standard ML flashcard note"""
    # Handle empty fields (genanki requirement)
    fields = [
        front or " ",
        back or " ",
        formula or " ",
        source or " ",
        tags or " ",
        extra or " "
    ]
    
    return genanki.Note(
        model=ML_CARD_MODEL,
        fields=fields,
        guid=genanki.guid_for(front, back)  # Stable GUID for updates
    )

def create_formula_note(concept, formula, variables="", source="", tags="", example=""):
    """Create a formula-focused flashcard note"""
    fields = [
        concept or " ",
        formula or " ",
        variables or " ",
        source or " ",
        tags or " ",
        example or " "
    ]
    
    return genanki.Note(
        model=FORMULA_CARD_MODEL,
        fields=fields,
        guid=genanki.guid_for(concept, formula)
    )