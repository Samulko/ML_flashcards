#!/usr/bin/env python3
"""
Test script to demonstrate the new card template structure
"""

# Simple test without dependencies
def parse_extra_content(extra_text):
    """Parse structured extra content into separate fields"""
    if not extra_text or not extra_text.strip():
        return {}
    
    import re
    
    # Initialize fields
    fields = {
        'analogy': '',
        'key_insight': '',
        'technical': '',
        'connections': '',
        'practical': ''
    }
    
    # Extract analogy
    analogy_match = re.search(r'ANALOGY:([^\n]*(?:\n(?!\w+:)[^\n]*)*)', extra_text, re.IGNORECASE | re.MULTILINE)
    if analogy_match:
        fields['analogy'] = analogy_match.group(1).strip()
    
    # Extract key insight
    insight_patterns = ['KEY INSIGHT:', 'INTUITION:', 'SIMPLE INTUITION:', 'INTUITIVE MEANING:', 'KEY CONCEPT:']
    for pattern in insight_patterns:
        insight_match = re.search(rf'{pattern}([^\n]*(?:\n(?!\w+:)[^\n]*)*)', extra_text, re.IGNORECASE | re.MULTILINE)
        if insight_match:
            fields['key_insight'] = insight_match.group(1).strip()
            break
    
    # Extract technical details
    tech_patterns = ['TECHNICAL:', 'TECHNICAL NOTES:', 'MATHEMATICAL:', 'COMPUTATIONAL:', 'ALGORITHM:', 'PROCESS:']
    tech_parts = []
    for pattern in tech_patterns:
        tech_match = re.search(rf'{pattern}([^\n]*(?:\n(?!\w+:)[^\n]*)*)', extra_text, re.IGNORECASE | re.MULTILINE)
        if tech_match:
            tech_parts.append(tech_match.group(1).strip())
    if tech_parts:
        fields['technical'] = ' | '.join(tech_parts)
    
    # Extract connections
    conn_match = re.search(r'CONNECTIONS?:([^\n]*(?:\n(?!\w+:)[^\n]*)*)', extra_text, re.IGNORECASE | re.MULTILINE)
    if conn_match:
        fields['connections'] = conn_match.group(1).strip()
    
    # Extract practical applications
    practical_patterns = ['PRACTICAL:', 'APPLICATIONS?:', 'REAL-WORLD EXAMPLES?:', 'WHEN TO USE:', 'USAGE:']
    practical_parts = []
    for pattern in practical_patterns:
        practical_match = re.search(rf'{pattern}([^\n]*(?:\n(?!\w+:)[^\n]*)*)', extra_text, re.IGNORECASE | re.MULTILINE)
        if practical_match:
            practical_parts.append(practical_match.group(1).strip())
    if practical_parts:
        fields['practical'] = ' | '.join(practical_parts)
    
    return fields

# Test with sample data
if __name__ == "__main__":
    sample_extra = '''ANALOGY: Like finding the best camera angles to photograph a 3D sculpture - you want views that capture the most detail with fewest shots.

KEY INSIGHT: High-dimensional data often lies on lower-dimensional manifolds - PCA finds linear approximations of these manifolds.

TECHNICAL NOTES:
• X must be mean-centered first!
• Formula assumes X is (n×p) with rows=samples, cols=features

CONNECTIONS:
• Related to SVD (Singular Value Decomposition) and eigendecomposition
• Used before clustering (curse of dimensionality)

PRACTICAL: Essential for avoiding curse of dimensionality in high-dim spaces.'''

    print("🔍 Testing the new card template parsing...")
    print("=" * 60)
    
    parsed = parse_extra_content(sample_extra)
    
    print("📊 Parsed fields:")
    for key, value in parsed.items():
        if value:
            print(f"   {key.upper()}: {value}")
        else:
            print(f"   {key.upper()}: (empty)")
    
    print("\n✨ Card will now display these as separate, color-coded sections!")