"""
Content extraction from Jupyter notebooks for ML flashcard generation
Focuses on key concepts, formulas, and algorithms from completed assignments
"""

import json
import re
import os
from pathlib import Path
import nbformat

class MLContentExtractor:
    """Extract ML concepts and formulas from Jupyter notebooks"""
    
    def __init__(self, assignments_dir="../assignments"):
        self.assignments_dir = Path(assignments_dir)
        self.key_topics = {
            "assignment_1": ["PCA", "preprocessing", "dimensionality reduction"],
            "assignment_2": ["K-means", "EM", "clustering", "GMM"],
            "assignment_3": ["linear regression", "correlation", "feature analysis"],
            "assignment_4": ["ridge regression", "regularization", "classification"],
            "assignment_5": ["naive bayes", "breast cancer", "bayesian"],
            "assignment_6": ["logistic regression", "bayes error", "multi-class"]
        }
        
        # High-priority formulas from TOPICS.md
        self.priority_formulas = [
            # PCA
            r"Cov\(X\)|covariance.*matrix",
            r"PC\s*=\s*X.*v|principal.*component",
            
            # K-means
            r"min.*\|\|x.*μ\|\|²|objective.*function",
            r"μ.*=.*\(\s*1/\|.*\|\s*\)|centroid.*update",
            
            # EM Algorithm
            r"Q\(θ\|θ.*\)|expectation.*log.*likelihood",
            r"argmax.*Q\(θ\|θ.*\)|maximize.*expectation",
            
            # Neural Networks
            r"h\s*=\s*σ\(.*\)|forward.*pass",
            r"∂L/∂W.*∂L/∂h.*∂h/∂W|chain.*rule|backward.*pass",
            
            # Regression
            r"β̂.*=.*\(X.*X.*λI\).*X.*y|ridge.*solution",
            r"p.*=.*exp\(.*\)/.*exp\(.*\)|logistic.*softmax",
            
            # Decision Trees
            r"H\(S\).*-.*p.*log.*p|entropy",
            r"IG\(S,A\).*H\(S\)|information.*gain",
            
            # SVM
            r"min.*½\|\|w\|\|².*C.*ξ|primal.*objective",
            r"y.*\(w.*x.*b\).*≥.*1.*ξ|constraint",
            
            # Evaluation
            r"TP.*\(TP.*FP\)|precision",
            r"TP.*\(TP.*FN\)|recall",
            r"2.*\(Precision.*Recall\)|F1.*score"
        ]
        
    def extract_from_notebook(self, notebook_path):
        """Extract content from a single Jupyter notebook"""
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            content = {
                'concepts': [],
                'formulas': [],
                'algorithms': [],
                'source': str(notebook_path.name)
            }
            
            for cell in nb.cells:
                if cell.cell_type == 'markdown':
                    self._extract_from_markdown(cell.source, content)
                elif cell.cell_type == 'code':
                    self._extract_from_code(cell.source, content)
                    
            return content
            
        except Exception as e:
            print(f"Error processing {notebook_path}: {e}")
            return None
    
    def _extract_from_markdown(self, text, content):
        """Extract concepts and formulas from markdown cells"""
        
        # Extract headers as potential concepts
        headers = re.findall(r'^#+\s*(.+)$', text, re.MULTILINE)
        for header in headers:
            if self._is_ml_concept(header):
                content['concepts'].append({
                    'title': header.strip(),
                    'type': 'concept',
                    'content': self._extract_definition_after_header(text, header)
                })
        
        # Extract mathematical formulas
        # LaTeX math environments
        latex_formulas = re.findall(r'\$\$([^$]+)\$\$|\$([^$]+)\$', text)
        for formula_match in latex_formulas:
            formula = formula_match[0] or formula_match[1]
            if self._is_priority_formula(formula):
                content['formulas'].append({
                    'formula': formula.strip(),
                    'context': self._get_surrounding_context(text, formula)
                })
        
        # Extract algorithmic descriptions
        algorithms = re.findall(r'(?:Algorithm|Steps?|Procedure):\s*\n((?:(?:\d+\.|\*|\-).+\n?)+)', text, re.IGNORECASE)
        for algo in algorithms:
            content['algorithms'].append({
                'steps': algo.strip(),
                'type': 'algorithm'
            })
    
    def _extract_from_code(self, code, content):
        """Extract key concepts from code cells"""
        
        # Look for important function definitions
        functions = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\):', code)
        for func in functions:
            if self._is_ml_function(func):
                content['concepts'].append({
                    'title': f"Function: {func}",
                    'type': 'implementation',
                    'content': self._extract_function_docstring(code, func)
                })
        
        # Extract comments with mathematical content
        comments = re.findall(r'#\s*(.+)', code)
        for comment in comments:
            if any(term in comment.lower() for term in ['formula', 'equation', 'calculate']):
                content['concepts'].append({
                    'title': comment.strip(),
                    'type': 'implementation_note',
                    'content': comment.strip()
                })
    
    def _is_ml_concept(self, text):
        """Check if text represents an ML concept"""
        ml_keywords = [
            'pca', 'principal component', 'kmeans', 'k-means', 'clustering',
            'regression', 'classification', 'neural network', 'gradient',
            'bayes', 'naive bayes', 'svm', 'support vector', 'decision tree',
            'random forest', 'cross validation', 'overfitting', 'underfitting',
            'bias', 'variance', 'regularization', 'logistic', 'sigmoid',
            'softmax', 'backpropagation', 'forward pass', 'activation',
            'loss function', 'cost function', 'optimization', 'convergence'
        ]
        
        return any(keyword in text.lower() for keyword in ml_keywords)
    
    def _is_priority_formula(self, formula):
        """Check if formula matches priority patterns"""
        return any(re.search(pattern, formula, re.IGNORECASE) for pattern in self.priority_formulas)
    
    def _is_ml_function(self, func_name):
        """Check if function name suggests ML implementation"""
        ml_func_patterns = [
            r'.*pca.*', r'.*kmeans.*', r'.*cluster.*', r'.*regression.*',
            r'.*classify.*', r'.*predict.*', r'.*fit.*', r'.*transform.*',
            r'.*train.*', r'.*test.*', r'.*evaluate.*', r'.*score.*'
        ]
        
        return any(re.match(pattern, func_name, re.IGNORECASE) for pattern in ml_func_patterns)
    
    def _extract_definition_after_header(self, text, header):
        """Extract definition text that follows a header"""
        lines = text.split('\n')
        header_found = False
        definition_lines = []
        
        for line in lines:
            if header in line and line.startswith('#'):
                header_found = True
                continue
            elif header_found:
                if line.startswith('#'):  # Next header
                    break
                elif line.strip():  # Non-empty line
                    definition_lines.append(line.strip())
                elif definition_lines:  # Empty line after content
                    break
        
        return ' '.join(definition_lines[:3])  # First 3 sentences
    
    def _get_surrounding_context(self, text, formula):
        """Get context around a formula"""
        # Find the sentence containing the formula
        sentences = text.split('.')
        for sentence in sentences:
            if formula in sentence:
                return sentence.strip()
        return ""
    
    def _extract_function_docstring(self, code, func_name):
        """Extract docstring from function definition"""
        func_pattern = rf'def\s+{func_name}\s*\([^)]*\):\s*(?:"""([^"]*)"""|\'\'\'([^\']*)\'\'\')?'
        match = re.search(func_pattern, code, re.DOTALL)
        if match:
            return (match.group(1) or match.group(2) or "").strip()
        return ""
    
    def extract_all_assignments(self):
        """Extract content from all assignment directories"""
        all_content = {}
        
        if not self.assignments_dir.exists():
            print(f"Assignments directory not found: {self.assignments_dir}")
            return all_content
        
        for assignment_dir in self.assignments_dir.iterdir():
            if assignment_dir.is_dir() and assignment_dir.name.startswith('assignment_'):
                print(f"Processing {assignment_dir.name}...")
                assignment_content = self._extract_from_assignment(assignment_dir)
                if assignment_content:
                    all_content[assignment_dir.name] = assignment_content
        
        return all_content
    
    def _extract_from_assignment(self, assignment_dir):
        """Extract content from all notebooks in an assignment directory"""
        assignment_content = {
            'concepts': [],
            'formulas': [],
            'algorithms': []
        }
        
        for notebook_path in assignment_dir.glob('*.ipynb'):
            print(f"  - Processing {notebook_path.name}")
            nb_content = self.extract_from_notebook(notebook_path)
            if nb_content:
                # Add source information and merge content
                for category in ['concepts', 'formulas', 'algorithms']:
                    for item in nb_content[category]:
                        item['source'] = f"{assignment_dir.name}/{notebook_path.name}"
                        assignment_content[category].append(item)
        
        return assignment_content