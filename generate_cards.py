"""
Main script to generate Anki flashcards from ML assignments
Creates .apkg files ready for import into Anki
"""

import os
import csv
from pathlib import Path
import genanki
from card_templates import ML_CARD_MODEL, FORMULA_CARD_MODEL, create_ml_note, create_formula_note
from content_extractor import MLContentExtractor

class MLFlashcardGenerator:
    """Generate Anki flashcards from extracted ML content"""
    
    def __init__(self, output_dir="generated_decks", temp_dir="temp_data"):
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Single deck ID (generated once, hardcoded for consistency)
        self.deck_id = 1607392821
        
        # All cards will go into single deck
        self.deck_name = "ML Foundations Exam Prep"
        
    def generate_all_cards(self):
        """Main entry point - generate all flashcard decks"""
        print("🚀 Starting ML flashcard generation...")
        
        # Focus on manually curated high-quality cards
        print("🎯 Generating high-quality curated cards...")
        
        # Generate comprehensive manual cards based on exam topics
        manual_cards = self._generate_comprehensive_cards()
        print(f"✅ Generated {len(manual_cards)} curated cards")
        
        # Combine all cards into single deck
        all_cards = manual_cards
        
        # Create and save single Anki deck
        self._create_single_anki_deck(all_cards)
        
        # Save intermediate CSV files for review
        self._save_csv_files(all_cards)
        
        print(f"🎉 Generated {len(all_cards)} total cards in single deck")
        print(f"📁 Output files saved to: {self.output_dir}")
        
    
    def _generate_comprehensive_cards(self):
        """Generate comprehensive, high-quality cards based on exam topics"""
        cards = [
            # PCA (High Priority - New Topic)
            {
                'front': 'What is Principal Component Analysis (PCA)?',
                'back': 'Dimensionality reduction technique that finds linear combinations of features with maximum variance',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'PCA dimensionality-reduction',
                'extra': 'Used to reduce feature space while preserving most information'
            },
            {
                'front': 'What is the covariance matrix formula in PCA?',
                'back': 'Matrix representing relationships between features',
                'formula': '\\[Cov(X) = \\frac{1}{n}X^TX\\]',
                'source': 'TOPICS.md',
                'tags': 'PCA covariance-matrix formula',
                'extra': 'Used to find principal components via eigendecomposition'
            },
            {
                'front': 'How are principal components calculated?',
                'back': 'Linear combinations of original features using eigenvectors',
                'formula': '\\[PC = X \\cdot v\\]',
                'source': 'TOPICS.md',
                'tags': 'PCA principal-components',
                'extra': 'v are eigenvectors of the covariance matrix, ordered by eigenvalue magnitude'
            },
            {
                'front': 'What does explained variance tell us in PCA?',
                'back': 'Proportion of dataset variance captured by each principal component',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'PCA explained-variance',
                'extra': 'Used to decide how many components to keep'
            },
            
            # K-means Clustering (High Priority - New Topic)
            {
                'front': 'What is K-means clustering?',
                'back': 'Partitioning algorithm that divides data into k clusters by minimizing within-cluster variance',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'clustering k-means',
                'extra': 'Assumes spherical clusters of similar size'
            },
            {
                'front': 'What is the K-means objective function?',
                'back': 'Minimize within-cluster sum of squared distances',
                'formula': '\\[\\min \\sum_i \\sum_{x \\in C_i} ||x - \\mu_i||^2\\]',
                'source': 'TOPICS.md',
                'tags': 'k-means objective-function',
                'extra': 'Also called within-cluster sum of squares (WCSS) or inertia'
            },
            {
                'front': 'How do you update centroids in K-means?',
                'back': 'Take the mean of all points assigned to each cluster',
                'formula': '\\[\\mu_i = \\frac{1}{|C_i|} \\sum_{x \\in C_i} x\\]',
                'source': 'TOPICS.md',
                'tags': 'k-means centroid-update',
                'extra': 'This is the M-step in Lloyd\'s algorithm'
            },
            {
                'front': 'What are the two steps of Lloyd\'s K-means algorithm?',
                'back': 'Assignment step (assign points to nearest centroid) and Update step (recalculate centroids)',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'k-means lloyds-algorithm',
                'extra': 'Alternates between these steps until convergence'
            },
            
            # EM Algorithm (High Priority - New Topic)
            {
                'front': 'What is the EM algorithm?',
                'back': 'Expectation-Maximization algorithm for finding parameters when latent variables exist',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'EM algorithm latent-variables',
                'extra': 'Commonly used for Gaussian Mixture Models'
            },
            {
                'front': 'What is the E-step in EM algorithm?',
                'back': 'Calculate expectation of log-likelihood given current parameters',
                'formula': '\\[Q(θ|θ^{(t)}) = E[\\log L(θ|X,Z)|X,θ^{(t)}]\\]',
                'source': 'TOPICS.md',
                'tags': 'EM e-step expectation',
                'extra': 'Estimates probability of latent variable assignments'
            },
            {
                'front': 'What is the M-step in EM algorithm?',
                'back': 'Find parameters that maximize the expectation from E-step',
                'formula': '\\[θ^{(t+1)} = \\arg\\max Q(θ|θ^{(t)})\\]',
                'source': 'TOPICS.md',
                'tags': 'EM m-step maximization',
                'extra': 'Updates model parameters based on expected latent assignments'
            },
            
            # Neural Networks (Highest Priority - 36 points)
            {
                'front': 'What is a neural network?',
                'back': 'Computational model inspired by biological neurons, with layers of interconnected nodes',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'neural-networks definition',
                'extra': 'Universal function approximators capable of learning complex patterns'
            },
            {
                'front': 'What is the forward pass in neural networks?',
                'back': 'Process of computing output by propagating input through network layers',
                'formula': '\\[h = \\sigma(Wx + b)\\]',
                'source': 'TOPICS.md',
                'tags': 'neural-networks forward-pass',
                'extra': 'σ is activation function (ReLU, sigmoid, tanh), W is weights, b is bias'
            },
            {
                'front': 'What is backpropagation?',
                'back': 'Algorithm for computing gradients of loss function with respect to network weights',
                'formula': '\\[\\frac{\\partial L}{\\partial W} = \\frac{\\partial L}{\\partial h} \\cdot \\frac{\\partial h}{\\partial W}\\]',
                'source': 'TOPICS.md',
                'tags': 'neural-networks backpropagation',
                'extra': 'Uses chain rule to propagate error backwards through network'
            },
            {
                'front': 'What is the ReLU activation function?',
                'back': 'Rectified Linear Unit: outputs input if positive, zero otherwise',
                'formula': '\\[ReLU(x) = \\max(0, x)\\]',
                'source': 'ML Fundamentals',
                'tags': 'neural-networks activation-function relu',
                'extra': 'Most popular activation function, helps with vanishing gradient problem'
            },
            {
                'front': 'What is gradient descent?',
                'back': 'Optimization algorithm that iteratively updates parameters in direction of steepest descent',
                'formula': '\\[θ = θ - α∇J(θ)\\]',
                'source': 'ML Fundamentals',
                'tags': 'optimization gradient-descent',
                'extra': 'α is learning rate, ∇J(θ) is gradient of cost function'
            },
            
            # Linear Regression (High Priority - 27 points)
            {
                'front': 'What is linear regression?',
                'back': 'Statistical method for modeling relationship between dependent variable and independent variables',
                'formula': '\\[y = β_0 + β_1x_1 + β_2x_2 + ... + ε\\]',
                'source': 'ML Fundamentals',
                'tags': 'linear-regression supervised-learning',
                'extra': 'Assumes linear relationship between features and target'
            },
            {
                'front': 'What is the normal equation for linear regression?',
                'back': 'Closed-form solution for optimal parameters',
                'formula': '\\[β = (X^TX)^{-1}X^Ty\\]',
                'source': 'ML Fundamentals',
                'tags': 'linear-regression normal-equation',
                'extra': 'Minimizes least squares error analytically'
            },
            {
                'front': 'What is Ridge regression?',
                'back': 'Linear regression with L2 regularization to prevent overfitting',
                'formula': '\\[\\hat{β} = (X^TX + λI)^{-1}X^Ty\\]',
                'source': 'TOPICS.md',
                'tags': 'ridge-regression regularization',
                'extra': 'λ is regularization parameter; higher λ means more regularization'
            },
            {
                'front': 'What is the difference between Ridge and Lasso regression?',
                'back': 'Ridge uses L2 penalty (squared weights), Lasso uses L1 penalty (absolute weights)',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'regularization ridge lasso',
                'extra': 'Lasso can shrink coefficients to zero (feature selection), Ridge cannot'
            },
            
            # Logistic Regression (High Priority)
            {
                'front': 'What is logistic regression?',
                'back': 'Classification algorithm using logistic function to model probability of binary outcomes',
                'formula': '\\[p = \\frac{1}{1 + e^{-z}}\\]',
                'source': 'ML Fundamentals',
                'tags': 'logistic-regression classification',
                'extra': 'z = β₀ + β₁x₁ + β₂x₂ + ..., outputs probability between 0 and 1'
            },
            {
                'front': 'What is the softmax function?',
                'back': 'Generalization of logistic function for multi-class classification',
                'formula': '\\[p_i = \\frac{\\exp(w_i^T x)}{\\sum_j \\exp(w_j^T x)}\\]',
                'source': 'TOPICS.md',
                'tags': 'logistic-regression softmax multi-class',
                'extra': 'Outputs probability distribution over all classes'
            },
            
            # Decision Trees (19 points)
            {
                'front': 'What is entropy in decision trees?',
                'back': 'Measure of impurity or randomness in a dataset',
                'formula': '\\[H(S) = -\\sum p_i \\log_2(p_i)\\]',
                'source': 'TOPICS.md',
                'tags': 'decision-trees entropy',
                'extra': 'Lower entropy means more homogeneous (pure) dataset'
            },
            {
                'front': 'What is information gain?',
                'back': 'Reduction in entropy after splitting on an attribute',
                'formula': '\\[IG(S,A) = H(S) - \\sum \\frac{|S_v|}{|S|} H(S_v)\\]',
                'source': 'TOPICS.md',
                'tags': 'decision-trees information-gain',
                'extra': 'Used to select best attribute for splitting at each node'
            },
            {
                'front': 'What is Gini impurity?',
                'back': 'Alternative to entropy for measuring node impurity',
                'formula': '\\[Gini = 1 - \\sum p_i^2\\]',
                'source': 'ML Fundamentals',
                'tags': 'decision-trees gini-impurity',
                'extra': 'Computationally faster than entropy, similar results'
            },
            {
                'front': 'What is Random Forest?',
                'back': 'Ensemble method combining multiple decision trees with bagging',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'ensemble random-forest decision-trees',
                'extra': 'Reduces variance and overfitting compared to single decision tree'
            },
            
            # SVM (12 points)
            {
                'front': 'What is Support Vector Machine (SVM)?',
                'back': 'Classification algorithm that finds optimal hyperplane maximizing margin between classes',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'SVM classification margin',
                'extra': 'Support vectors are data points closest to decision boundary'
            },
            {
                'front': 'What is the SVM primal objective function?',
                'back': 'Minimize weights while allowing some misclassification',
                'formula': '\\[\\min \\frac{1}{2}||w||^2 + C\\sum ξ_i\\]',
                'source': 'TOPICS.md',
                'tags': 'SVM primal objective',
                'extra': 'C controls trade-off between margin size and misclassification penalty'
            },
            {
                'front': 'What is the SVM constraint?',
                'back': 'Points must be on correct side of margin or pay penalty',
                'formula': '\\[y_i(w^Tx_i + b) ≥ 1 - ξ_i\\]',
                'source': 'TOPICS.md',
                'tags': 'SVM constraint slack-variables',
                'extra': 'ξᵢ are slack variables allowing soft margin'
            },
            {
                'front': 'What is the kernel trick in SVM?',
                'back': 'Technique to implicitly map data to higher-dimensional space for non-linear classification',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'SVM kernel-trick',
                'extra': 'Common kernels: linear, polynomial, RBF (Gaussian)'
            },
            
            # Naive Bayes (13 points)
            {
                'front': 'What is Naive Bayes classifier?',
                'back': 'Probabilistic classifier based on Bayes theorem with strong independence assumption',
                'formula': '\\[P(C|X) = \\frac{P(X|C)P(C)}{P(X)}\\]',
                'source': 'ML Fundamentals',
                'tags': 'naive-bayes classification bayes-theorem',
                'extra': 'Assumes features are conditionally independent given class'
            },
            {
                'front': 'What is the naive independence assumption?',
                'back': 'Features are conditionally independent given the class label',
                'formula': '\\[P(x_1,...,x_n|C) = \\prod P(x_i|C)\\]',
                'source': 'ML Fundamentals',
                'tags': 'naive-bayes independence-assumption',
                'extra': 'Simplifies computation but often violated in practice'
            },
            {
                'front': 'What is Laplace smoothing?',
                'back': 'Technique to handle zero probabilities by adding small constant to counts',
                'formula': '\\[P(x_i|C) = \\frac{count(x_i,C) + α}{count(C) + α|V|}\\]',
                'source': 'ML Fundamentals',
                'tags': 'naive-bayes laplace-smoothing',
                'extra': 'α is smoothing parameter (usually 1), |V| is vocabulary size'
            },
            
            # Bias-Variance Tradeoff (15 points)
            {
                'front': 'What is the bias-variance tradeoff?',
                'back': 'Fundamental tradeoff between model complexity and generalization ability',
                'formula': '\\[Error = Bias^2 + Variance + Noise\\]',
                'source': 'ML Fundamentals',
                'tags': 'bias-variance tradeoff',
                'extra': 'High bias = underfitting, high variance = overfitting'
            },
            {
                'front': 'What is bias in machine learning?',
                'back': 'Error from overly simplistic assumptions in learning algorithm',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'bias underfitting',
                'extra': 'High bias leads to underfitting and poor performance on training data'
            },
            {
                'front': 'What is variance in machine learning?',
                'back': 'Error from sensitivity to small fluctuations in training set',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'variance overfitting',
                'extra': 'High variance leads to overfitting and poor generalization'
            },
            {
                'front': 'What is Bayes error?',
                'back': 'Lowest possible error rate for any classifier on a given problem',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'bayes-error irreducible-error',
                'extra': 'Represents irreducible error due to noise and overlapping classes'
            },
            
            # Evaluation Metrics
            {
                'front': 'What is accuracy?',
                'back': 'Fraction of correct predictions out of total predictions',
                'formula': '\\[Accuracy = \\frac{TP + TN}{TP + TN + FP + FN}\\]',
                'source': 'ML Fundamentals',
                'tags': 'evaluation accuracy',
                'extra': 'Can be misleading with imbalanced datasets'
            },
            {
                'front': 'What is precision?',
                'back': 'Fraction of true positives among predicted positives',
                'formula': '\\[Precision = \\frac{TP}{TP + FP}\\]',
                'source': 'TOPICS.md',
                'tags': 'evaluation precision',
                'extra': 'Answers: Of all positive predictions, how many were correct?'
            },
            {
                'front': 'What is recall (sensitivity)?',
                'back': 'Fraction of true positives among actual positives',
                'formula': '\\[Recall = \\frac{TP}{TP + FN}\\]',
                'source': 'TOPICS.md',
                'tags': 'evaluation recall sensitivity',
                'extra': 'Answers: Of all actual positives, how many were found?'
            },
            {
                'front': 'What is F1-score?',
                'back': 'Harmonic mean of precision and recall',
                'formula': '\\[F1 = \\frac{2 \\times Precision \\times Recall}{Precision + Recall}\\]',
                'source': 'TOPICS.md',
                'tags': 'evaluation f1-score',
                'extra': 'Balances precision and recall, useful for imbalanced datasets'
            },
            {
                'front': 'What is specificity?',
                'back': 'Fraction of true negatives among actual negatives',
                'formula': '\\[Specificity = \\frac{TN}{TN + FP}\\]',
                'source': 'ML Fundamentals',
                'tags': 'evaluation specificity',
                'extra': 'Answers: Of all actual negatives, how many were correctly identified?'
            },
            
            # CRISP-DM and Data Preprocessing
            {
                'front': 'What are the 6 phases of CRISP-DM?',
                'back': 'Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'CRISP-DM methodology',
                'extra': 'Iterative process for data mining projects'
            },
            {
                'front': 'What is data normalization?',
                'back': 'Scaling features to have similar ranges, typically [0,1]',
                'formula': '\\[x_{norm} = \\frac{x - x_{min}}{x_{max} - x_{min}}\\]',
                'source': 'ML Fundamentals',
                'tags': 'preprocessing normalization',
                'extra': 'Prevents features with large scales from dominating'
            },
            {
                'front': 'What is data standardization?',
                'back': 'Scaling features to have zero mean and unit variance',
                'formula': '\\[x_{std} = \\frac{x - μ}{σ}\\]',
                'source': 'ML Fundamentals',
                'tags': 'preprocessing standardization z-score',
                'extra': 'Results in standard normal distribution (mean=0, std=1)'
            },
            {
                'front': 'What is cross-validation?',
                'back': 'Technique for assessing model performance by splitting data into multiple train/validation sets',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'evaluation cross-validation',
                'extra': 'K-fold CV divides data into k subsets, trains k times'
            },
            {
                'front': 'What is overfitting?',
                'back': 'Model performs well on training data but poorly on unseen data',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'overfitting generalization',
                'extra': 'Model memorizes training data instead of learning patterns'
            },
            {
                'front': 'What is underfitting?',
                'back': 'Model is too simple to capture underlying patterns in data',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'underfitting bias',
                'extra': 'Results in poor performance on both training and test data'
            },
            
            # CNN Concepts (CRITICAL - Agent 1 Priority)
            {
                'front': 'What is a convolutional layer in CNNs?',
                'back': 'Layer that applies filters/kernels to detect local features while preserving spatial relationships',
                'formula': '',
                'source': 'Neural Networks',
                'tags': 'CNN convolutional-layer',
                'extra': 'Uses shared weights and local connectivity to reduce parameters'
            },
            {
                'front': 'What is pooling in CNNs?',
                'back': 'Downsampling operation that reduces spatial dimensions while retaining important features',
                'formula': '',
                'source': 'Neural Networks', 
                'tags': 'CNN pooling',
                'extra': 'Max pooling takes maximum value, average pooling takes mean value in each region'
            },
            {
                'front': 'What are CNN filters/kernels?',
                'back': 'Small matrices that slide over input to detect specific features like edges, textures, or patterns',
                'formula': '',
                'source': 'Neural Networks',
                'tags': 'CNN filters kernels',
                'extra': 'Each filter learns to detect different features through backpropagation'
            },
            {
                'front': 'What is stride in convolution?',
                'back': 'Step size when moving the filter across the input - larger stride means smaller output',
                'formula': '\\[Output\\_size = \\frac{Input\\_size - Filter\\_size + 2*Padding}{Stride} + 1\\]',
                'source': 'Neural Networks',
                'tags': 'CNN stride convolution',
                'extra': 'Stride of 1 preserves most spatial information, stride >1 reduces output size'
            },
            
            # Loss Functions (CRITICAL - Agent 1 Priority)
            {
                'front': 'What is Mean Squared Error (MSE)?',
                'back': 'Loss function measuring average squared differences between predicted and actual values',
                'formula': '\\[MSE = \\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y_i})^2\\]',
                'source': 'ML Fundamentals',
                'tags': 'loss-function MSE regression',
                'extra': 'Used for regression problems, penalizes large errors more heavily'
            },
            {
                'front': 'What is cross-entropy loss?',
                'back': 'Loss function measuring difference between predicted and actual probability distributions',
                'formula': '\\[CE = -\\sum_{i=1}^{n} y_i \\log(\\hat{y_i})\\]',
                'source': 'ML Fundamentals',
                'tags': 'loss-function cross-entropy classification',
                'extra': 'Used for classification, works well with softmax activation'
            },
            {
                'front': 'When to use MSE vs cross-entropy loss?',
                'back': 'MSE for regression problems (continuous outputs), cross-entropy for classification problems (probability outputs)',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'loss-function selection',
                'extra': 'Choice depends on problem type and output layer activation function'
            },
            
            # Activation Functions (Agent 1 Priority)
            {
                'front': 'What is the sigmoid activation function?',
                'back': 'Activation function that squashes input to range (0,1), commonly used for binary classification',
                'formula': '\\[\\sigma(x) = \\frac{1}{1 + e^{-x}}\\]',
                'source': 'Neural Networks',
                'tags': 'activation-function sigmoid',
                'extra': 'Can cause vanishing gradient problem in deep networks'
            },
            {
                'front': 'What is the tanh activation function?',
                'back': 'Activation function that squashes input to range (-1,1), zero-centered version of sigmoid',
                'formula': '\\[tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}\\]',
                'source': 'Neural Networks',
                'tags': 'activation-function tanh',
                'extra': 'Often works better than sigmoid due to zero-centered output'
            },
            
            # Training Techniques (Agent 1 Priority)
            {
                'front': 'What is dropout in neural networks?',
                'back': 'Regularization technique that randomly sets some neurons to zero during training to prevent overfitting',
                'formula': '',
                'source': 'Neural Networks',
                'tags': 'regularization dropout',
                'extra': 'Forces network to not rely on specific neurons, improves generalization'
            },
            {
                'front': 'What is batch normalization?',
                'back': 'Technique that normalizes layer inputs to accelerate training and reduce internal covariate shift',
                'formula': '\\[BN(x) = \\gamma \\frac{x - \\mu}{\\sigma} + \\beta\\]',
                'source': 'Neural Networks',
                'tags': 'batch-normalization training',
                'extra': 'Allows higher learning rates and makes network less sensitive to initialization'
            },
            
            # PCA Computational Details (Agent 2 Priority)
            {
                'front': 'How do you determine the number of principal components to keep?',
                'back': 'Use scree plot (elbow method), cumulative explained variance threshold (e.g., 90%), or Kaiser criterion (eigenvalues > 1)',
                'formula': '\\[Cumulative\\_Var = \\frac{\\sum_{i=1}^{k} \\lambda_i}{\\sum_{i=1}^{n} \\lambda_i}\\]',
                'source': 'PCA',
                'tags': 'PCA component-selection scree-plot',
                'extra': 'Balance between dimensionality reduction and information preservation'
            },
            {
                'front': 'What preprocessing is required before PCA?',
                'back': 'Standardize features to have zero mean and unit variance, handle missing values',
                'formula': '\\[z = \\frac{x - \\mu}{\\sigma}\\]',
                'source': 'PCA',
                'tags': 'PCA preprocessing standardization',
                'extra': 'Without standardization, features with larger scales dominate the principal components'
            },
            {
                'front': 'What are the assumptions and limitations of PCA?',
                'back': 'Assumes linear relationships, sensitive to scaling, components may not be interpretable, requires numerical data',
                'formula': '',
                'source': 'PCA',
                'tags': 'PCA assumptions limitations',
                'extra': 'Works best when variables are correlated and relationships are linear'
            },
            
            # K-Means Enhancements (Agent 2 Priority)
            {
                'front': 'What is the elbow method for determining optimal k?',
                'back': 'Plot within-cluster sum of squares (WCSS) vs k, look for elbow where rate of decrease slows significantly',
                'formula': '',
                'source': 'Clustering',
                'tags': 'k-means elbow-method optimal-k',
                'extra': 'Point where adding more clusters doesn\\'t significantly reduce WCSS'
            },
            {
                'front': 'What is K-means++ initialization?',
                'back': 'Smart initialization that chooses initial centroids far apart to improve convergence',
                'formula': '',
                'source': 'Clustering',
                'tags': 'k-means initialization k-means++',
                'extra': 'First centroid random, subsequent ones chosen proportional to squared distance from nearest existing centroid'
            },
            {
                'front': 'What are the main limitations of K-means?',
                'back': 'Assumes spherical clusters, sensitive to initialization, requires pre-specifying k, sensitive to outliers',
                'formula': '',
                'source': 'Clustering',
                'tags': 'k-means limitations',
                'extra': 'Struggles with varying cluster sizes, non-spherical clusters, and different densities'
            },
            
            # ROC/AUC Evaluation (Agent 3 & 4 Priority)
            {
                'front': 'What is ROC curve?',
                'back': 'Plot of True Positive Rate vs False Positive Rate at various classification thresholds',
                'formula': '\\[TPR = \\frac{TP}{TP+FN}, FPR = \\frac{FP}{FP+TN}\\]',
                'source': 'Evaluation',
                'tags': 'evaluation ROC classification',
                'extra': 'Shows trade-off between sensitivity and specificity across all thresholds'
            },
            {
                'front': 'What is AUC in classification?',
                'back': 'Area Under ROC Curve - measures model\\'s ability to distinguish between classes (0.5-1.0)',
                'formula': '',
                'source': 'Evaluation',
                'tags': 'evaluation AUC classification',
                'extra': 'AUC of 0.5 = random classifier, AUC of 1.0 = perfect classifier'
            },
            
            # Regression Evaluation (Agent 3 Priority)
            {
                'front': 'What is R² (coefficient of determination)?',
                'back': 'Measures proportion of variance in dependent variable explained by independent variables',
                'formula': '\\[R^2 = 1 - \\frac{SS_{res}}{SS_{tot}}\\]',
                'source': 'Evaluation',
                'tags': 'evaluation r-squared regression',
                'extra': 'R² = 1 means perfect fit, R² = 0 means model no better than mean'
            },
            {
                'front': 'What are residuals in regression?',
                'back': 'Differences between observed and predicted values used to assess model assumptions',
                'formula': '\\[e_i = y_i - \\hat{y_i}\\]',
                'source': 'Evaluation',
                'tags': 'regression residuals evaluation',
                'extra': 'Residual plots help identify heteroscedasticity, non-linearity, and outliers'
            },
            
            # Statistical Foundations (Agent 4 Priority)
            {
                'front': 'What are Type I and Type II errors?',
                'back': 'Type I: False Positive (rejecting true null hypothesis). Type II: False Negative (accepting false null hypothesis)',
                'formula': '',
                'source': 'Statistics',
                'tags': 'statistics type-1-error type-2-error',
                'extra': 'Trade-off controlled by decision threshold - lowering threshold reduces Type II but increases Type I'
            },
            {
                'front': 'What is statistical significance in ML?',
                'back': 'Measure of whether observed difference in model performance is likely due to chance',
                'formula': '',
                'source': 'Statistics',
                'tags': 'statistics significance hypothesis-testing',
                'extra': 'Use paired t-test on CV scores or McNemar\\'s test for classification comparisons'
            },
            {
                'front': 'What is data leakage?',
                'back': 'Information from future or target variable inappropriately leaking into features during training',
                'formula': '',
                'source': 'Data Preprocessing',
                'tags': 'data-leakage preprocessing',
                'extra': 'Prevent by proper train/test split timing, avoiding future information, careful feature engineering'
            },
            
            # Model Validation (Agent 3 & 4 Priority)
            {
                'front': 'What is hyperparameter tuning?',
                'back': 'Process of finding optimal hyperparameter values using techniques like grid search or random search',
                'formula': '',
                'source': 'Model Selection',
                'tags': 'hyperparameter-tuning optimization model-selection',
                'extra': 'Use nested cross-validation to avoid overfitting to validation set'
            },
            {
                'front': 'How to handle class imbalance?',
                'back': 'Use techniques like SMOTE, stratified sampling, cost-sensitive learning, or balanced evaluation metrics',
                'formula': '',
                'source': 'Classification',
                'tags': 'classification class-imbalance',
                'extra': 'Accuracy can be misleading with imbalanced data - use precision, recall, F1-score instead'
            },
            
            # Advanced Concepts (Agent 2 & 4 Priority)
            {
                'front': 'What is silhouette score?',
                'back': 'Cluster validation metric measuring how similar points are to their own cluster vs other clusters',
                'formula': '\\[s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))}\\]',
                'source': 'Clustering',
                'tags': 'clustering silhouette-score validation',
                'extra': 'Values range from -1 to 1, higher values indicate better clustering'
            },
            {
                'front': 'What is feature importance in ML?',
                'back': 'Measure of how much each feature contributes to model predictions',
                'formula': '',
                'source': 'Model Interpretability',
                'tags': 'feature-importance interpretability',
                'extra': 'Can be calculated using permutation importance, SHAP values, or model-specific methods'
            },
            
            # Ensemble Methods (Agent 4 Priority)
            {
                'front': 'What\\'s the difference between bagging and boosting?',
                'back': 'Bagging trains models in parallel on bootstrap samples. Boosting trains models sequentially, each correcting previous errors',
                'formula': '',
                'source': 'Ensemble Methods',
                'tags': 'ensemble bagging boosting',
                'extra': 'Bagging reduces variance (Random Forest), boosting reduces bias (AdaBoost, XGBoost)'
            },
            {
                'front': 'What is hard vs soft voting in ensembles?',
                'back': 'Hard voting: majority class vote. Soft voting: average predicted probabilities',
                'formula': '',
                'source': 'Ensemble Methods',
                'tags': 'ensemble voting',
                'extra': 'Soft voting generally performs better when base models output calibrated probabilities'
            },
            
            # PyTorch Concepts (Agent 1 Priority)
            {
                'front': 'What is automatic differentiation in PyTorch?',
                'back': 'Computational technique that automatically computes gradients for backpropagation',
                'formula': '',
                'source': 'PyTorch',
                'tags': 'pytorch automatic-differentiation',
                'extra': 'Tracks operations on tensors to build computational graph for gradient computation'
            },
            {
                'front': 'What are PyTorch tensors?',
                'back': 'Multi-dimensional arrays similar to NumPy arrays but with GPU acceleration and automatic differentiation',
                'formula': '',
                'source': 'PyTorch',
                'tags': 'pytorch tensors',
                'extra': 'Foundation of PyTorch - support operations needed for neural network computations'
            }
        ]
        
        return cards
    
    def _convert_to_mathjax(self, latex_formula):
        """Convert LaTeX formula to MathJax format"""
        # Basic conversions for common patterns
        formula = latex_formula.strip()
        
        # Remove existing LaTeX delimiters
        formula = re.sub(r'^\$\$?|\$\$?$', '', formula)
        
        # Common symbol replacements
        replacements = {
            r'\\beta': r'\\beta',
            r'\\mu': r'\\mu',
            r'\\sigma': r'\\sigma',
            r'\\theta': r'\\theta',
            r'\\lambda': r'\\lambda',
            r'\\sum': r'\\sum',
            r'\\prod': r'\\prod',
            r'\\frac': r'\\frac',
            r'\\sqrt': r'\\sqrt'
        }
        
        for old, new in replacements.items():
            formula = formula.replace(old, new)
            
        return formula
    
    def _create_single_anki_deck(self, all_cards):
        """Create and save single Anki deck file"""
        if not all_cards:
            print("❌ No cards to save")
            return
            
        # Create single deck
        deck = genanki.Deck(self.deck_id, self.deck_name)
        
        for card_data in all_cards:
            note = create_ml_note(
                front=card_data['front'],
                back=card_data['back'],
                formula=card_data['formula'],
                source=card_data['source'],
                tags=card_data['tags'],
                extra=card_data['extra']
            )
            deck.add_note(note)
        
        # Save deck as .apkg file
        output_file = self.output_dir / "ML_Foundations_Exam_Prep.apkg"
        package = genanki.Package(deck)
        package.write_to_file(str(output_file))
        
        print(f"💾 Saved {len(all_cards)} cards to {output_file}")
    
    def _save_csv_files(self, all_cards):
        """Save intermediate CSV files for review and editing"""
        csv_file = self.temp_dir / "all_cards.csv"
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['front', 'back', 'formula', 'source', 'tags', 'extra'])
            writer.writeheader()
            writer.writerows(all_cards)
            
        print(f"📄 Saved CSV review file: {csv_file}")

if __name__ == "__main__":
    generator = MLFlashcardGenerator()
    generator.generate_all_cards()