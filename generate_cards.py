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
                'extra': 'ANALOGY: Like finding the best camera angles to photograph a 3D sculpture - you want views that capture the most detail with fewest shots. PCA finds the "best angles" (principal components) in your data space. CONNECTIONS: Related to SVD (Singular Value Decomposition), eigendecomposition. Used before clustering (curse of dimensionality), neural networks (feature extraction), and visualization. INTUITION: High-dimensional data often lies on lower-dimensional manifolds - PCA finds linear approximations of these manifolds. Essential for avoiding curse of dimensionality in high-dim spaces.'
            },
            {
                'front': 'What is the covariance matrix formula in PCA?',
                'back': 'Matrix representing relationships between features',
                'formula': '\\[Cov(X) = \\frac{1}{n}X^TX\\]',
                'source': 'TOPICS.md',
                'tags': 'PCA covariance-matrix formula',
                'extra': 'INTUITION: Covariance matrix is like a "correlation map" showing how features move together. Diagonal elements = variance of each feature, off-diagonal = covariance between features. ANALOGY: Like a dance partner compatibility matrix - shows which dancers move in sync. MATHEMATICAL INSIGHT: X must be mean-centered first! Formula assumes X is (n×p) with rows=samples, cols=features. CONNECTIONS: Eigendecomposition of this matrix gives principal components. Related to correlation matrix (normalized version). PRACTICAL: Large covariances indicate redundant features - perfect candidates for dimensionality reduction.'
            },
            {
                'front': 'How are principal components calculated?',
                'back': 'Linear combinations of original features using eigenvectors',
                'formula': '\\[PC = X \\cdot v\\]',
                'source': 'TOPICS.md',
                'tags': 'PCA principal-components',
                'extra': 'DEEP DIVE: Eigenvectors (v) are the "directions of maximum variance" - like finding the main axis of a football (not round!). PROCESS: 1) Mean-center data 2) Compute covariance matrix 3) Find eigenvalues/eigenvectors 4) Sort eigenvectors by eigenvalue (largest first) 5) Project data onto top-k eigenvectors. ANALOGY: Like rotating coordinate system to align with data\'s natural "grain" - imagine wood grain patterns. CONNECTIONS: Each PC is orthogonal (uncorrelated), first PC captures most variance. Used in: face recognition (eigenfaces), genomics, finance portfolio analysis.'
            },
            {
                'front': 'What does explained variance tell us in PCA?',
                'back': 'Proportion of dataset variance captured by each principal component',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'PCA explained-variance',
                'extra': 'DECISION MAKING: Like budgeting information - "How much of the story does each component tell?" RULE OF THUMB: Keep components until cumulative explained variance reaches 90-95%. ANALOGY: Imagine explaining a movie plot - first PC gives main storyline (most important), subsequent PCs add subplots and details. SCREE PLOT: Graph eigenvalues, look for "elbow" where slope flattens dramatically. CONNECTIONS: Related to eigenvalues (larger eigenvalue = more explained variance). Kaiser criterion: keep components with eigenvalue > 1. PRACTICAL: Trade-off between information retention and dimensionality reduction.'
            },
            
            # K-means Clustering (High Priority - New Topic)
            {
                'front': 'What is K-means clustering?',
                'back': 'Partitioning algorithm that divides data into k clusters by minimizing within-cluster variance',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'clustering k-means',
                'extra': 'VISUAL ANALOGY: Like organizing a messy room by creating k boxes and putting similar items together, then adjusting box positions until items are closest to their own box. ASSUMPTIONS: Spherical clusters (like circles, not crescents), similar sizes, similar densities. FAILS when: Clusters are elongated, nested, or have vastly different sizes. CONNECTIONS: Related to EM algorithm (hard assignment version), Vector Quantization, Voronoi diagrams. PREPROCESSING: Scale features first! Distance-based algorithm sensitive to feature scales. APPLICATIONS: Market segmentation, image compression, gene sequencing.'
            },
            {
                'front': 'What is the K-means objective function?',
                'back': 'Minimize within-cluster sum of squared distances',
                'formula': '\\[\\min \\sum_i \\sum_{x \\in C_i} ||x - \\mu_i||^2\\]',
                'source': 'TOPICS.md',
                'tags': 'k-means objective-function',
                'extra': 'INTUITION: "Make each point as close as possible to its cluster center" - like minimizing total walking distance in a city with k meeting points. ALTERNATIVE NAMES: Within-Cluster Sum of Squares (WCSS), inertia, distortion. OPTIMIZATION: NP-hard problem! Lloyd\'s algorithm finds local optima. CONNECTIONS: Related to variance decomposition - minimizing within-cluster variance. Similar to expectation in EM algorithm. PRACTICAL: Elbow method plots WCSS vs k to find optimal number of clusters. DISTANCE: Usually Euclidean, but can use Manhattan, cosine similarity.'
            },
            {
                'front': 'How do you update centroids in K-means?',
                'back': 'Take the mean of all points assigned to each cluster',
                'formula': '\\[\\mu_i = \\frac{1}{|C_i|} \\sum_{x \\in C_i} x\\]',
                'source': 'TOPICS.md',
                'tags': 'k-means centroid-update',
                'extra': 'ANALOGY: Like finding the "center of mass" or "balance point" of each group - if you put weights at each data point, where would you place the fulcrum? MATHEMATICAL INSIGHT: Mean minimizes sum of squared distances - this is why K-means uses Euclidean distance. ALGORITHM STEP: This is the M-step (Maximization) in Lloyd\'s algorithm, alternates with E-step (assignment). CONVERGENCE: Centroids move less each iteration until they stabilize. Can also stop after fixed iterations or when objective function change is small. EMPTY CLUSTERS: Handle by reinitializing or using K-means++.'
            },
            {
                'front': 'What are the two steps of Lloyd\'s K-means algorithm?',
                'back': 'Assignment step (assign points to nearest centroid) and Update step (recalculate centroids)',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'k-means lloyds-algorithm',
                'extra': 'DANCE ANALOGY: Like a dance where partners (points) choose their favorite dancer (centroid), then dancers move to the center of their group, repeat until everyone is happy. E-STEP (Assignment): Hard assignment - each point belongs to exactly one cluster (contrast with soft assignment in EM). M-STEP (Update): Recalculate centroids as means. CONVERGENCE: Guaranteed to converge to local optimum, but depends on initialization. CONNECTIONS: Similar to EM algorithm structure. Coordinate descent optimization. PRACTICAL: Usually converges in few iterations, but can get stuck in poor local optima.'
            },
            
            # EM Algorithm (High Priority - New Topic)
            {
                'front': 'What is the EM algorithm?',
                'back': 'Expectation-Maximization algorithm for finding parameters when latent variables exist',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'EM algorithm latent-variables',
                'extra': 'CHICKEN-AND-EGG ANALOGY: Like trying to learn two things at once - "If I knew which cluster each point belonged to, I could estimate cluster parameters. If I knew cluster parameters, I could assign points to clusters." EM solves this circular dependency. LATENT VARIABLES: Hidden/unobserved variables (like cluster membership). APPLICATIONS: Gaussian Mixture Models, Hidden Markov Models, factor analysis, missing data imputation. GUARANTEE: Always increases likelihood (or stays same), guaranteed to converge to local maximum. CONNECTIONS: Generalizes K-means (soft assignment vs hard), related to variational inference.'
            },
            {
                'front': 'What is the E-step in EM algorithm?',
                'back': 'Calculate expectation of log-likelihood given current parameters',
                'formula': '\\[Q(θ|θ^{(t)}) = E[\\log L(θ|X,Z)|X,θ^{(t)}]\\]',
                'source': 'TOPICS.md',
                'tags': 'EM e-step expectation',
                'extra': 'INTUITIVE MEANING: "Given my current model, how likely is each data point to belong to each cluster?" Computes soft assignments (probabilities) rather than hard assignments. GAUSSIAN MIXTURE EXAMPLE: For each point, calculate probability it came from each Gaussian component using current means/covariances. MATHEMATICAL: Takes expectation over latent variables Z given observed data X and current parameters. CONNECTIONS: Similar to K-means assignment step but with probabilities. Creates "responsibility" matrix showing how responsible each cluster is for each point.'
            },
            {
                'front': 'What is the M-step in EM algorithm?',
                'back': 'Find parameters that maximize the expectation from E-step',
                'formula': '\\[θ^{(t+1)} = \\arg\\max Q(θ|θ^{(t)})\\]',
                'source': 'TOPICS.md',
                'tags': 'EM m-step maximization',
                'extra': 'INTUITIVE MEANING: "Given these soft assignments, what are the best parameters for my model?" Uses weighted versions of standard estimators. GAUSSIAN MIXTURE EXAMPLE: Update means using weighted averages (weights = responsibilities from E-step), update covariances using weighted sample covariances. WEIGHTED UPDATES: Each data point contributes to parameter estimates proportional to its assignment probability. CONNECTIONS: Generalizes K-means centroid update (hard weights vs soft weights). Maximum likelihood estimation with weighted data.'
            },
            
            # Neural Networks (Highest Priority - 36 points)
            {
                'front': 'What is a neural network?',
                'back': 'Computational model inspired by biological neurons, with layers of interconnected nodes',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'neural-networks definition',
                'extra': 'BIOLOGICAL ANALOGY: Like a simplified brain where artificial neurons receive signals, process them, and pass signals forward. Each connection has a "strength" (weight). UNIVERSAL APPROXIMATION: With enough hidden units, can approximate any continuous function (theoretically). LAYERS: Input layer (data), hidden layers (feature extraction/transformation), output layer (predictions). POWER: Can learn non-linear relationships, feature interactions, and hierarchical representations. CONNECTIONS: Generalize linear regression (single layer = linear regression), logistic regression (single layer + sigmoid). APPLICATIONS: Image recognition, NLP, game playing, drug discovery.'
            },
            {
                'front': 'What is the forward pass in neural networks?',
                'back': 'Process of computing output by propagating input through network layers',
                'formula': '\\[h = \\sigma(Wx + b)\\]',
                'source': 'TOPICS.md',
                'tags': 'neural-networks forward-pass',
                'extra': 'ASSEMBLY LINE ANALOGY: Like a factory assembly line where each layer transforms the input, passing it to the next station. Raw materials (input) → processed goods (hidden layers) → final product (output). COMPUTATION: Linear transformation (Wx + b) followed by non-linear activation (σ). WITHOUT ACTIVATION: Network would just be linear regression! LAYER-BY-LAYER: Output of layer i becomes input to layer i+1. MATRIX OPERATIONS: Highly parallelizable, efficient on GPUs. INFORMATION FLOW: Only forward direction during inference, no feedback loops (unlike RNNs).'
            },
            {
                'front': 'What is backpropagation?',
                'back': 'Algorithm for computing gradients of loss function with respect to network weights',
                'formula': '\\[\\frac{\\partial L}{\\partial W} = \\frac{\\partial L}{\\partial h} \\cdot \\frac{\\partial h}{\\partial W}\\]',
                'source': 'TOPICS.md',
                'tags': 'neural-networks backpropagation',
                'extra': 'BLAME ASSIGNMENT ANALOGY: Like tracing responsibility for a mistake backwards through a company hierarchy - "How much did each department contribute to the final error?" CHAIN RULE: Mathematical technique for computing derivatives of composite functions. EFFICIENT: Computes all gradients in one backward pass, reusing computations. PROCESS: 1) Forward pass computes predictions 2) Compute loss 3) Backward pass computes gradients 4) Update weights. VANISHING GRADIENTS: Problem in deep networks where gradients become very small in early layers. COMPUTATIONAL GRAPH: Network as graph of operations, backprop traverses graph backwards.'
            },
            {
                'front': 'What is the ReLU activation function?',
                'back': 'Rectified Linear Unit: outputs input if positive, zero otherwise',
                'formula': '\\[ReLU(x) = \\max(0, x)\\]',
                'source': 'ML Fundamentals',
                'tags': 'neural-networks activation-function relu',
                'extra': 'SWITCH ANALOGY: Like an electrical switch - if signal is positive, let it through; if negative, block it completely. ADVANTAGES: Simple computation, gradient is 1 for positive inputs (no vanishing gradient), sparsity (many neurons output 0). BIOLOGICAL: Somewhat resembles neuron firing patterns. PROBLEMS: "Dying ReLU" - neurons can get stuck outputting 0 and never recover. VARIANTS: Leaky ReLU (small slope for negative), ELU, Swish. GRADIENT: ∂ReLU/∂x = 1 if x>0, else 0 (undefined at 0, usually set to 0). DEEP NETWORKS: Enables training of very deep networks without vanishing gradients.'
            },
            {
                'front': 'What is gradient descent?',
                'back': 'Optimization algorithm that iteratively updates parameters in direction of steepest descent',
                'formula': '\\[θ = θ - α∇J(θ)\\]',
                'source': 'ML Fundamentals',
                'tags': 'optimization gradient-descent',
                'extra': 'MOUNTAIN CLIMBING ANALOGY: Like hiking down a mountain in fog - you can only see your immediate surroundings, so you always step in the steepest downward direction. LEARNING RATE (α): Step size - too large and you overshoot the valley, too small and training is slow. GRADIENT: Points in direction of steepest increase, so we go opposite direction (negative gradient). VARIANTS: SGD (stochastic - use mini-batches), Adam (adaptive learning rates), momentum (remembers previous directions). LOCAL MINIMA: Can get stuck in local valleys instead of finding global minimum. CONVEX FUNCTIONS: Guaranteed to find global minimum, neural networks are non-convex.'
            },
            
            # Linear Regression (High Priority - 27 points)
            {
                'front': 'What is linear regression?',
                'back': 'Statistical method for modeling relationship between dependent variable and independent variables',
                'formula': '\\[y = β_0 + β_1x_1 + β_2x_2 + ... + ε\\]',
                'source': 'ML Fundamentals',
                'tags': 'linear-regression supervised-learning',
                'extra': 'INTUITION: Finding the "best-fit line" through data points - like drawing a straight line through a scatter plot that minimizes distances to points. ASSUMPTIONS: Linear relationship, independence of errors, homoscedasticity (constant error variance), normality of residuals. COEFFICIENTS: β_0 = intercept (value when all x=0), β_i = slope (change in y per unit change in x_i). ERROR TERM (ε): Captures unmeasured factors, noise, model limitations. GEOMETRICALLY: In p-dimensional space, fits hyperplane to data. CONNECTIONS: Foundation for logistic regression, neural networks (single layer), polynomial regression.'
            },
            {
                'front': 'What is the normal equation for linear regression?',
                'back': 'Closed-form solution for optimal parameters',
                'formula': '\\[β = (X^TX)^{-1}X^Ty\\]',
                'source': 'ML Fundamentals',
                'tags': 'linear-regression normal-equation',
                'extra': 'MAGIC FORMULA: Directly computes optimal weights without iteration! DERIVATION: Set gradient of least squares loss to zero, solve for β. COMPUTATIONAL: X^TX is Gram matrix (p×p), computationally expensive for large p. REQUIREMENTS: X^TX must be invertible (full rank) - problems with multicollinearity. WHEN TO USE: Small datasets, few features (<10k), want exact solution. ALTERNATIVES: Gradient descent for large datasets, Ridge regression when X^TX is singular. GEOMETRIC: Projects y onto column space of X, finds closest point.'
            },
            {
                'front': 'What is Ridge regression?',
                'back': 'Linear regression with L2 regularization to prevent overfitting',
                'formula': '\\[\\hat{β} = (X^TX + λI)^{-1}X^Ty\\]',
                'source': 'TOPICS.md',
                'tags': 'ridge-regression regularization',
                'extra': 'PENALTY ANALOGY: Like speed limits for coefficients - prevents any single coefficient from becoming too large and dominating the model. SHRINKAGE: Pulls coefficients toward zero but never exactly zero (contrast with Lasso). BIAS-VARIANCE: Adds bias but reduces variance, often improving generalization. MULTICOLLINEARITY: Handles correlated features well by distributing weight among them. REGULARIZATION PARAMETER (λ): Cross-validation to choose optimal value. GEOMETRIC: Constrains coefficients to lie within L2 ball (sphere). MATRIX INSIGHT: λI makes X^TX + λI always invertible, fixes singularity issues.'
            },
            {
                'front': 'What is the difference between Ridge and Lasso regression?',
                'back': 'Ridge uses L2 penalty (squared weights), Lasso uses L1 penalty (absolute weights)',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'regularization ridge lasso',
                'extra': 'GEOMETRIC INTUITION: Ridge constraint is a circle (smooth), Lasso is a diamond (corners). Corners of diamond cause coefficients to hit exactly zero. FEATURE SELECTION: Lasso automatically selects features (sparse solutions), Ridge keeps all features but shrinks them. CORRELATED FEATURES: Ridge spreads weights evenly among correlated features, Lasso arbitrarily picks one. COMPUTATIONAL: Ridge has closed-form solution, Lasso requires iterative algorithms. ELASTIC NET: Combines both penalties - L1 for sparsity, L2 for groupings. WHEN TO USE: Lasso when you believe few features matter, Ridge when many features contribute.'
            },
            
            # Logistic Regression (High Priority)
            {
                'front': 'What is logistic regression?',
                'back': 'Classification algorithm using logistic function to model probability of binary outcomes',
                'formula': '\\[p = \\frac{1}{1 + e^{-z}}\\]',
                'source': 'ML Fundamentals',
                'tags': 'logistic-regression classification',
                'extra': 'S-CURVE ANALOGY: Like a smooth switch that gradually transitions from 0 to 1, instead of linear regression\'s unlimited range. LINEAR PREDICTOR: z = β₀ + β₁x₁ + β₂x₂ + ... (same as linear regression). SIGMOID FUNCTION: Maps any real number to (0,1) interval - perfect for probabilities! DECISION BOUNDARY: When p = 0.5, z = 0, so β₀ + β₁x₁ + ... = 0 defines boundary. ODDS INTERPRETATION: log(p/(1-p)) = z, so coefficients represent log-odds ratios. CONNECTIONS: Generalized Linear Model (GLM), neural network with single layer + sigmoid activation.'
            },
            {
                'front': 'What is the softmax function?',
                'back': 'Generalization of logistic function for multi-class classification',
                'formula': '\\[p_i = \\frac{\\exp(w_i^T x)}{\\sum_j \\exp(w_j^T x)}\\]',
                'source': 'TOPICS.md',
                'tags': 'logistic-regression softmax multi-class',
                'extra': 'COMPETITION ANALOGY: Like a talent competition where each class "competes" with a score (w_i^T x), and probabilities are determined by relative performance. NORMALIZATION: Probabilities sum to 1 across all classes. EXPONENTIAL: Amplifies differences between scores - small differences in scores become large differences in probabilities. TEMPERATURE: Can add temperature parameter to control sharpness of distribution. ONE-HOT: Often used with cross-entropy loss and one-hot encoded targets. CONNECTIONS: Reduces to sigmoid for binary case, used as final layer in neural networks for classification.'
            },
            
            # Decision Trees (19 points)
            {
                'front': 'What is entropy in decision trees?',
                'back': 'Measure of impurity or randomness in a dataset',
                'formula': '\\[H(S) = -\\sum p_i \\log_2(p_i)\\]',
                'source': 'TOPICS.md',
                'tags': 'decision-trees entropy',
                'extra': 'INFORMATION THEORY ANALOGY: Like measuring "surprise" in a message - if all examples are same class (pure), entropy = 0 (no surprise). If equal mix of classes, entropy is maximum (most surprise). DECISION MAKING: Entropy guides tree splits - we want to ask questions that reduce uncertainty the most. BINARY EXAMPLE: 50-50 split has entropy = 1 bit, 90-10 split has entropy ≈ 0.47 bits. CONNECTIONS: Related to information gain, Gini impurity (similar concept), Shannon information theory. PRACTICAL: Lower entropy after split means better question/split.'
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
                'extra': 'SAFETY BUFFER ANALOGY: Like drawing the widest possible "no man\'s land" between two armies - points closest to border (support vectors) determine the boundary. GEOMETRIC INTUITION: In 2D, finds line with maximum distance to nearest points from each class. In higher dimensions, finds hyperplane. SPARSE SOLUTION: Only support vectors matter for decision boundary - can ignore all other training points! ROBUSTNESS: Maximum margin principle provides better generalization than simply finding any separating boundary. SUPPORT VECTORS: Critical points that define the solution - removing them changes the decision boundary.'
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
                'extra': 'ARCHERY ANALOGY: Bias = systematic error (consistently missing the target in same direction), Variance = inconsistency (shots scattered around). IDEAL: Low bias AND low variance (tight cluster at bullseye). REAL TRADEOFF: Usually can\'t have both - complex models fit training data well (low bias) but predictions vary with different training sets (high variance). IRREDUCIBLE ERROR: Noise component can\'t be reduced regardless of model. PRACTICAL: Simple models (linear) = high bias/low variance, Complex models (deep neural nets) = low bias/high variance. SWEET SPOT: Find model complexity that minimizes total error.'
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
                'extra': 'SIMPLE INTUITION: "How often is the model right?" IMBALANCED DATA TRAP: 99% accuracy sounds great, but if 99% of data is negative class, a "always predict negative" model achieves this! MEDICAL EXAMPLE: Cancer screening with 1% cancer rate - 99% accuracy might mean missing all cancer cases. BASELINE: Always compare to simple baselines (majority class, random guessing). ALTERNATIVES: Use precision, recall, F1-score, or balanced accuracy for imbalanced datasets. WHEN USEFUL: Balanced datasets where all classes matter equally.'
            },
            {
                'front': 'What is precision?',
                'back': 'Fraction of true positives among predicted positives',
                'formula': '\\[Precision = \\frac{TP}{TP + FP}\\]',
                'source': 'TOPICS.md',
                'tags': 'evaluation precision',
                'extra': 'QUALITY CONTROL ANALOGY: Like quality control in manufacturing - "Of all products we labeled as \'good\', what fraction actually are good?" FALSE POSITIVE COST: Emphasizes minimizing false alarms. EMAIL SPAM: High precision means few legitimate emails marked as spam. MEDICAL: High precision means few healthy patients diagnosed with disease. TRADE-OFF: Increasing precision often decreases recall (fewer positive predictions overall). EXTREME CASE: Predict positive only when 100% certain → perfect precision but terrible recall.'
            },
            {
                'front': 'What is recall (sensitivity)?',
                'back': 'Fraction of true positives among actual positives',
                'formula': '\\[Recall = \\frac{TP}{TP + FN}\\]',
                'source': 'TOPICS.md',
                'tags': 'evaluation recall sensitivity',
                'extra': 'SEARCH AND RESCUE ANALOGY: "Of all people who are actually lost, how many did we find?" Missing people (false negatives) is catastrophic. FALSE NEGATIVE COST: Emphasizes not missing positive cases. MEDICAL SCREENING: High recall means catching most disease cases, even if some false alarms. SECURITY: Airport screening prioritizes recall - better to flag innocent travelers than miss threats. SYNONYMS: Sensitivity, True Positive Rate. EXTREME CASE: Predict everyone as positive → perfect recall but terrible precision.'
            },
            {
                'front': 'What is F1-score?',
                'back': 'Harmonic mean of precision and recall',
                'formula': '\\[F1 = \\frac{2 \\times Precision \\times Recall}{Precision + Recall}\\]',
                'source': 'TOPICS.md',
                'tags': 'evaluation f1-score',
                'extra': 'BALANCE ANALOGY: Like finding the sweet spot between two competing goals - quality (precision) vs completeness (recall). HARMONIC MEAN: Penalizes extreme values more than arithmetic mean - if either precision or recall is low, F1 is low. SINGLE METRIC: Convenient single number for model comparison, especially with imbalanced data. LIMITATIONS: Treats precision and recall equally - may not match business needs. VARIANTS: Fβ score weights recall β times as important as precision. INTERPRETATION: F1=1 is perfect, F1=0 is worst possible.'
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
                'extra': 'MEMORIZATION ANALOGY: Like a student who memorizes textbook problems perfectly but fails on new exam questions - learned specific examples, not general principles. SYMPTOMS: Large gap between training and validation performance, model performs worse as complexity increases beyond optimal point. CAUSES: Too many parameters relative to data, training too long, noise in training data. SOLUTIONS: Regularization (Ridge, Lasso, dropout), cross-validation, early stopping, more data, simpler model. DETECTION: Use validation set or cross-validation to monitor generalization performance during training.'
            },
            {
                'front': 'What is underfitting?',
                'back': 'Model is too simple to capture underlying patterns in data',
                'formula': '',
                'source': 'ML Fundamentals',
                'tags': 'underfitting bias',
                'extra': 'OVERSIMPLIFICATION ANALOGY: Like trying to describe a symphony with only three notes - missing essential complexity. SYMPTOMS: Poor performance on both training AND test data, training error remains high. HIGH BIAS: Model makes strong assumptions that don\'t match reality (e.g., linear model for curved relationship). SOLUTIONS: Increase model complexity (more features, polynomial terms, deeper networks), reduce regularization, train longer. GOLDILOCKS PRINCIPLE: Need model that\'s "just right" - complex enough to capture patterns but simple enough to generalize.'
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
                'extra': 'Point where adding more clusters doesn\'t significantly reduce WCSS'
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
                'back': 'Area Under ROC Curve - measures model\'s ability to distinguish between classes (0.5-1.0)',
                'formula': '',
                'source': 'Evaluation',
                'tags': 'evaluation AUC classification',
                'extra': 'PROBABILITY INTERPRETATION: AUC = probability that model ranks a random positive example higher than a random negative example. SCALE: 0.5 = random guessing (coin flip), 1.0 = perfect separation, <0.5 = worse than random (flip predictions!). THRESHOLD-INDEPENDENT: Single number summarizing model performance across all possible thresholds. RANKING QUALITY: Measures how well model ranks examples, not just classification accuracy. IMBALANCED DATA: Can be misleading - high AUC even when precision/recall are poor. PRACTICAL: Good baseline metric, but supplement with precision-recall for imbalanced datasets.'
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
                'extra': 'Use paired t-test on CV scores or McNemars test for classification comparisons'
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
                'front': 'What is the difference between bagging and boosting?',
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