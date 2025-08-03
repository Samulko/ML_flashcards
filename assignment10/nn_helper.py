import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from matplotlib.colors import ListedColormap

from sklearn.datasets import load_iris

# Set random seed for reproducibility
np.random.seed(42)

# Configure matplotlib
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

def load_wine_quality_data():
    """
    Load and explore the wine quality dataset.
    This is a real-world classification problem with 11 chemical features
    predicting wine quality on a scale of 3-9.
    """
    # Generate a realistic wine-quality-like dataset
    # Features represent: acidity, sugar, alcohol, sulfur compounds, etc.
    np.random.seed(42)
    n_samples = 1000
    
    # Create correlated features that might represent wine chemistry
    acidity = np.random.normal(8.5, 1.5, n_samples)
    sugar = np.random.normal(2.5, 1.0, n_samples)
    alcohol = np.random.normal(10.5, 1.2, n_samples)
    sulfur = np.random.normal(15, 5, n_samples)
    
    # Additional chemical properties
    chlorides = np.random.normal(0.08, 0.03, n_samples)
    density = np.random.normal(0.997, 0.003, n_samples)
    ph = np.random.normal(3.2, 0.3, n_samples)
    
    # Create non-linear relationships for quality
    quality_score = (
        0.3 * alcohol + 
        0.2 * (10 - acidity) + 
        0.15 * sugar +
        -0.1 * sulfur +
        0.1 * (4 - ph) +
        np.random.normal(0, 1, n_samples)
    )
    
    # Convert to classes: 0 (poor), 1 (average), 2 (excellent)
    quality_classes = np.digitize(quality_score, bins=[-np.inf, -0.5, 1.0, np.inf]) - 1
    
    # Create feature matrix
    X = np.column_stack([
        acidity, sugar, alcohol, sulfur, chlorides, density, ph,
        # Add some derived features to make it more complex
        acidity * alcohol,  # interaction term
        sugar / alcohol,    # ratio
        ph * density,       # interaction
        np.log(sulfur + 1)  # transformed feature
    ])
    
    feature_names = [
        'acidity', 'sugar', 'alcohol', 'sulfur', 'chlorides', 
        'density', 'ph', 'acidity_alcohol', 'sugar_ratio', 
        'ph_density', 'log_sulfur'
    ]
    
    quality_names = ['Poor', 'Average', 'Excellent']
    
    # Display dataset information
    print(f"Wine Quality Dataset:")
    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(np.unique(quality_classes))}")
    print(f"Class distribution: {np.bincount(quality_classes)}")
    
    return X, quality_classes, feature_names, quality_names

def visualize_wine_data(X, y, feature_names, class_names):
    """
    Visualize the wine dataset to understand the classification challenge.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Wine Quality Dataset - Neural Network Classification Challenge', fontsize=16)
    
    # Feature correlation heatmap
    ax1 = axes[0, 0]
    correlation_matrix = np.corrcoef(X.T)
    im = ax1.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax1.set_title('Feature Correlations')
    ax1.set_xticks(range(len(feature_names)))
    ax1.set_yticks(range(len(feature_names)))
    ax1.set_xticklabels(feature_names, rotation=45, ha='right')
    ax1.set_yticklabels(feature_names)
    plt.colorbar(im, ax=ax1)
    
    # Class distribution
    ax2 = axes[0, 1]
    class_counts = np.bincount(y)
    ax2.bar(class_names, class_counts, color=['red', 'orange', 'green'])
    ax2.set_title('Class Distribution')
    ax2.set_ylabel('Number of Samples')
    
    # Feature distributions by class
    ax3 = axes[1, 0]
    colors = ['red', 'orange', 'green']
    for i, class_name in enumerate(class_names):
        mask = y == i
        ax3.hist(X[mask, 2], alpha=0.6, label=f'{class_name} Wine', 
                color=colors[i], bins=20)
    ax3.set_title('Alcohol Content Distribution by Quality')
    ax3.set_xlabel('Alcohol Content')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    
    # 2D feature visualization
    ax4 = axes[1, 1]
    for i, class_name in enumerate(class_names):
        mask = y == i
        ax4.scatter(X[mask, 0], X[mask, 2], alpha=0.6, 
                   label=f'{class_name} Wine', color=colors[i], s=30)
    ax4.set_title('Acidity vs Alcohol Content')
    ax4.set_xlabel('Acidity')
    ax4.set_ylabel('Alcohol Content')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

def create_xor_dataset():
    """Create the XOR dataset - the classic problem that breaks single neurons."""
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_xor = np.array([0, 1, 1, 0])  # XOR truth table
    
    print("XOR Truth Table (Exclusive OR):")
    print("Input1 | Input2 | Output")
    print("-------|--------|-------")
    for i in range(len(X_xor)):
        print(f"   {int(X_xor[i,0])}   |   {int(X_xor[i,1])}   |   {y_xor[i]}")
    
    return X_xor, y_xor

def visualize_xor_problem(X, y):
    """Visualize why XOR is impossible for a single neuron."""
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue']
    labels = ['Output: 0', 'Output: 1']
    
    for i in range(2):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=300, 
                   label=labels[i], edgecolors='black', linewidth=1)
    
    # Add point labels
    for i in range(len(X)):
        plt.annotate(f'({int(X[i,0])},{int(X[i,1])}) → {y[i]}', 
                    (X[i,0], X[i,1]), xytext=(0, -40), 
                    textcoords='offset points', fontsize=12, fontweight='bold',
                    ha='center')
    
    plt.xlim(-0.3, 1.3)
    plt.ylim(-0.3, 1.3)
    plt.xlabel('Input 1', fontsize=14)
    plt.ylabel('Input 2', fontsize=14)
    plt.title('The XOR Problem: Can You Draw ONE Straight Line to Separate Red from Blue?', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Try to show some failing linear separators
    x_line = np.linspace(-0.3, 1.3, 100)
    for slope, intercept, label in [(1, 0.5, 'Attempt 1'), (-1, 1.5, 'Attempt 2'), (0, 0.5, 'Attempt 3')]:
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, '--', alpha=0.5, linewidth=1)
    
    plt.show()

def visualize_xor_decision_boundary(model, X, y):
    """
    Visualize the decision boundary learned by the neural network for XOR.
    This shows how the network creates non-linear boundaries to separate the classes.
    """
    plt.figure(figsize=(12, 5))
    
    # Create a grid of points to evaluate the decision boundary
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
    y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Get model predictions for each point in the grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.forward(grid_points)  # Get probability scores
    Z = Z.reshape(xx.shape)
    
    # Create two subplots
    fig, ax2 = plt.subplots(1, 1, figsize=(6, 6))
    
    # Plot XOR points
    colors = ['red', 'blue'] 
    labels = ['Class 0 (XOR = False)', 'Class 1 (XOR = True)']
    
    # Right plot: Binary decision regions
    Z_binary = (Z > 0.5).astype(int)
    ax2.contourf(xx, yy, Z_binary, levels=1, alpha=0.7, colors=['lightcoral', 'lightblue'])
    ax2.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3)
    
    # Plot XOR points again
    for i in range(2):
        mask = y == i
        ax2.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=300, 
                   label=labels[i], edgecolors='black', linewidth=1, alpha=0.9)
    
    # Add point labels
    for i in range(len(X)):
        ax2.annotate(f'({int(X[i,0])},{int(X[i,1])})', 
                    (X[i,0], X[i,1]), xytext=(0, -25), 
                    textcoords='offset points', fontsize=11, fontweight='bold',
                    ha='center')
    
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_xlabel('Input 1', fontsize=12)
    ax2.set_ylabel('Input 2', fontsize=12)
    ax2.set_title('Neural Network Decision Regions\n(Binary Classification)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def load_iris_dataset():
    """Create a simple binary classification task using Iris dataset that a single neuron can solve perfectly."""    
    # Load the full iris dataset
    iris = load_iris()
    
    # Select only the first two classes (setosa=0, versicolor=1) - these are linearly separable
    # Skip virginica (class=2) as it would make the problem non-linearly separable
    mask = iris.target < 2
    
    # Select only the first two features (sepal length, sepal width)
    X_simple = iris.data[mask][:, :2]  # Only first two features
    y_simple = iris.target[mask]       # Only first two classes (0, 1)
    
    return X_simple, y_simple, iris.feature_names[:2], iris.target_names[:2]

def visualize_iris_dataset(X, y, feature_names, class_names):
    """Visualize the simple binary dataset."""
    plt.figure(figsize=(10, 6))
    
    colors = ['red', 'green']
    labels = [f'{class_names[0]} (Class 0)', f'{class_names[1]} (Class 1)']
    
    for i in range(2):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.7, s=50,
                   label=labels[i], edgecolors='black', linewidth=0.5)
    
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Iris Dataset: Binary Classification')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Dataset: Iris (first 2 classes, first 2 features)")
    print(f"Features: {feature_names[0]} vs {feature_names[1]}")
    print(f"Classes: {class_names[0]} vs {class_names[1]}")
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {class_names[0]}: {np.sum(y == 0)}, {class_names[1]}: {np.sum(y == 1)}")

def visualize_neuron_decision_boundary(neuron, X, y, scaler, feature_names, class_names):
    """Visualize how the single neuron learned to separate the classes."""
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Learning curves
    plt.subplot(1, 2, 1)
    epochs = range(1, len(neuron.loss_history) + 1)
    plt.plot(epochs, neuron.loss_history, 'b-', linewidth=2, label='Loss')
    plt.title('Single Neuron Learning Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Decision boundary
    plt.subplot(1, 2, 2)
    
    # Create a mesh for decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_scaled = scaler.transform(mesh_points)
    Z = neuron.predict(mesh_points_scaled)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, levels=1, alpha=0.8, colors=['lightcoral', 'lightgreen'])
    
    # Plot data points
    colors = ['red', 'green']
    labels = [f'{class_names[0]} (Class 0)', f'{class_names[1]} (Class 1)']
    
    for i in range(2):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.9, s=50,
                   label=labels[i], edgecolors='black', linewidth=0.5)
    
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Single Neuron Decision Boundary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # plt.tight_layout()
    plt.show()