import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris, make_classification, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import seaborn as sns

np.random.seed(42)

def load_and_explore_iris_data():
    """
    Load the Iris dataset and prepare it for binary classification.
    We'll use Setosa (0) vs Versicolor (1) for simplicity.
    
    Returns:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target vector (-1 for Setosa, +1 for Versicolor)
        feature_names (list): Names of the features
    """
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    
    # Select only Setosa (0) and Versicolor (1) classes for binary classification
    binary_mask = y <= 1
    X = X[binary_mask]
    y = y[binary_mask]
    
    # Convert labels to -1 and +1 for SVM
    y = np.where(y == 0, -1, 1)
    
    return X, y, feature_names

def explore_iris_dataset(X, y, feature_names):
    """
    Explore the Iris dataset by creating visualizations and printing statistics.
    """
    print(f"Dataset shape: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature names: {feature_names}")
    print(f"Class distribution: {np.bincount(y == 1)}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Iris Dataset - Linear Separability Analysis', fontsize=16)
    
    feature_pairs = [(0, 1), (0, 2), (1, 3), (2, 3)]
    
    for i, (feat1, feat2) in enumerate(feature_pairs):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        setosa_mask = y == -1
        versicolor_mask = y == 1
        
        ax.scatter(X[setosa_mask, feat1], X[setosa_mask, feat2], 
                  c='red', label='Setosa (-1)', alpha=0.7, s=50)
        ax.scatter(X[versicolor_mask, feat1], X[versicolor_mask, feat2], 
                  c='blue', label='Versicolor (+1)', alpha=0.7, s=50)
        ax.set_xlabel(feature_names[feat1])
        ax.set_ylabel(feature_names[feat2])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{feature_names[feat1]} vs {feature_names[feat2]}')
    
    plt.tight_layout()
    plt.show(block=False)

def preprocess_iris_data(X, y, test_size=0.2):
    """
    Preprocess the Iris data: standardization and train-test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    print(f"Training labels distribution (-1, +1): {np.unique(y_train, return_counts=True)}")
    print(f"Test labels distribution (-1, +1): {np.unique(y_test, return_counts=True)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def plot_linear_svm_boundary(svm, X_2d, y, feature_names, title="Linear SVM Decision Boundary"):
    """
    Plot the Linear SVM decision boundary for 2D data.
    """
    # Create mesh grid
    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Compute decision function on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm.decision_function(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    plt.colorbar(label='Decision Function Value')
    
    # Plot decision boundary
    plt.contour(xx, yy, Z, levels=[0], colors='black', linestyles='--', linewidths=2)
    
    # Plot data points
    class_neg_mask = y == -1
    class_pos_mask = y == 1
    plt.scatter(X_2d[class_neg_mask, 0], X_2d[class_neg_mask, 1], 
               c='red', marker='o', s=50, label='Setosa (-1)', edgecolors='black')
    plt.scatter(X_2d[class_pos_mask, 0], X_2d[class_pos_mask, 1], 
               c='blue', marker='s', s=50, label='Versicolor (+1)', edgecolors='black')
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

def create_simple_nonlinear_example():
    """
    Create a simple XOR-like dataset to demonstrate Linear SVM limitations.
    """
    np.random.seed(42)
    n_samples = 100
    
    # Create XOR-like pattern
    X_xor = np.random.randn(n_samples, 2) * 0.8
    # XOR logic: positive class when both features have same sign
    y_xor = np.where((X_xor[:, 0] * X_xor[:, 1]) > 0, 1, -1)
    
    # Add some margin for visualization
    X_xor = X_xor * 1.5
    
    print(f"Created XOR-like dataset with {n_samples} samples")
    print(f"Class distribution: {np.unique(y_xor, return_counts=True)}")
    
    return X_xor, y_xor

def visualize_linear_svm_failure(X_xor, y_xor, svm_model):
    """
    Visualize how Linear SVM fails on non-linearly separable data.
    """
    if X_xor is None or y_xor is None or svm_model is None:
        print("Cannot visualize: Missing data or model.")
        return
        
    # Create mesh grid
    h = 0.02
    x_min, x_max = X_xor[:, 0].min() - 1, X_xor[:, 0].max() + 1
    y_min, y_max = X_xor[:, 1].min() - 1, X_xor[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Compute decision function
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_model.decision_function(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Create the plot
    plt.figure(figsize=(12, 5))
    
    # Left plot: Data points only
    plt.subplot(1, 2, 1)
    pos_mask = y_xor == 1
    neg_mask = y_xor == -1
    plt.scatter(X_xor[pos_mask, 0], X_xor[pos_mask, 1], 
               c='blue', marker='o', s=50, label='Class +1', alpha=0.8, edgecolors='k')
    plt.scatter(X_xor[neg_mask, 0], X_xor[neg_mask, 1], 
               c='red', marker='s', s=50, label='Class -1', alpha=0.8, edgecolors='k')
    plt.title('XOR-like Pattern\n(Non-linearly Separable)', fontsize=14)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Right plot: With Linear SVM decision boundary
    plt.subplot(1, 2, 2)
    plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
    plt.contour(xx, yy, Z, levels=[0], colors='black', linestyles='--', linewidths=2)
    plt.scatter(X_xor[pos_mask, 0], X_xor[pos_mask, 1], 
               c='blue', marker='o', s=50, label='Class +1', alpha=0.8, edgecolors='k')
    plt.scatter(X_xor[neg_mask, 0], X_xor[neg_mask, 1], 
               c='red', marker='s', s=50, label='Class -1', alpha=0.8, edgecolors='k')
    
    # Highlight misclassified points
    y_pred_all = svm_model.predict(X_xor)
    misclassified = y_xor != y_pred_all
    if np.any(misclassified):
        plt.scatter(X_xor[misclassified, 0], X_xor[misclassified, 1], 
                   s=200, facecolors='none', edgecolors='orange', linewidths=3,
                   label=f'Misclassified ({np.sum(misclassified)})')
    
    plt.title('Linear SVM Attempt\n(Fails with Linear Boundary)', fontsize=14)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=False)

def create_synthetic_datasets():
    """
    Create synthetic datasets that demonstrate the need for non-linear kernels.
    """
    datasets = {}
    n_samples = 200
    
    # 1. XOR-like problem (requires non-linear kernel)
    np.random.seed(42)
    X_xor = np.random.randn(n_samples, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)
    datasets['XOR'] = (X_xor, y_xor)
    
    # 2. Concentric circles (perfect for RBF kernel)
    X_circles, y_circles = make_circles(n_samples=n_samples, noise=0.1, factor=0.3, random_state=42)
    y_circles = np.where(y_circles == 0, -1, 1)
    datasets['Circles'] = (X_circles, y_circles)
    
    # 3. Half moons (good for polynomial kernel)
    X_moons, y_moons = make_moons(n_samples=n_samples, noise=0.15, random_state=42)
    y_moons = np.where(y_moons == 0, -1, 1)
    datasets['Moons'] = (X_moons, y_moons)
    
    # 4. Spiral data (challenging for all kernels)
    n_points = n_samples // 2
    theta = np.sqrt(np.random.rand(n_points)) * 2 * np.pi
    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    r_b = -2 * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    
    X_spiral = np.vstack((data_a, data_b))
    y_spiral = np.hstack((np.ones(n_points), -np.ones(n_points)))
    
    # Add noise
    X_spiral += np.random.normal(0, 0.3, X_spiral.shape)
    datasets['Spiral'] = (X_spiral, y_spiral)
    
    return datasets

def visualize_synthetic_datasets(datasets):
    """
    Visualize the synthetic datasets to understand their complexity.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Synthetic Datasets for Kernel Demonstration', fontsize=16)
    
    dataset_names = list(datasets.keys())
    
    for i, name in enumerate(dataset_names):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        X, y = datasets[name]
        
        pos_mask = y == 1
        neg_mask = y == -1
        
        ax.scatter(X[pos_mask, 0], X[pos_mask, 1], 
                  c='red', marker='o', s=50, label='Class +1', alpha=0.7)
        ax.scatter(X[neg_mask, 0], X[neg_mask, 1], 
                  c='blue', marker='s', s=50, label='Class -1', alpha=0.7)
        
        ax.set_title(f'{name} Dataset', fontsize=14)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        descriptions = {
            'XOR': 'Linear kernel will fail\n(Not linearly separable)',
            'Circles': 'Perfect for RBF kernel\n(Radial decision boundary)',
            'Moons': 'Good for polynomial kernel\n(Curved decision boundary)',
            'Spiral': 'Challenging for all kernels\n(Complex non-linear pattern)'
        }
        ax.text(0.05, 0.95, descriptions[name], transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show(block=False)

def compare_kernels_on_synthetic_data(SVMClass, datasets, kernels, kernel_params, optimal_gammas):
    """
    Compare different kernels on synthetic datasets using proper SVM implementation.
    """
    results = {}
    
    for dataset_name, (X, y) in datasets.items():
        print(f"\n{'='*60}")
        print(f"TESTING KERNELS ON {dataset_name.upper()} DATASET")
        print('='*60)
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results[dataset_name] = {}
        
        for kernel in kernels:
            print(f"\n--- {kernel.upper()} KERNEL ({dataset_name}) ---")
            
            # Create and train SVM
            current_kernel_params = kernel_params[kernel].copy()
            
            # Handle rbf_tight as rbf kernel with different gamma
            actual_kernel = 'rbf' if kernel == 'rbf_tight' else kernel
            
            # Use dataset-specific optimal gamma for RBF kernels
            if actual_kernel == 'rbf' and dataset_name in optimal_gammas:
                if kernel in optimal_gammas[dataset_name]:
                    current_kernel_params['gamma'] = optimal_gammas[dataset_name][kernel]
                    print(f"Using optimized gamma: {current_kernel_params['gamma']}")
            
            # Enhanced training parameters for better convergence
            enhanced_params = {
                'learning_rate': 0.01,
                'n_epochs': 1000,  # Reduced epochs for proper implementation
                # 'tolerance': 1e-6
            }
            
            svm = SVMClass(
                kernel=actual_kernel, 
                C=1.0, 
                **enhanced_params,
                gamma=current_kernel_params.get('gamma', 1.0),
                degree=current_kernel_params.get('degree', 3),
                coef0=current_kernel_params.get('coef0', 1)
            )
            svm.fit(X_train, y_train)
            
            # Evaluate
            y_pred = svm.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"  Test Accuracy: {accuracy:.4f}")
            
            results[dataset_name][kernel] = {
                'model': svm,
                'accuracy': accuracy,
                'X_train': X_train,
                'y_train': y_train
            }
    
    return results

def plot_kernel_comparison(results, datasets):
    """
    Create comprehensive visualizations comparing kernel performance.
    """
    dataset_names = list(results.keys())
    kernels = ['linear', 'poly', 'rbf', 'rbf_tight']
    
    num_datasets = len(dataset_names)
    num_kernels = len(kernels)

    fig, axes = plt.subplots(num_datasets, num_kernels, figsize=(6 * num_kernels, 5 * num_datasets), squeeze=False)
    
    for i, dataset_name in enumerate(dataset_names):
        for j, kernel_name in enumerate(kernels):
            ax = axes[i, j]
            
            # Get data and model
            kernel_data = results[dataset_name].get(kernel_name)
            if not kernel_data:
                ax.set_title(f"{dataset_name} - {kernel_name.upper()}\nData N/A")
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue

            svm = kernel_data['model']
            X_train = kernel_data['X_train']
            y_train = kernel_data['y_train']
            accuracy = kernel_data['accuracy']
            
            # Create mesh grid
            h = 0.05 # Adjusted for performance on potentially larger ranges
            x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
            y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
            
            # Compute decision function on mesh
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            Z = svm.decision_function(mesh_points)
            Z = Z.reshape(xx.shape)
            
            # Plot decision regions
            ax.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdBu_r')
            
            # Plot decision boundary
            ax.contour(xx, yy, Z, levels=[0], colors='black', linestyles='-', linewidths=2)
            
            # Plot data points
            pos_mask = y_train == 1
            neg_mask = y_train == -1
            ax.scatter(X_train[pos_mask, 0], X_train[pos_mask, 1], 
                      c='red', marker='o', s=30, label='Class +1', edgecolors='black', alpha=0.8)
            ax.scatter(X_train[neg_mask, 0], X_train[neg_mask, 1], 
                      c='blue', marker='s', s=30, label='Class -1', edgecolors='black', alpha=0.8)
            
            # Plot support vectors
            if hasattr(svm, 'support_vectors') and svm.support_vectors is not None and len(svm.support_vectors) > 0:
                ax.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], 
                          s=100, facecolors='none', edgecolors='green', linewidths=2, label='Support Vectors')
            
            # Create descriptive title for different kernel variants
            if kernel_name in ['rbf', 'rbf_tight'] and hasattr(svm, 'gamma'):
                kernel_display = f'RBF (γ={svm.gamma})'
            elif kernel_name == 'poly' and hasattr(svm, 'degree'):
                kernel_display = f'POLY (deg={svm.degree})'
            else:
                kernel_display = kernel_name.upper()
            
            ax.set_title(f'{dataset_name} - {kernel_display} Kernel\nAccuracy: {accuracy:.3f}')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            if j == 0 and i == 0: # Add legend only once or per row if needed
                ax.legend(fontsize='small')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    fig.suptitle("Kernel Performance on Synthetic Datasets", fontsize=18, y=0.99)
    plt.show(block=False)
    
    # Summary table
    print("\n" + "="*80)
    print("KERNEL PERFORMANCE SUMMARY")
    print("="*80)
    header = f"{'Dataset':<15} " + " ".join([f'{k.upper():<12}' for k in kernels]) + f"{'Best Kernel':<15}"
    print(header)
    print("-" * len(header))
    
    for dataset_name in dataset_names:
        line = f"{dataset_name:<15} "
        accuracies = []
        for kernel_name in kernels:
            acc = results[dataset_name].get(kernel_name, {}).get('accuracy', 0.0)
            accuracies.append(acc)
            line += f"{acc:<12.3f} "
        
        best_kernel_idx = np.argmax(accuracies) if accuracies else -1
        if best_kernel_idx != -1:
            best_kernel_raw = kernels[best_kernel_idx]
            # Display name for summary table
            if best_kernel_raw == 'rbf_tight':
                best_kernel_name = "RBF(High γ)"
            elif best_kernel_raw == 'rbf':
                best_kernel_name = "RBF(Low γ)"
            else:
                best_kernel_name = best_kernel_raw.upper()
        else:
            best_kernel_name = "N/A"
        line += f"{best_kernel_name:<15}"
        print(line)