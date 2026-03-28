import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import ipywidgets as widgets
from ipywidgets import interact
from matplotlib import cm
from matplotlib.colors import Normalize
from botorch.utils.transforms import normalize, unnormalize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .gp_bo import GaussianProcess

class PlotGP: 
    def __init__(self, gp_class: GaussianProcess, bounds=None):
        self.gp_class = gp_class
        self.bounds = bounds if bounds is not None else torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        self.gp_model = gp_class.get_model()
        self.train_X = gp_class.train_X
        self.train_y = gp_class.train_y
        self.dtype = torch.float64

    def plot_gp(self, x=None, y=None, num_points=100):
        if x is None:
            x = np.linspace(self.bounds[0, 0], self.bounds[1, 0], num_points)
        if y is None:
            y = np.linspace(self.bounds[0, 1], self.bounds[1, 1], num_points)

        X_test = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        mean, std_dev = self.gp_model.predict(X_test)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(X_test[:, 0], mean, 'r-', label='GP Mean')
        ax.fill_between(X_test[:, 0], mean - 2 * std_dev, mean + 2 * std_dev, alpha=0.5, label='Confidence Interval')
        ax.scatter(self.train_X[:, 0], self.train_y, c='blue', marker='x', label='Training Data')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        # ax.legend()
        plt.show()

    def interactive_3d_staircase_contours(x_grid, y_grid, z_slices, slice_values,labels,title="3D Staircase Contour", cmap='viridis', minmax = None):
        """
        Plots stacked 2D contour plots to simulate a 3D staircase of contours with interactive rotation.

        Parameters:
            x_grid (2D array): meshgrid of x values
            y_grid (2D array): meshgrid of y values
            z_slices (list of 2D arrays): function values at each slice (z-level)
            slice_values (list): the z-axis or staircase values (e.g. time, pressure, etc.)
            title (str): Title of the plot
            cmap (str): Colormap for the contours
        """
        def plot_with_rotation(azim, elev):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(projection='3d')

            if minmax is not None: 
                global_min = minmax[0]
                global_max = minmax[1]
            else: 
                # Get global min and max across all slices
                global_min = min(np.min(z) for z in z_slices)
                global_max = max(np.max(z) for z in z_slices)

            levels = np.linspace(global_min, global_max, 15)
            norm = Normalize(vmin=global_min, vmax=global_max)

            for i, z_val in enumerate(slice_values):
                z_data = z_slices[i]
                ax.contourf(
                    x_grid, y_grid, z_data, 
                    zdir='z', offset=z_val, 
                    levels=levels, cmap=cmap, alpha=0.8, norm=norm
                )

            # Add a fake ScalarMappable for colorbar
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array([])  # needed for colorbar
            fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='Intensity')

            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_zlabel(labels[2])
            ax.set_zlim(min(slice_values), max(slice_values))
            ax.set_title(title)
            ax.view_init(elev=elev, azim=azim)
            plt.tight_layout()
            plt.show()

        interact(
            plot_with_rotation,
            azim=widgets.FloatSlider(value=45, min=0, max=360, step=1, description='Azimuth'),
            elev=widgets.FloatSlider(value=30, min=-90, max=90, step=1, description='Elevation')
        )

    def data_prep(self, df, x1, x2, x3, x4):
        x1 = torch.tensor(df[x1].values, dtype=torch.float32).reshape(-1, 1)
        x2 = torch.tensor(df[x2].values, dtype=torch.float32).reshape(-1, 1)
        x3 = torch.tensor(df[x3].values, dtype=torch.float32).reshape(-1, 1)
        x4 = torch.tensor(df[x4].values, dtype=torch.float32).reshape(-1, 1)

        train_x = torch.hstack([x1, x2, x3, x4])
        train_y = torch.tensor(df['yield product'].values, dtype=torch.float32).reshape(-1, 1)
        train_yvar = torch.tensor(df['var yield'].values, dtype=torch.float32).reshape(-1, 1)

        norm_x = normalize(train_x, self.bounds)
        return norm_x, train_y, train_yvar

    def generate_input_data(self, A, B, c, d, combination, dtype=torch.float32):
        if combination == ('Temperature', 'Anneal Time', 'R BAAc'):
            return torch.tensor([[A[i, j], d, B[i, j], c] for i in range(A.shape[0]) for j in range(A.shape[1])], dtype=dtype)
        elif combination == ('Temperature', 'Anneal Time', 'R MAI'):
            return torch.tensor([[A[i, j], d, c, B[i, j]] for i in range(A.shape[0]) for j in range(A.shape[1])], dtype=dtype)
        elif combination == ('R BAAc', 'R MAI', 'Temperature'):
            return torch.tensor([[d, A[i, j], B[i, j], c] for i in range(A.shape[0]) for j in range(A.shape[1])], dtype=dtype)
        elif combination == ('R BAAc', 'R MAI', 'Anneal Time'):
            return torch.tensor([[A[i, j], c, B[i, j], d] for i in range(A.shape[0]) for j in range(A.shape[1])], dtype=dtype)
        else:
            raise ValueError(f"Unsupported combination: {combination}")

    def gp_eval(self, test_x):
        self.gp_model.eval()
        with torch.no_grad():
            posterior = self.gp_model.posterior(test_x)

        mean = posterior.mean.squeeze().numpy()
        var = posterior.variance.squeeze().numpy()
        return mean, var

    def create_slices(self, A, B, c_slices, d_fixed, combination):
        mean_values = []
        var_values = []
        for c in c_slices:
            input_data = self.generate_input_data(A, B, c, d_fixed, combination)
            norm_input = normalize(input_data, self.bounds)
            mean, var = self.gp_eval(norm_input)
            mean_values.append(mean.reshape(A.shape))
            var_values.append(var.reshape(A.shape))
        return mean_values, var_values

    # Function to generate input data based on variable combination
    def generate_input_data_2D(self, A, B, c, d, combination):
        if combination == ('R BAAc', 'R MAI'):
            return torch.tensor(np.array([[A[i, j], B[i, j], c, d] for i in range(A.shape[0]) for j in range(A.shape[1])]), dtype=self.dtype)
        elif combination == ('Temperature', 'Anneal Time'):
            return torch.tensor(np.array([[A[i, j], c, B[i, j], d] for i in range(A.shape[0]) for j in range(A.shape[1])]), dtype=self.dtype)        
        elif combination == ('R MAI', 'Temperature'):
            return torch.tensor(np.array([[A[i, j], d, B[i, j], c ] for i in range(A.shape[0]) for j in range(A.shape[1])]), dtype=self.dtype)
        elif combination == ('R BAAc', 'Temperature'):
            return torch.tensor(np.array([[d, A[i, j], B[i, j], c] for i in range(A.shape[0]) for j in range(A.shape[1])]), dtype=self.dtype)
        elif combination == ('R MAI', 'Anneal Time'):
            return torch.tensor(np.array([[d, A[i, j], c, B[i, j]] for i in range(A.shape[0]) for j in range(A.shape[1])]), dtype=self.dtype)
        elif combination == ('R BAAc', 'Anneal Time'):
            return torch.tensor(np.array([[d, c,A[i, j], B[i, j]] for i in range(A.shape[0]) for j in range(A.shape[1])]), dtype=self.dtype)


    def staircase_plot_with_histograms(self, A, B, c, d, centers, cmap_name='viridis'):
        
        num_features = self.train_X.shape[1]  # Assuming 4D input

        
        # Define all combinations of variables for triple pair plots
        variable_combinations = [
            ('Temperature', 'Anneal Time'),
            ('R BAAc', 'Anneal Time'),
            ('R MAI', 'Anneal Time'),
            ('R BAAc', 'Temperature'),
            ('R MAI', 'Temperature'),
            ('R BAAc', 'R MAI'),
        ]
        feature_names = ["Temperature", "Anneal Time", "R BAAc", "R MAI"]

        mean_values = []
        var_values = []
        for combination in variable_combinations:
            input_data = self.generate_input_data_2D(A, B, c, d, combination)
            mean, var =  self.gp_eval(input_data)
            mean_values.append(mean)
            var_values.append(var)

        num_points = A.shape[0]  # Assuming A and B are meshgrids with the same shape
    
        fig, axes = plt.subplots(num_features, num_features, figsize=(12, 10))
        count = 0
        for i in range(num_features):
            for j in range(num_features):
                ax = axes[i, j]
                if i < j:  # Upper triangle (leave empty)
                    ax.axis('off')
                elif i == j:  # Diagonal (histograms)
                    ax.hist(self.train_X[:, i], bins=10, color='gray', edgecolor='black', alpha=0.7)
                    ax.set_xlabel(feature_names[i])
                    ax.set_box_aspect(1)  # Set the aspect ratio to be equal (cube-shaped)
                    ax.set_ylabel("Frequency")
                else:  # Lower triangle (pairwise scatter + contours)
                    means = mean_values[count]
                    # Determine which features are plotted on x (A) and y (B)
                    x_feat, y_feat = variable_combinations[count]  # order: (x, y)
                    
                    # Map feature names to indices
                    feat_to_idx = {name: idx for idx, name in enumerate(feature_names)}
                    x_idx = feat_to_idx[x_feat]
                    y_idx = feat_to_idx[y_feat]
                    # Plot normalized centers for each sample as scatter points
                    # Plot normalized centers for each sample as scatter points
                    sc = ax.contourf(A, B, means.reshape(num_points, num_points), cmap=cmap_name,
                                    alpha=0.8, levels=10, vmax=1, vmin=0) #max(means)
                    
                    if centers is not None:
                        # Determine which features are being plotted on axes
                        x_feature = feature_names[j]
                        y_feature = feature_names[i]
                        # Map feature names to normalized column names in centers DataFrame
                        norm_map = {
                            "Temperature": "Temperature_norm",
                            "Anneal Time": "Anneal Time_norm",
                            "R BAAc": "R BAAc_norm",
                            "R MAI": "R MAI_norm"
                        }
                        x_col = norm_map[x_feature]
                        y_col = norm_map[y_feature]
                        # Scatter plot the centers
                        ax.scatter(centers[x_col], centers[y_col], c='red', marker='x', s=80, label='Center')

                    # Map normalized ticks back to unnormalized values using bounds
                    x_ticks = np.linspace(0, 1, 5)
                    y_ticks = np.linspace(0, 1, 5)
                    x_bounds = self.bounds[:, x_idx].numpy()
                    y_bounds = self.bounds[:, y_idx].numpy()
                    x_unnorm = x_ticks * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
                    y_unnorm = y_ticks * (y_bounds[1] - y_bounds[0]) + y_bounds[0]

                    ax.set_xticks(x_ticks)
                    ax.set_xticklabels([f"{val:.1f}" for val in x_unnorm])
                    ax.set_yticks(y_ticks)
                    ax.set_yticklabels([f"{val:.1f}" for val in y_unnorm])
                    ax.set_xlabel(feature_names[j])
                    ax.set_ylabel(feature_names[i])
                    fig.colorbar(sc, ax=ax, orientation='vertical')

                    ax.set_box_aspect(1)
                    count += 1
                plt.tight_layout()
        

    def sliced_4D_plotting(self, A, B, c_slices, d_fixed, combination, colormap, minmax=(0,1), plot_type = 'mean'):
        self.A = A
        self.B = B
        self.c_slices = c_slices
        self.d_fixed = d_fixed
        self.combination = combination
        
        if plot_type == 'mean':
            plot_type = 0
        elif plot_type == 'variance':
            plot_type = 1

        # Get Posterior mean values for each slice
        mean_vals = []
        for i in range(len(self.d_fixed)):
            mean = self.create_slices(self.A, self.B, self.c_slices, self.d_fixed[i], self.combination)[plot_type]
            mean_vals.append(mean)

        mean_vals1, mean_vals2, mean_vals3, mean_vals4, mean_vals5 = mean_vals

        global_min = minmax[0]
        global_max = max(np.max(m) for m in mean_vals1 + mean_vals2 + mean_vals3 + mean_vals4 + mean_vals5)

        # Determine which variable is fixed (not in combination)
        all_labels = ['Temperature', 'Anneal Time', 'R BAAc', 'R MAI']
        fixed_label = next(label for label in all_labels if label not in self.combination)


        # Create a new figure with subplots for each combination
        fig = make_subplots(rows=1, cols=5, subplot_titles=(f'{fixed_label}: 0', f'{fixed_label}: 0.25',f'{fixed_label}: 0.5', f'{fixed_label}: 0.75',f'{fixed_label}: 1.0'),
                        specs=[[{'type': 'surface'}, {'type': 'surface'},{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]])

        for i, (c, y_grid1, y_grid2, y_grid3, y_grid4, y_grid5) in enumerate(zip(self.c_slices, mean_vals1,mean_vals2,mean_vals3,mean_vals4,mean_vals5), start=1):
            fig.add_trace(go.Surface(
                x=self.A,
                y=self.B,
                z=c * np.ones_like(self.A),  # Z-coordinate for slicing
                surfacecolor=y_grid1,  # Use predicted `y` as contour
                colorscale=colormap,
                cmin=global_min,
                cmax=global_max,
                showscale=True if i == 1 else False,  # Show color scale only on the first slice
                #   colorbar_x=0.45,
                opacity=0.7
            ), row=1, col=1)
    
            fig.add_trace(go.Surface(
                x=self.A,
                y=self.B,
                z=c * np.ones_like(self.A),  # Z-coordinate for slicing
                surfacecolor=y_grid2,  # Use predictediance as contour
                cmin=global_min,
                cmax=global_max,
                colorscale=colormap,
                showscale=True if i == 1 else False,  # Show color scale only on the first slice
                #colorbar_x=0.45,
                opacity=0.7
            ), row=1, col=2)

            fig.add_trace(go.Surface(
                x=self.A,
                y=self.B,
                z=c * np.ones_like(self.A),  # Z-coordinate for slicing
                surfacecolor=y_grid3,  # Use predictediance as contour
                cmin=global_min,
                cmax=global_max,
                colorscale=colormap,
                showscale=True if i == 1 else False,  # Show color scale only on the first slice
                opacity=0.7
            ), row=1, col=3)

            fig.add_trace(go.Surface(
                x=self.A,
                y=self.B,
                z=c * np.ones_like(self.A),  # Z-coordinate for slicing
                surfacecolor=y_grid4,  # Use predictediance as contour
                cmin=global_min,
                cmax=global_max,
                colorscale=colormap,
                showscale=True if i == 1 else False,  # Show color scale only on the first slice
                opacity=0.7
            ), row=1, col=4)

            fig.add_trace(go.Surface(
                x=self.A,
                y=self.B,
                z=c * np.ones_like(self.A),  # Z-coordinate for slicing
                surfacecolor=y_grid5,  # Use predictediance as contour
                cmin=global_min,
                cmax=global_max,
                colorscale=colormap,
                showscale=True if i == 1 else False,  # Show color scale only on the first slice
                opacity=0.7
            ), row=1, col=5)
            

        fig.update_layout(
            height=400,
            width=1300,
            margin=dict(l=50, r=50, b=50, t=50),
            scene=dict(
                xaxis_title=self.combination[0],
                yaxis_title=self.combination[1],
                zaxis_title=self.combination[2]
            ),
            scene2=dict(
                xaxis_title=self.combination[0],
                yaxis_title=self.combination[1],
                zaxis_title=self.combination[2]
            ),
            scene3=dict(
                xaxis_title=self.combination[0],
                yaxis_title=self.combination[1],
                zaxis_title=self.combination[2]
            ),
            scene4=dict(
                xaxis_title=self.combination[0],
                yaxis_title=self.combination[1],
                zaxis_title=self.combination[2]
            ),
            scene5=dict(
                xaxis_title=self.combination[0],
                yaxis_title=self.combination[1],
                zaxis_title=self.combination[2]
            )
        )

        fig.show()
