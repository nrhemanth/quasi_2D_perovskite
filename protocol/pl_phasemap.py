import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm  # use `tqdm` if not in a notebook
from scipy.signal import find_peaks, find_peaks_cwt
from scipy.special import wofz
from collections import defaultdict
from tqdm.notebook import tqdm  # use `tqdm` if not in a notebook
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d
from visualize_results import VisualizeResults

class PhasePlots:    
    def __init__(self, csv_path, x_label=None, y_label=None):
        """
        Parameters
        ----------
        csv_path : str
            Path to spectra CSV file.
        x_label, y_label : str, optional
            Labels for the reagent concentration axes.
        """
        self.csv_path = csv_path
        self.x_label = x_label
        self.y_label = y_label
        
        # Data containers
        self.wavelengths = None
        self.spectra = {}  # Dict {(i,j): spectrum_array}
        self.X_values = None  # sorted x concentrations
        self.Y_values = None  # sorted y concentrations
        
        # Appearance defaults
        self.scale_x = 0.3
        self.scale_y = 0.7
        self.color_map_function = None  # user-defined, optional
        
        # Load data
        self._load_csv_placeholder()  # replaced once format confirmed
    
    
    # ----------------------------------------------------
    # CSV LOADING (This will be replaced once format confirmed)
    # ----------------------------------------------------
    def _load_csv_placeholder(self):
        print("\n⚠ CSV format not yet specified (A/B/C).")
        print("➡ Please reply 'A', 'B', or 'C' from previous examples.\n")
    
    
    # ----------------------------------------------------
    # User option: specify a function to color spectra
    # ----------------------------------------------------
    def set_color_function(self, func):
        """
        func(i, j, spectrum) → matplotlib color
        
        Example:
        plotter.set_color_function(lambda i,j,s: "red" if max(s)>0.5 else "blue")
        """
        self.color_map_function = func
    
    
    # ----------------------------------------------------
    # Plot method
    # ----------------------------------------------------
    def plot_grid(self, mark_point=None, figsize=(9,7)):
        """
        Parameters
        ----------
        mark_point : (x,y) tuple to mark (optional)
        figsize : tuple
        """
        if self.wavelengths is None or not self.spectra:
            raise ValueError("No spectra loaded. CSV format not set.")
        
        plt.figure(figsize=figsize)
        ax = plt.gca()
        
        for i, y_val in enumerate(self.Y_values):
            for j, x_val in enumerate(self.X_values):
                spectrum = self.spectra[(i,j)]
                
                # Normalize vertically to small inset
                s = (spectrum - spectrum.min())/(spectrum.max()-spectrum.min()+1e-9) * self.scale_y
                
                # Assign color
                if self.color_map_function:
                    color = self.color_map_function(i, j, spectrum)
                else:
                    color = 'black'
                
                # Plot shifted mini-spectrum
                plt.plot(x_val + self.wavelengths/self.wavelengths.max()*self.scale_x,
                         y_val + s,
                         color=color,
                         lw=1)
        
        if mark_point:
            plt.scatter(mark_point[0], mark_point[1], marker='x', s=200, color='black', linewidths=3)
        
        if self.x_label:
            plt.xlabel(self.x_label)
        if self.y_label:
            plt.ylabel(self.y_label)
        
        plt.xlim(min(self.X_values) - 0.3, max(self.X_values) + 0.7)
        plt.ylim(min(self.Y_values) - 0.3, max(self.Y_values) + 0.7)
        plt.tight_layout()
        plt.show()


