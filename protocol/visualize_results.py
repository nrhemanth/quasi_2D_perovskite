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

class VisualizeResults:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.csv_files = self._get_csv_files()
        self.filtered_data = {}
        self.scores = []
        self.centers = [520, 570, 615, 655, 680, 700, 715, 727, 736, 743, 749, 775 ]
        #print("Loaded files:", self.csv_files)
    
    def _get_csv_files(self):
        file_names = os.listdir(self.folder_path)
        csv_files = [
            f for f in file_names
            if f.strip().lower().endswith('.csv') and 'front' not in f.lower()
        ]
        return sorted(csv_files, key=lambda x: int(re.search(r'\d+', x).group()))

    def _filter_file(self, file_name, min_range=500, max_range=850):
        path_csv = os.path.join(self.folder_path, file_name)
        df = pd.read_csv(path_csv)
        df.columns = ['Wavelength', 'Intensity']

        index_min = df['Wavelength'].sub(min_range).abs().idxmin()
        index_max = df['Wavelength'].sub(max_range).abs().idxmin()
        if index_min > index_max:
            index_min, index_max = index_max, index_min

        filtered_df = df.iloc[index_min:index_max].copy()
        filtered_df['Filtered_Intensity'] = filtered_df['Intensity']
        
        wavelength = filtered_df['Wavelength'].to_numpy()
        intensity = filtered_df['Filtered_Intensity'].to_numpy()

        return wavelength, intensity
    
    def _extract_all_filtered_data(self,sample_groups=None):
        '''
        Extract all the data and store in self.filtered_data in a dictionary.
        '''
        if sample_groups is not None:
            # Flatten the list of sample groups into a single list of all sample names
            sample_names = [sample for group in sample_groups for sample in group]
            target_files = [f for f in self.csv_files if any(name in f for name in sample_names)] # all the names of the files 
        else:
            target_files = self.csv_files

        for file in target_files:
            wavelength, intensity = self._filter_file(file)
            self.filtered_data[file] = (wavelength, intensity) # Store in dictionary with file name as key

    def get_filtered_data(self, sample_groups=None): 
        self._extract_all_filtered_data(sample_groups=sample_groups)
        return self.filtered_data

    def plot_sample_groups(self, sample_groups, fig_size = (3,2) ,show_n=False):
        """
        Plot selected sample groups on subplots.
        Each group should be a list of substrings to match (e.g. ['sample 1', 'sample 2']).
        """

        fig, axes = plt.subplots(fig_size[0], fig_size[1], figsize=(15, 15))
        all_files = list(self.filtered_data.keys())

        for i, ax in enumerate(axes.flatten()):
            if i >= len(sample_groups):
                break
            group = sample_groups[i]

            # Find files that match this group
            matching_files = [f for f in all_files if any(label in f for label in group)]

            for file_name in matching_files:
                wavelength, intensity = self.filtered_data[file_name]
                ax.plot(wavelength, intensity, label=file_name[:15])

            ax.set_title(f'Sample Group {i + 1}')
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Filtered Intensity')
            # ax.legend()
            ax.grid(True)

            if show_n:
                self.quantum_wells(ax=ax, plot=True)  # ✅ call quantum_wells_plot for each subplot
        plt.tight_layout()
        return axes
    
    def plot_samples_number(self, substring, show_n=False):
        """
        Plot all data files whose filename contains a given substring on one figure.
        """

        # 🔥 Load data if not loaded yet
        if not self.filtered_data:
            self.get_filtered_data()

        all_files = list(self.filtered_data.keys())

        # Case-insensitive matching
        matching_files = [f for f in all_files if substring.lower() in f.lower()]

        # If nothing found
        if not matching_files:
            print(f"No files found containing '{substring}'.")
            return None

        plt.figure(figsize=(8, 5))

        for file_name in matching_files:
            wavelength, intensity = self.filtered_data[file_name]
            plt.plot(wavelength, intensity, label=file_name[:24])  # truncate for clean legend

        plt.title(f"PL Spectra for files containing '{substring}'")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Filtered Intensity")
        plt.grid(True)
        plt.legend(loc='best')

        # Optional overlay of quantum well peaks
        if show_n:
            self.quantum_wells(ax=plt.gca(), plot=True)

        plt.tight_layout()
        plt.show()


