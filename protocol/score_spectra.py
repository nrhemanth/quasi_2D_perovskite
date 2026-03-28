import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm  # use `tqdm` if not in a notebook
from scipy.signal import find_peaks, find_peaks_cwt
from lmfit.models import GaussianModel, LorentzianModel, ConstantModel, VoigtModel
from lmfit import Model, CompositeModel
from scipy.special import wofz
from collections import defaultdict
from tqdm.notebook import tqdm  # use `tqdm` if not in a notebook
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d

class ScoreSpectra:
    def __init__(self, folder_path, ):
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
    
    def quantum_wells(self, target_qw=None, ax=None, uncertainty=0, plot=False):
        '''
        Define quantum well ranges and optionally plot them.

        Args: 
            target_qw: int to specify which to quantum well to highlight.
            uncertainty: int to expand the range of each quantum well by this amount.
            ax: matplotlib axis to plot on (optional).
            plot: bool to specify whether to plot the quantum wells.

        Returns: 
            a dictionary of quantum well ranges.
        '''
        # Quantum well ranges based on the image you provided
        quantum_well_ranges = {
            1:  (515, 525),
            2:  (570, 580),
            3:  (615, 625),
            4:  (650, 660),
            5:  (675, 685),
            6:  (695, 705),
            7:  (710, 720),
            8:  (722, 732),
            9:  (731, 741),
            10: (738, 748),
            11: (744, 754),
            12: (749, 759),
            99: (760, 785) # 3D perovskite Made large on purpose
        }

        # Expand ranges by uncertainty (if provided)
        self. expanded_ranges = {
            k: (v[0] - uncertainty, v[1] + uncertainty)
            for k, v in quantum_well_ranges.items()
        }

        # Optional plot
        if plot and ax is not None:
            for n, (low, high) in self.expanded_ranges.items():
                color = 'red' if n == target_qw else 'grey'
                ax.axvspan(low, high, color=color, alpha=0.3, label=f'n={n}')

        return self.expanded_ranges
    
    def phase_purity_score(self, result, x_data, qw_regions, dominance_weight=0.5, asymmetry_weight=0.2):
        """
        Computes phase purity based on:
        - number of quantum wells present (fewer = better),
        - dominance of one phase over others (area-wise),
        - symmetry of the spectrum (more symmetric = better)

        Args:
            result: lmfit fit result
            x_data: wavelength array
            qw_regions: dict mapping n to (low, high) ranges
            dominance_weight: weight for how much dominance matters vs count
            asymmetry_weight: weight for how much asymmetry penalty reduces purity

        Returns:
            float: Purity score (higher is purer)
        """
        phase_areas = {}
        y_data = result.best_fit

        # Extract phase areas from fitted components
        for name, comp in result.eval_components().items():
            if not name.startswith('g'):
                continue
            prefix = name.strip('_')
            center_param = result.params.get(f"{prefix}_center")
            if center_param is None:
                continue

            # Fix: safely get the float value from Parameter or plain float
            center = center_param.value if hasattr(center_param, "value") else center_param

            for n, (low, high) in qw_regions.items():
                if low <= center <= high:
                    area = np.trapz(comp, x_data)
                    phase_areas[n] = phase_areas.get(n, 0) + area
                    break

        if not phase_areas:
            return 0

        # Phase dominance metrics
        total_area = sum(phase_areas.values())
        n_phases = len(phase_areas)
        max_area = max(phase_areas.values())
        dominance = max_area / total_area
        count_score = 1 / n_phases  # 1.0 if one phase, drops with more

        # --- Skew penalty from best_fit (filtered)
        y_smooth = gaussian_filter1d(y_data, sigma=2)
        spectral_skew = skew(y_smooth)
        skew_penalty = np.exp(-asymmetry_weight * abs(spectral_skew))  # 1 when symmetric, decays with skew

        # Combine
        base_score = (1 - dominance_weight) * count_score + dominance_weight * dominance
        final_score = base_score * skew_penalty
        return final_score


    def get_qw_assignment(self, center, qw_regions):
        for n, (low, high) in qw_regions.items():
            if low <= center <= high:
                return n
        return 'enter a valid quantum'

    def phase_distribution_score(self, result, target_n, x_data, qw_regions):
        '''
        Scores how much of the spectral area is assigned to the target QW phase.
        Returns 1 if all peaks are assigned to target_n; returns lower score if others are present.
        '''
        total_area = 0
        target_area = 0
        neighbor_area = 0  # optional: include ±1 phases
        for name, comp in result.eval_components().items():
            if name.startswith('g'):
                prefix = name.strip('_')
                center = result.params[f'{prefix}_center'].value
                n = self.get_qw_assignment(center, qw_regions)
                if n is None:
                    continue
                area = np.trapz(comp, x_data)
                total_area += area

                if n == target_n:
                    target_area += area
                elif abs(n - target_n) == 1:
                    neighbor_area += area  # assign partial credit to nearby phases

        if total_area == 0:
            return 0

        # Pure target phase
        if target_area / total_area >= 0.95:
            return 1.0

        # Weighted score: full credit for target, half credit for neighbors
        score = (target_area + 0.5 * neighbor_area) / total_area
        return score

    def phase_identification_score(self, result, x_data, target_n, qw_regions, method="amplitude"):
        """
        Computes the phase identification score — how well the target phase is present.
        Supports area- and amplitude-based variants.

        Args:
            result: lmfit fit result object
            x_data: x-values used for integration
            target_n: int, target quantum well
            qw_regions: dict mapping n -> (low, high)
            method: 'area' (default), 'amplitude', or 'area+amplitude'

        Returns:
            float: score between 0 and 1
        """
        if target_n not in qw_regions:
            return 0

        low, high = qw_regions[target_n]
        total_area = 0
        target_area = 0
        total_amp = 0
        target_amp = 0

        for name in result.eval_components().keys():
            if name.startswith("g"):
                prefix = name.strip("_")
                center = result.params.get(f"{prefix}_center").value
                sigma = result.params.get(f"{prefix}_sigma").value
                amp = result.params.get(f"{prefix}_amplitude").value

                component = result.eval_components()[name]
                area = np.trapz(component, x_data)

                total_area += area
                total_amp += amp

                if low <= center <= high:
                    target_area += area
                    target_amp += amp

        if method == "area":
            return target_area / total_area if total_area > 0 else 0
        elif method == "amplitude":
            return target_amp / total_amp if total_amp > 0 else 0
        elif method == "area+amplitude":
            a = target_area / total_area if total_area > 0 else 0
            b = target_amp / total_amp if total_amp > 0 else 0
            return 0.5 * a + 0.5 * b  # or adjust weights as needed
        else:
            raise ValueError("method must be 'area', 'amplitude', or 'area+amplitude'")

    def _voigt_profile(self, x, sigma, gamma):
        """Normalized Voigt profile (area = 1)."""
        z = (x + 1j*gamma) / (sigma*np.sqrt(2))
        return np.real(wofz(z)) / (sigma*np.sqrt(2*np.pi))

    def voigt_wavelet(self, M, s, alpha=0.7, k=10.0):
        """
        Voigt wavelet for find_peaks_cwt.

        Parameters
        ----------
        M : int
            Number of points in the wavelet (provided by find_peaks_cwt).
        s : float
            Scale (width) passed by find_peaks_cwt; interpret it ~ target FWHM in *samples*.
        alpha : float in [0,1]
            Fraction of Lorentzian broadening (0=Gaussian, 1=Lorentzian).
        k : float
            Half-width of the sampling window in units of s (controls truncation).

        Returns
        -------
        w : (M,) ndarray
            Symmetric, zero-mean, L2-normalized wavelet.
        """
        # Convert FWHM-like scale 's' into Gaussian sigma and Lorentzian HWHM gamma
        # FWHM_gauss = 2*sqrt(2*ln2)*sigma ;  FWHM_lorentz = 2*gamma
        # Blend them by alpha to keep overall scale ~ s
        fwhm = max(float(s), 1e-6)
        sigma = (1 - alpha) * fwhm / (2*np.sqrt(2*np.log(2))) + 1e-9
        gamma = alpha * fwhm / 2.0 + 1e-9

        # Sample a symmetric window ~ ±k*s
        x = np.linspace(-k*fwhm, k*fwhm, int(M))
        w = self._voigt_profile(x, sigma, gamma)

        # Make it a wavelet: zero-mean and L2-normalized (required by SciPy’s CWT peak finder)
        w = w - w.mean()
        norm = np.linalg.norm(w)
        if norm > 0:
            w = w / norm
        return w

    def fit_multiple_voigts_to_qw(self, filtered_data, widths=np.arange(1, 35), amp_threshold=0.05, sigma=3.5, uncertainty = 3):
        x_data = filtered_data.iloc[:, 0].values
        y_data = filtered_data.iloc[:, 1].values
        y_smooth = gaussian_filter1d(y_data, sigma=sigma)

        # Dominant peak check
        dom_peaks, _ = find_peaks(y_smooth, prominence=0.01, height=0.3)
        centers = self.quantum_wells(plot=False)

        dominant_in_qw = [any(low - uncertainty <= x_data[idx] <= high + uncertainty for (low, high) in centers.values()) for idx in dom_peaks]

        if sum(dominant_in_qw) > 1:
            try:
                #peak_idxs = find_peaks_cwt(y_smooth, widths)
                peak_idxs = find_peaks_cwt(y_smooth, widths=widths, wavelet=self.voigt_wavelet)
            except Exception as e:
                print(f"Peak detection failed: {e}")
                return None

            if len(peak_idxs) == 0:
                return None

            peak_idxs = np.array(peak_idxs)
            amplitudes = y_smooth[peak_idxs]
            peak_idxs = peak_idxs[amplitudes >= amp_threshold * np.max(y_smooth)]
            if len(peak_idxs) == 0:
                return None
        else:
            peak_idxs = find_peaks_cwt(y_smooth, widths)
            amplitudes = y_smooth[peak_idxs]
            peak_idxs = peak_idxs[amplitudes >= amp_threshold+0.2 * np.max(y_smooth)]

        # Fitting
        model = ConstantModel(prefix='c_')
        params = model.make_params(c_c=0)

        for i, (qw, (low, high)) in enumerate(centers.items()):
            low_bound = low - uncertainty
            high_bound = high + uncertainty
            in_region = [idx for idx in peak_idxs if low_bound <= x_data[idx] <= high_bound]
            if not in_region:
                continue
            peak_idx = in_region[0]
            amp_guess = y_data[peak_idx]
            center_guess = x_data[peak_idx]
            prefix = f'g{i+1}_' if i != 12 else 'g99_'

            voigt = VoigtModel(prefix=prefix)
            model += voigt
            params.update(voigt.make_params())
            params[f'{prefix}amplitude'].set(value=amp_guess, min=0)
            params[f'{prefix}center'].set(value=center_guess, min=low, max=high)
            params[f'{prefix}sigma'].set(value=8, min=0.5, max=23)
            params[f'{prefix}gamma'].set(value=5, min=0.1, max=30)
       
        if len(model.components) <= 1: # Now always return a fit result
            # Fallback to fitting dominant peak only
            peak_idx = np.argmax(y_smooth)
            amp_guess = y_data[peak_idx]
            center_guess = x_data[peak_idx]
            prefix = 'g_fallback_'

            voigt = VoigtModel(prefix=prefix)
            model += voigt
            params.update(voigt.make_params())
            params[f'{prefix}amplitude'].set(value=amp_guess, min=0)
            params[f'{prefix}center'].set(value=center_guess, min=x_data[0], max=x_data[-1])
            params[f'{prefix}sigma'].set(value=8, min=40, max=0.5)
            params[f'{prefix}gamma'].set(value=5, min=15, max=1)

        return model.fit(y_data, params, x=x_data)
        # return model.fit(y_data, params, x=x_data) if len(model.components) > 1 else None

    def sequential_scoring(self, filtered_data, widths=np.arange(1, 35), amp_threshold=0.05, sigma=3.5, uncertainty = 3):
        x_data = filtered_data.iloc[:, 0].values
        y_data = filtered_data.iloc[:, 1].values
        y_smooth = gaussian_filter1d(y_data, sigma=5)

        # Dominant peak check
        #dom_peaks, _ = find_peaks(y_smooth, prominence=0.01, height=0.05)
        centers = self.quantum_wells(plot=False)
        # Are dominant peaks in the QW regions?
        #dominant_in_qw = [any(low - uncertainty <= x_data[idx] <= high + uncertainty for (low, high) in centers.values()) for idx in dom_peaks]

        try:
            #peak_idxs = find_peaks_cwt(y_smooth, widths)
            peak_idxs = find_peaks_cwt(y_smooth, widths=widths, wavelet=self.voigt_wavelet)
        except Exception as e:
            print(f"Peak detection failed: {e}")
            return None

        if len(peak_idxs) >  1: # Not phase pure
            # Fitting
            peak_idxs = np.array(peak_idxs)
            amplitudes = y_smooth[peak_idxs]
            peak_idxs = peak_idxs[amplitudes >= amp_threshold * np.max(y_smooth)]

            model = ConstantModel(prefix='c_')
            params = model.make_params(c_c=0)
            for i, (qw, (low, high)) in enumerate(centers.items()):
                low_bound = low - uncertainty
                high_bound = high + uncertainty
                in_region = [idx for idx in peak_idxs if low_bound <= x_data[idx] <= high_bound]
                if not in_region:
                    continue
                peak_idx = in_region[0]
                amp_guess = gaussian_filter1d(y_data, sigma=sigma)[peak_idx]
                center_guess = x_data[peak_idx]
                prefix = f'g{i+1}_' if i != 12 else 'g99_'

                voigt = VoigtModel(prefix=prefix)
                model += voigt
                params.update(voigt.make_params())
                params[f'{prefix}amplitude'].set(value=amp_guess, min=0)
                params[f'{prefix}center'].set(value=center_guess, min=low, max=high)
                params[f'{prefix}sigma'].set(value=8, min=23, max=1)
                params[f'{prefix}gamma'].set(value=5, min=15, max=1)

            self.result = model.fit(gaussian_filter1d(y_data, sigma=2.5), params, x=x_data)

        return self.result
            
    def fit_all_spectra(self,amp_threshold=0.05, sigma=3.5, uncertainty = 3): 
        """
        Fit Vigot peaks to all filtered data in the folder, with progress bar.
        """
        self.results = {}

        for file_name, (wavelength, intensity) in tqdm(self.filtered_data.items(), desc="Fitting spectra"):
            filtered_set = pd.DataFrame({'Wavelength': wavelength, 'Intensity': intensity})
            #result = self.fit_multiple_gaussians_to_qw(filtered_set)
            #result = self.fit_multiple_voigts_to_qw(filtered_set) # Use Voigt model for fitting
            result = self.sequential_scoring(filtered_set, amp_threshold=amp_threshold, sigma=sigma, uncertainty = uncertainty)
            self.results[file_name] = result
        return self.results
    
    def extract_scores(self, result, filtered_data, qw_regions, target, uncertainty=3, amp_threshold=0.05):
        """
        Returns (peak_score, amp_score, distance_score, final_score).
        If sequential_scoring returns 0 => (0,0,0,0)
        If sequential_scoring returns 1 => (1,1,1,1)
        Else it uses the lmfit ModelResult to compute the four scores.
        """
        
        # Otherwise, treat synthesize_score as a lmfit.ModelResult
        x = filtered_data.iloc[:, 0].values

        # Is fit good?
        if result.rsquared < 0.70: # less than 70% variance fit....
            print(f'R^2 = {result.rsquared} --> Poor fit quality (R² < 0.3); returning zero scores.')
            return 0.0, 0.0, 0.0, 0.0

        low, high = qw_regions[target]
        target_center = (low + high) / 2.0
        # Collect peaks from fitted components
        comps = result.eval_components(x=x)
        peaks = []
        target_peak_found = False
        for name, comp in comps.items():
            if not name.startswith('g'):
                continue
            prefix = name.strip('_')
            cpar = result.params.get(f"{prefix}_center")
            apar = result.params.get(f"{prefix}_height") # height: This is the height of the actual
            areapar = result.params.get(f"{prefix}_amplitude") # amplitude: This is area under the curve of the, 

            if cpar is None or apar is None:
                continue
            center = cpar.value if hasattr(cpar, "value") else float(cpar)
            amp    = apar.value if hasattr(apar, "value") else float(apar)
            area   = float(np.trapz(comp, x))
            peaks.append({"center": center, "amp": amp, "area": area})

            if low - uncertainty <= center <= high + uncertainty:
                target_peak_found = True

        if not target_peak_found:
            return  0.0, 0.0, 0.0, 0.0

        if len(peaks) == 0:
            return  0.0, 0.0, 0.0, 0.0
        
        # --- Peak count score: 1 for one peak, -> 0 for 13 peaks (cap to [0,1])
        # Only count peaks with amplitude > 0.001 * max intensity (0.1%)
        max_intensity = max([p["amp"] for p in peaks]) if peaks else 0
        n_peaks = len([p for p in peaks if p["amp"] > amp_threshold * max_intensity]) 
        peak_score = 1.0 - (n_peaks - 1) / 12.0
        peak_score = float(np.clip(peak_score, 0.0, 1.0))

        # --- Dominance score: target peak vs next highest non-target
        target_idx = int(np.argmin([abs(p["center"] - target_center) for p in peaks]))
        target_amp = peaks[target_idx]["amp"]

        if n_peaks == 1:
            amp_score = 1.0 if target_amp > 0 else 0.0
        else:
            non_target_amps = [p["amp"] for i, p in enumerate(peaks) if i != target_idx]
            next_highest = max(non_target_amps) if non_target_amps else 0.0
            denom = target_amp + next_highest
            amp_score = float(target_amp / denom) if denom > 0 else 0.0

        # --- Distance score
        target_width = (high - low)/2
        # sum_dist = sum(abs(p["center"] - target_center) for p in peaks)
        sum_dist = sum(abs(p["center"] - target_center) for i, p in enumerate(peaks) if i != target_idx)
        distance_score = 1.0 / (1.0 + (sum_dist / target_width))

        # --- Final combined score
        final_score = float((peak_score + amp_score + distance_score) / 3.0)

        return peak_score, amp_score, distance_score, final_score

    def get_all_scores(self, sample_groups = None, amp_threshold=0.015):
        """
        Fit Voigt peaks to all filtered data and extract scores.
        Stores per-target amp, distance, and final scores.
        Peak score is computed once per file (target-independent).
        """
        self.all_results = {}
        qw_regions = self.quantum_wells()
        if sample_groups is not None: 
            # Flatten sample_groups to a list of all sample names
            sample_names = [sample for group in sample_groups for sample in group]
            selected_files = [(fname, self.filtered_data[fname]) for fname in self.filtered_data if any(name in fname for name in sample_names)]
        else: 
            selected_files = self.filtered_data.items()

        for file_name, (wavelength, intensity) in tqdm(selected_files, desc="Getting Score"):
            result = self.results[file_name]
            filtered_set = pd.DataFrame({'Wavelength': wavelength, 'Intensity': gaussian_filter1d(intensity, sigma=2.5)})

            amp_scores = []
            distance_scores = []
            final_scores = []
            peak_score = None

            for idx, target in enumerate(qw_regions.keys()):
                p_score, a_score, d_score, f_score = self.extract_scores(result, filtered_set, qw_regions, target, amp_threshold = amp_threshold)

                # Capture peak_score only from the first call since it’s target-independent
                if peak_score is None:
                    peak_score = p_score

                amp_scores.append(a_score)
                distance_scores.append(d_score)
                final_scores.append(f_score)

            self.all_results[file_name] = {
                'fit_result': result,
                'data': filtered_set,
                'peak_score': peak_score,
                'amp_scores': amp_scores,
                'distance_scores': distance_scores,
                'final_scores': final_scores,
            }

        return self.all_results

    def plot_gaussian_quantum_fits(self, ax, sample_name):
        data = self.all_results[sample_name]['data']
        result = self.all_results[sample_name]['fit_result']
        y_norm = data['Intensity'] / np.max(data['Intensity'])
        y_filt = gaussian_filter1d(data['Intensity'], sigma=1.5)
        ax.plot(data['Wavelength'], y_filt, label='Normalized Intensity', color='tab:gray', alpha=0.5)
        ax.plot(data['Wavelength'], data['Intensity'], label='Filtered Set', color='tab:blue')
        ax.plot(data['Wavelength'], result.best_fit, label='Total Fit', linestyle='--', color='tab:red')
        # Plot the components of the fit
        components = result.eval_components()
        for j, (name, comp) in enumerate(components.items()):
            # Find the peak position of each component for annotation
            peak_idx = np.argmax(comp)
            peak_x = data['Wavelength'].values[peak_idx]
            peak_y = comp[peak_idx]
            ax.plot(data['Wavelength'], comp, linestyle=':', label=f'G{result.components[j].prefix}')
            ax.annotate(f'{result.components[j].prefix}', xy=(peak_x, peak_y), xytext=(5, 5), textcoords='offset points',
                        fontsize=9, color='black', arrowprops=dict(arrowstyle='->', lw=0.5), ha='left')

        # Fill between vertical lines at x=760 and x=770
        # ax.axvspan(760, 780, color='k', alpha=0.2, label='n99')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Intensity')
        # ax.legend()
        plt.tight_layout()
        # plt.show()

    def plot_sample_groups_gaussian(self, sample_groups, fig_size=(4,3), show_n=False, title='Gaussian Fits by Group'):
        rows, cols = fig_size
        fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*3), sharex=True, sharey=False)
        axes = np.atleast_2d(axes)

        all_keys = list(self.results.keys())
        flat_axes = axes.ravel()

        # iterate over subplots (groups), NOT files
        for gi, ax in enumerate(flat_axes):
            if gi >= len(sample_groups):
                ax.axis('off')
                continue

            group = sample_groups[gi]  # e.g., ['S01','S02'] substrings
            matching_files = [k for k in all_keys if any(sub in k for sub in group)]
            if not matching_files:
                ax.text(0.5, 0.5, "No matching samples", ha='center', va='center',
                        transform=ax.transAxes, color='red')
                ax.set_title(f"Sample Group {gi + 1}")
                continue

            # ✅ draw each sample on THIS subplot
            for file_name in matching_files:
                self.plot_gaussian_quantum_fits(ax, file_name)

            ax.set_title(f"Sample Group {gi + 1}")
            ax.grid(True)

            if show_n and hasattr(self, "quantum_wells"):
                try:
                    centers = self.quantum_wells(plot=False)
                    for lo, hi in centers.values():
                        ax.axvline(lo, lw=0.6, alpha=0.5)
                        ax.axvline(hi, lw=0.6, alpha=0.5)
                except Exception:
                    pass

        if title:
            fig.suptitle(title, y=1.02)
        fig.tight_layout()
        return fig, axes

    def plot_scores_file(self,file_name):
        """
        Plot the purity, distribution, and identification scores for each target quantum well,
        for the specified file_name using self.all_results.
        The x-axis is the target quantum well (1-12, ∞ for 3D perovskite).
        """
        if file_name not in self.all_results:
            print(f"No results found for file: {file_name}")
            return

        res = self.all_results[file_name]
        purity_scores = np.array(res['purity_score'])
        distribution_scores = np.array(res['distribution_score'])
        identification_scores = np.array(res['identification_score'])

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, 14), purity_scores, marker='o', label='Purity Score', linestyle='-', alpha=0.7, color='k')
        plt.plot(range(1, 14), distribution_scores, marker='s', label='Distribution Score', linestyle='-', alpha=0.7)
        plt.plot(range(1, 14), identification_scores, marker='^', label='Identification Score', linestyle='-', alpha=0.7)
        plt.xlabel('Target Quantum Well (n)')
        plt.ylabel('Score')
        plt.title(f'Phase Scores vs Target Quantum Well\nFile: {file_name}')
        plt.legend()
        plt.grid(axis='x', which='both', linestyle='--', alpha=0.5)
        xtick_labels = [str(i) for i in range(1, 13)] + ['∞']
        plt.xticks(range(1, 14), xtick_labels)
        plt.tight_layout()
        plt.show()
    
    def plot_score_comparison(self, sample_groups=None):
        """
        Plot each quantum well score for each sample, with subplots for each scoring type.
        X-axis: sample number
        Y-axis: score
        Points: quantum well (n)
        """
        # Define QW phases including 99 for ∞
        all_n =  list(range(1, 13)) + [99]
        score_types = ['purity_score', 'distribution_score', 'identification_score']

        # Step 1: Filter relevant files
        if sample_groups:
            target_files = [f for f in self.all_results if any(group in f for sub in sample_groups for group in sub)]
        else:
            target_files = list(self.all_results.keys())

        sample_nums = []
        score_data = {score_type: {n: [] for n in all_n} for score_type in score_types}

        # Step 2: Gather data
        for fname in target_files:
            match = re.search(r'sample\s*(\d+)', fname)
            sample_num = int(match.group(1)) if match else fname
            sample_nums.append(sample_num)

            result_dict = self.all_results[fname]
            for score_type in score_types:
                score_dict = result_dict.get(score_type, {})
                for idx, n in enumerate(all_n):
                    try:
                        score_data[score_type][n].append(score_dict[n-1 if n != 99 else 12])  # assumes 13 values
                    except IndexError:
                        score_data[score_type][n].append(np.nan)

        # Step 3: Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
        # Use a colormap for distinct colors
        cmap = plt.get_cmap('tab20')
        color_list = [cmap(i % cmap.N) for i in range(len(all_n))]
        for ax, score_type in zip(axes, score_types):
            for idx, n in enumerate(all_n):
                y = score_data[score_type][n]
                label = f'n={n if n != 99 else "∞"}'
                if score_type == 'purity_score':
                    ax.scatter(sample_nums, y, s=60, color='k')
                else:
                    ax.scatter(sample_nums, y, label=label, s=60, color=color_list[idx])
                ax.set_xlabel('Sample Number')
                ax.set_ylabel('Score')
                ax.set_title(score_type.replace("_", " ").title())
                ax.grid(axis='x', which='major', linestyle='--', alpha=0.5)
                ax.set_xticks(sample_nums)

        axes[1].legend(title='Quantum Well', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def print_scores(self, name ,sample_groups=None, print_scores=True):
        '''
        name: in form r'gridN#_(\d+)_\d+\.csv' <-- check the rename format
        '''
        if sample_groups:
            target_files = [f for f in self.filtered_data if any(group in f for sub in sample_groups for group in sub)]
        else:
            target_files = self.filtered_data.keys()

        score_list = []
        score_dict = {}
        for fname in target_files:
            score = self.scores.get(fname, "N/A")
            score_list.append(score)
            score_dict[fname] = score
            #print(f"{fname}: Score = {score:.3f}" if isinstance(score, float) else f"{fname}: Score = {score}")

        # Group by the middle number
        grouped = defaultdict(list)

        for key, val in score_dict.items():
            match = re.search(name, key)
            if match:
                sample_num = int(match.group(1))
                grouped[sample_num].append((key, val))

        # Sort each group by filename or value (optional)
        for k in grouped:
            grouped[k] = sorted(grouped[k], key=lambda x: x[0])  # Sort by filename
            # grouped[k] = sorted(grouped[k], key=lambda x: x[1])  # Or by value

        if print_scores:
            # To print them nicely:
            for sample_num in sorted(grouped):
                print(f"Sample {sample_num}:")
                for fname, val in grouped[sample_num]:
                    print(f"  {fname}: {val:.4f}")
        
        return grouped

    def get_scores_df(self,sample_groups=None):
        if sample_groups:
            target_files = [f for f in self.filtered_data if any(group in f for sub in sample_groups for group in sub)]
        else:
            target_files = self.filtered_data.keys()

        score_list = []
        for fname in target_files:
            score = self.scores.get(fname, "N/A")
            score_list.append(score)
            print(f"{fname}: Score = {score:.3f}" if isinstance(score, float) else f"{fname}: Score = {score}")

        # Remove '.csv' from filenames for display
        filenames = [fname[:-4] if fname.endswith('.csv') else fname for fname in target_files]
        score_df = pd.DataFrame({'Filename': filenames, 'Score': score_list})
        return score_df

    def save_scores(self, data_path):
        '''
        output_path: path of folder to save the scores + the name of the file
        data_path: path of the file containing the parameters
        '''
        df = pd.read_csv(data_path)
        df['Score'] = self.scores.values()

        df.to_csv(data_path, index=False)
        print("Scores saved to",data_path)
        return df

    # def plot_scores(self,param_df):
    #     fig, axes = plt.subplots(figsize=(10, 6))
    #     dimensions = ['Anneal Time', 'BAAc', 'MAI']
    #     for i, ax in enumerate(axes.flatten()):
    #         if i < len(dimensions):
    #             ax.scatter(param_df[dimensions[i]], param_df['Mean Score'], c=param_df['Mean Score'], cmap='viridis', s=100)
    #             ax.set_title(f'{dimensions[i]} vs Score')
    #             ax.set_xlabel(dimensions[i])
    #             ax.set_ylabel('Score')
    #             ax.set_box_aspect(1)
    #     else:
    #         fig.delaxes(ax)  # Remove the unused plot

    #     plt.tight_layout()
    #     plt.show()