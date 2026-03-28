import os
import re
import ast
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union
import difflib


import os
from typing import List, Union


class FileLocator:
    """
    Locate files inside a campaign data directory, where files are stored in
    subfolders and need to be matched by prefix (e.g., first 6 characters).
    """

    def __init__(self, base_dir: str, prefix_len: int = 6):
        """
        Parameters
        ----------
        base_dir : str
            Path to directory containing subfolders (e.g., ../N1_Campaign/Data/)
        prefix_len : int
            Number of characters from fname to match against folder names
        """
        self.base_dir = base_dir
        self.prefix_len = prefix_len

    def _find_matching_folder(self, fname_prefix: str) -> Union[str, None]:
        """
        Return first folder name in base_dir that contains the prefix.
        """
        for folder in os.listdir(self.base_dir):
            folder_path = os.path.join(self.base_dir, folder)
            if os.path.isdir(folder_path):
                if fname_prefix in folder.replace(" ", ""):
                    return folder  # return the folder name only
        return None
    
    def _extract_prefix(self, fname: str) -> str:
        """
        Extract a meaningful folder signature from a filename.
        Keeps '..._Subt2__0' or similar endings fully intact.
        """
        name = os.path.splitext(fname)[0]   # remove extension

        # Split before the timestamp (before last __)
        parts = name.split("__")

        # Keep everything except the timestamp block
        if len(parts) > 2:
            prefix = "__".join(parts[:-1])  # drop only the timestamp
        else:
            prefix = name

        return prefix.replace(" ", "")      # strip only spaces, not digits


    def locate_files(self, filenames: List[str]) -> List[str]:
        """
        Locate all files that belong to the filenames based on prefix matching.

        Parameters
        ----------
        filenames : List[str]
            List of sample file names (e.g. from top10['files'])

        Returns
        -------
        List[str]
            List of full resolved paths to the matching files.
        """
        resolved_paths = []

        for fname in filenames:
            prefix = fname[:self.prefix_len].replace(" ", "")

            # Find folder that contains prefix
            folder = self._find_matching_folder(prefix)

            if folder is None:
                print(f"[WARN] No folder found for prefix '{prefix}'")
                continue

            folder_path = os.path.join(self.base_dir, folder)
            candidate_path = os.path.join(folder_path, fname)

            if os.path.exists(candidate_path):
                resolved_paths.append(candidate_path)
                print(f"[OK] Found file: {candidate_path}")
            else:
                print(f"[WARN] File exists in folder '{folder}', but exact name not found:\n  {candidate_path}")

        return resolved_paths

    def locate_closest_files(self, filenames: List[str]) -> List[str]:
        """
        Locate CSV files by prefix. If exact CSV is missing, use closest CSV match.
        Only CSV files are considered.
        """
        resolved_paths = []

        for fname in filenames:
            # We assume user gives full filename, but we only match CSV version
            target_csv = fname + ".csv"
            
            prefix = fname[:self.prefix_len].replace(" ", "")

            # 1 — Find matching folder
            folder = self._find_matching_folder(prefix)
            if folder is None:
                print(f"[WARN] No folder found for prefix '{prefix}'")
                continue

            folder_path = os.path.join(self.base_dir, folder)

            # 2 — Try exact match
            candidate_path = os.path.join(folder_path, target_csv)
            if os.path.exists(candidate_path):
                resolved_paths.append(candidate_path)
                print(f"[OK] Exact CSV found: {candidate_path}")
                continue

            # 3 — No exact match → look for CSVs in the folder
            csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".csv")]

            if not csv_files:
                print(f"[WARN] No CSV files in folder '{folder}'")
                continue

            # 4 — Choose closest CSV by filename similarity
            closest_matches = difflib.get_close_matches(target_csv, csv_files, n=3, cutoff=0.5)

            if closest_matches:
                best = closest_matches[0]
                best_path = os.path.join(folder_path, best)
                resolved_paths.append(best_path)
                print(f"[~] Using closest CSV match:\n    {best}")
            else:
                print(f"[WARN] No close CSV match found in folder '{folder}' for '{target_csv}'")

        return resolved_paths


    # quick multi-panel plot of the located files' spectra
    # NEW METHOD: Multi-panel plotting of spectra
    def plot_spectra(self, file_paths: List[str], color = 'tab:blue',wavelength_min=480, wavelength_max=800):
        """
        Multi-panel plot for a list of spectrum files.

        Parameters
        ----------
        file_paths : List[str]
            Paths returned from locate_files()
        wavelength_min : float
            Minimum wavelength to keep
        wavelength_max : float
            Maximum wavelength to keep
        """

        n = len(file_paths)
        if n == 0:
            print("No files to plot.")
            return

        cols = min(5, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols,
                                 figsize=(cols * 3.5, rows * 3.5),
                                 squeeze=False)

        for i, fp in enumerate(file_paths):
            r, c = divmod(i, cols)
            ax = axes[r][c]

            try:
                # read .txt or .csv style spectra
                spec = pd.read_csv(
                    fp,
                    names=['wavelength', 'intensity'],
                    header=None,
                    comment='#'
                )

                spec['wavelength'] = pd.to_numeric(spec['wavelength'], errors='coerce')
                spec['intensity'] = pd.to_numeric(spec['intensity'], errors='coerce')

                spec = spec.dropna(subset=['wavelength', 'intensity'])

                # filter wavelengths
                spec = spec[(spec['wavelength'] >= wavelength_min) &
                            (spec['wavelength'] <= wavelength_max)]

                if spec.empty:
                    ax.text(0.5, 0.5,
                            f'No data in {wavelength_min}-{wavelength_max} nm',
                            ha='center', va='center')
                else:
                    ax.plot(spec['wavelength'], spec['intensity'], color=color)

            except Exception as e:
                ax.text(0.5, 0.5,
                        f"read error:\n{e}",
                        ha='center', va='center', fontsize=8)

            # title formatting
            title = os.path.basename(fp)
            if len(title) > 30:
                title = title[:27] + '...'
            ax.set_title(title)

            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Intensity")
            # ax.grid(alpha=0.2)

        # hide unused axes
        for j in range(n, rows * cols):
            r, c = divmod(j, cols)
            axes[r][c].axis("off")

        plt.tight_layout()
        plt.show()


class CampaignQuery:
    """
    Query system for a multi-round data collection campaign.
    Automatically loads all CSV metadata in a directory.
    """

    def __init__(self, metadata_dir: str, base_data_root: Optional[str] = None):
        self.metadata_dir = metadata_dir
        self.base_data_root = base_data_root  # where the measurement files live
        self.db = self._load_all()

    # === ---------------- Load All Metadata ---------------- === #
    def _load_all(self) -> pd.DataFrame:
        tables = []

        for file in os.listdir(self.metadata_dir):
            if not file.endswith(".csv"):
                continue

            full_path = os.path.join(self.metadata_dir, file)
            df = pd.read_csv(full_path)

            # Parse stringified lists
            if "Dataset" in df.columns:
                df["Dataset"] = df["Dataset"].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
            df = df[['ID', 'Temperature', 'Anneal Time', 'R BAAc', 'R MAI','Data Tag','Sample Tag','Dataset']]
            tables.append(df)

        if not tables:
            raise ValueError("No CSV metadata found in folder: " + self.metadata_dir)

        return pd.concat(tables, ignore_index=True)

    # === ------------------- Query API -------------------- === #
    def look_for(self,
               round: Optional[str] = None,
               sample_num: Optional[str] = None,
               temperature: Optional[float] = None,
               anneal_time: Optional[float] = None,
               r_ba: Optional[float] = None,
               r_mai: Optional[float] = None,
               r_pb: Optional[float] = None):
        df = self.db
        if round is not None:
            df = df[df["Data Tag"] == round]
        if sample_num is not None:
            df = df[df["Sample Tag"] == sample_num]
        if temperature is not None:
            df = df[df["Temperature"] == temperature]
        if anneal_time is not None:
            df = df[df["Anneal Time"] == anneal_time]
        if r_ba is not None:
            df = df[df["R BAAc"] == r_ba]
        if r_mai is not None:
            df = df[df["R MAI"] == r_mai]
        if r_pb is not None:
            df = df[df["R PbI2"] == r_pb]
        return df.copy()

    # === -------------- File Path Access ------------------ === #
    def get_files(self, row: pd.Series) -> List[str]:
        paths = row["Dataset"]
        if self.base_data_root:
            return [os.path.join(self.base_data_root, p) for p in paths]
        return paths
