import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import itertools
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement 
from botorch.optim import optimize_acqf, optimize_acqf_discrete, optimize_acqf_mixed
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.sampling.normal import SobolQMCNormalSampler
import gpytorch
from botorch.utils.transforms import normalize, unnormalize
from botorch.acquisition import qLogExpectedImprovement 


class GaussianProcess:
    def __init__(self, train_X=None, train_y=None, train_yvar=None, bounds=None):
        self.train_X = train_X
        self.train_y = train_y
        self.train_yvar = train_yvar
        self.gp = None
        self.bounds = bounds

    def fit(self):
        # Set the device and dtype
        dtype = torch.float64

        X = torch.tensor(self.train_X, dtype=dtype)
        X = normalize(X, bounds=self.bounds)
        y = torch.tensor(self.train_y, dtype=dtype).reshape(-1,1)

        # Handle train_yvar
        if self.train_yvar is not None:
            yvar = torch.tensor(self.train_yvar, dtype=dtype).reshape(-1,1)
                    # Define a GP model
            self.gp = SingleTaskGP(train_X=X, train_Y=y, train_Yvar=yvar)
            # Marginal Log Likelihood (MLL) for training
            mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
            # Optimize model
            fit_gpytorch_mll(mll)
        else:
            # Define a GP model
            self.gp = SingleTaskGP(train_X=X, train_Y=y)
            # Marginal Log Likelihood (MLL) for training
            mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
            # Optimize model
            fit_gpytorch_mll(mll)

        return self.gp
    
    def get_model(self):
        return self.gp

    def predict(self, X_test):
        dtype = torch.float64
        if isinstance(X_test, np.ndarray):
            X_test = torch.tensor(X_test, dtype=dtype)

        self.gp.eval()
        self.likelihood.eval()

        with torch.no_grad():
            posterior = self.gp.posterior(X_test)
            mean = posterior.mean.squeeze(-1)  # Mean prediction
            std_dev = posterior.variance.sqrt().squeeze(-1)  # Standard deviation

        return mean.numpy(), std_dev.numpy()

    def evaluate(self, X_test):
        dtype = torch.float64
        if isinstance(X_test, np.ndarray):
            X_test = torch.tensor(X_test, dtype=dtype)

        self.gp.eval()
        with torch.no_grad():
            posterior = self.gp.posterior(X_test)
            mean = posterior.mean.squeeze(-1)
            std_dev = posterior.variance.sqrt().squeeze(-1)
        return mean.numpy(), std_dev.numpy()

    def get_train_y(self):
        return self.train_y

#%% 
class BayesianOptimization: 
    def __init__(self, gp_model: GaussianProcess, bounds=None, batch_size=None, seed = None):
        self.gp_model = gp_model
        self.batch_size = batch_size
        self.bounds = bounds if bounds is not None else torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        self.batch_size = batch_size if batch_size is not None else 1
        self.seed = seed

    def suggest_next_point(self, acquisition='EI'):
        dtype = torch.float64
        device = torch.device('cpu')
        model = self.gp_model.get_model()
        best_f = self.gp_model.get_train_y().max() # theoretically should be 1.0
        batch_size = self.batch_size
        bounds = self.bounds.to(device, dtype=dtype)

        if acquisition == 'EI':
            sampler = SobolQMCNormalSampler(torch.Size([1024]), seed=self.seed)
            qEI = qLogExpectedImprovement(model=model, best_f=best_f, sampler=sampler)
        else:
            raise NotImplementedError(f"Acquisition function '{acquisition}' is not implemented.")

        bo_bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=dtype)

        torch.manual_seed(self.seed)

        candidate, _ = optimize_acqf(
            acq_function=qEI,
            bounds=bo_bounds,
            q=batch_size,
            num_restarts=15,
            raw_samples=1024, # more samples = better coverage
            options={"dtype": dtype, "device": device}
        )


        unnormalized_candidates = unnormalize(candidate, bounds).detach().cpu().numpy()
        self.df_candidates = pd.DataFrame(unnormalized_candidates, columns=['Anneal Time', 'R BAAc', 'R MAI'])

        return  self.df_candidates
    
    def full_grid(self, bounds = None):
        '''
        bounds must be in order [[R BAAc min, R BAAc max], [R MAI min, R MAI max], [Anneal Time min, Anneal Time max],[Temperature min, Temperature max]]
        '''
        #bounds = [[5, 60], [60, 150], [0.40, 1.1],[1.2, 1.6]] #  Time, Temp, BAAc, MAI,

        # Create fine grid for temperature and anneal time with step size 5
        anneal_time_range = np.arange(bounds[0][0], bounds[0][1] + 4, 5).round(1)
        temperature_range = np.arange(bounds[1][0], bounds[1][1] + 5, 5).round(1)

        # Create fine grid with step size 0.025 for R BAAc and R MAI, all anneal times, R PbI2=1
        r_baac_fine = np.arange(bounds[2][0], bounds[2][1] + 0.025, 0.025).round(3)
        r_mai_fine = np.arange(bounds[3][0], bounds[3][1] + 0.025, 0.025).round(3)
        r_pbi2_fine = np.array([1.])

        # Create fine grid for all 4 dimensions: R BAAc, R MAI, Temperature, Anneal Time, with R PbI2 fixed at 1
        fine_grid_4d = list(itertools.product(anneal_time_range,temperature_range, r_baac_fine, r_mai_fine))
        fine_grid_4d_df = pd.DataFrame(fine_grid_4d, columns=['Anneal Time','Temperature', 'R BAAc', 'R MAI'])

        return fine_grid_4d_df
    
    def suggest_next_point_mixed_discrete(self, pool=None, sampled_candidates=None):
        dtype = torch.float64
        device = torch.device('cpu')
        model = self.gp_model.get_model()
        best_f = self.gp_model.get_train_y().max().item()
        batch_size = self.batch_size
        bounds = self.bounds.to(device, dtype=dtype).T

        if pool is not None:
            candidate_pool = pool  # Ensure candidate_pool is a numpy array
        else:
            candidate_pool = self.full_grid(bounds.T) # 4D pool
        
        
        best = (None, None, -float("inf"))
        temps = np.unique(candidate_pool['Temperature'].to_numpy())
        
        if sampled_candidates is not None:
            # we want to avoid candidates that are already sampled
            sampled_candidates = normalize( torch.tensor(sampled_candidates, dtype=dtype, device=device), bounds=bounds)

    # before the loop
        X_raw = torch.tensor(candidate_pool.to_numpy(), dtype=dtype, device=device)
        X_norm = normalize(X_raw, bounds=bounds)           # b2 is bounds shaped [2, d]
        temp_col = 1                                   # <-- set your actual Temperature column
        temps = torch.unique(X_norm[:, temp_col])      # unique temps in normalized space
        with torch.no_grad():
            for Tn in temps:
                mask = torch.isclose(X_norm[:, temp_col], Tn, atol=1e-8, rtol=0)
                choices = X_norm[mask]
                if choices.shape[0] < batch_size: 
                    continue

                # Create acquisition function
                sampler = SobolQMCNormalSampler(torch.Size([256]), seed=self.seed)
                acqf = qLogExpectedImprovement(model=model, best_f=best_f, sampler=sampler) # Expected Improvement

                Xq, val = optimize_acqf_discrete(
                    acq_function=acqf,
                    choices=choices,
                    q=batch_size,
                    max_batch_size = min(4096, choices.shape[0]),  # max batch size for discrete optimization
                    unique=True,
                    X_avoid = sampled_candidates
                    # (optional) pass inequality/equality constraints, options, etc.
                )
                v = val.item() if val.ndim == 0 else val.sum().item()   # or .mean().item()
                if v > best[2]:
                    best = (Xq, Tn.item() if hasattr(Tn, "item") else Tn, v)

        if best[0] is None:
            raise RuntimeError("No feasible batch—check bounds / q.")
        
        unnormalized_candidates = unnormalize(best[0], bounds).detach().cpu().numpy()
        self.df_candidates = pd.DataFrame(unnormalized_candidates, columns=['Anneal Time','Temperature','R BAAc', 'R MAI'])
        return self.df_candidates

    def suggest_next_point_discrete(self, acquisition='EI'):
        dtype = torch.float64
        device = torch.device('cpu')
        model = self.gp_model.get_model()
        best_f = self.gp_model.get_train_y().max().item()
        batch_size = self.batch_size
        bounds = torch.tensor(self.bounds)

        # Step 1: Make a grid in normalized [0, 1] space
        # d = bounds.shape[1]
        # normalized_grids = [np.arange(0.0, 1.0 + stepsize, stepsize) for _ in range(d)]
        # mesh = np.meshgrid(*normalized_grids, indexing='ij')
        # flat_grid = np.stack([m.flatten() for m in mesh], axis=1)
        # candidate_pool_normalized = torch.tensor(flat_grid, dtype=dtype, device=device)
        candidate_pool = torch.tensor(self.full_grid(bounds).to_numpy())
        candidate_pool_normalized = normalize(candidate_pool, bounds=bounds).to(device, dtype=dtype)
        # Create acquisition function
        if acquisition == 'EI':
            sampler = SobolQMCNormalSampler(torch.Size([256]), seed=self.seed)
            qEI = qLogExpectedImprovement(model=model, best_f=best_f, sampler=sampler)
        else:
            raise NotImplementedError(f"Acquisition function '{acquisition}' is not implemented.")

        # Select batch from discrete candidate pool
        candidate, _ = optimize_acqf_discrete(
            acq_function=qEI,
            choices=candidate_pool_normalized,
            q=batch_size,
            unique=True,
        )

        unnormalized_candidates = unnormalize(candidate, bounds).detach().cpu().numpy()
        self.df_candidates = pd.DataFrame(unnormalized_candidates, columns=['Anneal Time', 'R BAAc', 'R MAI'])

        return self.df_candidates
    
    def Convert2Volume(self, df_candidates=None):
        """
        Convert the stoichiometric candidates to component volumes.
        The final volume of each mixture is scaled to a total of 200 µL.
        Assumes candidates_df has columns 'BAAc' and 'MAI'.
        The remaining fraction goes to 'PbI2' (i.e., PbI2 = 1 - BAAc - MAI).
        """
        max_volume = 100.0
        if df_candidates is not None:
            df = df_candidates.copy()
        else:
            # Copy stoichiometric fractions
            df = self.df_candidates.copy()

        # Compute PbI2
        df['R PbI2'] = 1.0 
        
        total = df['R PbI2'] + df['R BAAc'] + df['R MAI']
        fract = max_volume/total

        # Scale to total volume
        df['PbI2_vol'] = df['R PbI2'] * fract
        df['BAAc_vol'] = df['R BAAc'] * fract
        df['MAI_vol'] = df['R MAI'] * fract
        
        # Save to a new DataFrame
        self.volume_df = df[['Anneal Time','PbI2_vol', 'BAAc_vol', 'MAI_vol']].sort_values('Anneal Time', ascending=False).reset_index(drop=True)
        self.full_df = df[['Anneal Time', 'R PbI2', 'R BAAc', 'R MAI', 'PbI2_vol', 'BAAc_vol', 'MAI_vol' ]].sort_values('Anneal Time', ascending=False).reset_index(drop=True).round(2)
    
    def get_full_df(self):
        return self.full_df
    
    def get_volume_df(self): 
        return self.volume_df

    def save_results(self, full_df,output_path):
        """
        Save the candidates and their corresponding volumes to a CSV file.
        Parameters:
            output_path (str): Path to save the CSV file.
        """
        df = full_df.copy()
        df.insert(0, 'ID', range(1, len(df) + 1))
        df.to_csv(output_path, mode = 'a',index=False)
        print(f"Results saved to {output_path}")
