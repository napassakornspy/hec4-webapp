import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats, linalg
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

class HEC4:
    """
    HEC-4 Monthly Streamflow Simulation (Python implementation)
    Full Pipeline: Pseudo-Missing, Beta Validation, RMSE/NSE & DMA
    """
    def __init__(self, epsilon=0.001, min_concurrent=10, verbose=True):
        self.epsilon = epsilon
        self.min_concurrent = min_concurrent
        self.verbose = verbose
        self.data = self.log_data = self.filled_log = self.filled_data = None
        self.monthly_stats = self.M0 = self.M1 = self.A = self.B = None
        self.synthetic_traces = self.stations = self.n_stations = None

    def load_data(self, filepath, date_col="Date", flow_cols=None, date_format=None, missing_codes=(-999,-9999)):
        file_name = filepath.name.lower()
        if file_name.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("อัปโหลดได้เฉพาะไฟล์ .csv หรือ .xlsx เท่านั้นนะ")

        df[date_col] = pd.to_datetime(df[date_col], format=date_format)
        df = df.set_index(date_col).sort_index()
        if flow_cols: df = df[flow_cols]
        for c in missing_codes: df.replace(c, np.nan, inplace=True)
        df = df.resample("MS").mean()
        
        self.data = df
        self.stations = list(df.columns)
        self.n_stations = len(self.stations)
        
        if self.verbose:
            print("="*60)
            print("  HEC-4 Monthly Streamflow Simulation — Full Upgraded Pipeline")
            print("="*60)
            print(f"  Stations : {self.n_stations}  →  {self.stations}")
        return self

    def generate_pseudo_missing(self, missing_rate=0.1, seed=42):
        np.random.seed(seed)
        self.original_data = self.data.copy()
        self.pseudo_mask = (np.random.rand(*self.data.shape) < missing_rate) & ~self.data.isna()
        self.data[self.pseudo_mask] = np.nan
        self.log_data = np.log(self.data + self.epsilon)
        
        if self.verbose:
            missing_count = self.pseudo_mask.sum().sum()
            print(f"  ✓ สร้าง Pseudo-Missing Data จำลองข้อมูลหายไป {missing_count} จุด (สัดส่วน {missing_rate*100}%)")
        return self

    def infill_missing(self):
        if self.log_data is None: self.log_data = np.log(self.data + self.epsilon)
        filled = self.log_data.copy()
        n_filled = 0
        
        for station in self.stations:
            miss_idx = filled.index[filled[station].isna()]
            if len(miss_idx) == 0: continue
            
            for t in miss_idx:
                m = t.month
                best_r2 = -np.inf; best_pred = None; best_ab = (0.0, 0.0)
                
                for predictor in self.stations:
                    if predictor == station or pd.isna(filled.at[t, predictor]): continue
                    
                    same_m = filled.index.month == m
                    y = filled.loc[same_m, station]; x = filled.loc[same_m, predictor]
                    valid = y.notna() & x.notna()
                    if valid.sum() < self.min_concurrent: continue
                    
                    slope, intercept, r, _, _ = stats.linregress(x[valid].values, y[valid].values)
                    r2 = r**2
                    
                    if r2 > best_r2 and np.isfinite(slope) and (-0.5 <= slope <= 1.5):
                        best_r2 = r2
                        best_pred = predictor
                        best_ab = (slope, intercept)
                        
                if best_pred is not None:
                    filled.at[t, station] = best_ab[1] + best_ab[0] * filled.at[t, best_pred]
                    n_filled += 1
                else:
                    mu = filled.loc[filled.index.month == m, station].dropna().mean()
                    if np.isfinite(mu): 
                        filled.at[t, station] = mu
                        n_filled += 1
                        
        self.filled_log = filled
        self.filled_data = (np.exp(filled) - self.epsilon).clip(lower=0.0)
        if self.verbose: print(f"  ✓ เติมข้อมูลสำเร็จ: {n_filled} จุด (ผ่านเกณฑ์ Beta -0.5 ถึง 1.5)")
        return self

    def compute_statistics(self):
        if self.filled_log is None: self.filled_log = self.log_data
        stats_out = {}
        for station in self.stations:
            stats_out[station] = {}
            for m in range(1, 13):
                vals = self.filled_log.loc[self.filled_log.index.month == m, station].dropna().values
                n = len(vals)
                mu = float(np.mean(vals)) if n > 0 else np.nan
                sigma = float(np.std(vals, ddof=1)) if n > 1 else 0.0
                g = (n * np.sum((vals-mu)**3) / ((n-1)*(n-2)*sigma**3)) if n > 2 and sigma > 0 else 0.0
                rho1 = float(np.corrcoef(vals[:-1], vals[1:])[0,1]) if n > 3 else 0.0
                stats_out[station][m] = dict(n=n, mean_log=mu, std_log=sigma, skew_log=g, rho1_log=0.0 if np.isnan(rho1) else rho1)
        self.monthly_stats = stats_out
        return self

    def compute_correlation_matrices(self):
        n = self.n_stations; M0 = {}; M1 = {}; log = self.filled_log
        for m in range(1, 13):
            m_prev = 12 if m == 1 else m - 1
            data_m = log.loc[log.index.month == m, self.stations]
            m0 = np.eye(n)
            for i in range(n):
                for j in range(i+1, n):
                    yi = data_m.iloc[:,i].dropna(); yj = data_m.iloc[:,j].dropna()
                    common = yi.index.intersection(yj.index)
                    if len(common) >= 4:
                        r = float(np.corrcoef(yi[common], yj[common])[0,1])
                        m0[i,j] = m0[j,i] = 0.0 if np.isnan(r) else np.clip(r, -1, 1)
            
            data_prev = log.loc[log.index.month == m_prev, self.stations]
            years_m = data_m.index.year; years_prev = data_prev.index.year
            cy = np.intersect1d(years_m, years_prev + (1 if m == 1 else 0))
            
            idx_m = [t for t in data_m.index if t.year in cy]
            idx_prev = [t for t in data_prev.index if t.year in (cy - 1 if m == 1 else cy)]
            
            m1 = np.zeros((n,n))
            if len(idx_m) >= 4:
                cur = data_m.loc[idx_m].values; prev = data_prev.loc[idx_prev].values
                for i in range(n):
                    for j in range(n):
                        valid = np.isfinite(cur[:,i]) & np.isfinite(prev[:,j])
                        if valid.sum() >= 4:
                            r = float(np.corrcoef(cur[valid,i], prev[valid,j])[0,1])
                            m1[i,j] = 0.0 if np.isnan(r) else r
            M0[m] = m0; M1[m] = m1
        self.M0 = M0; self.M1 = M1
        return self

    def compute_generation_matrices(self):
        self.A = {}; self.B = {}
        for m in range(1, 13):
            try: A = self.M1[m] @ linalg.inv(self.M0[m])
            except: A = self.M1[m] @ linalg.pinv(self.M0[m])
            
            C = self.M0[m] - A @ self.M0[m].T @ A.T
            C = (C + C.T) / 2.0
            
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.any(eigvals < 0):
                C = eigvecs @ np.diag(np.maximum(eigvals, 0.0)) @ eigvecs.T
                
            try: B = linalg.cholesky(C, lower=True)
            except: B = linalg.cholesky(C + np.eye(self.n_stations)*1e-10, lower=True)
            self.A[m] = A; self.B[m] = B
        return self

    def generate_synthetic(self, n_years=50, n_traces=1, seed=None):
        rng = np.random.default_rng(seed); warmup = 24
        self.synthetic_traces = {}
        
        for tr in range(1, n_traces + 1):
            Z_prev = np.zeros(self.n_stations); records = []
            for t in range(n_years*12 + warmup):
                m = (t % 12) + 1
                Z = self.A[m] @ Z_prev + self.B[m] @ rng.standard_normal(self.n_stations)
                
                log_Q = [self.monthly_stats[self.stations[i]][m]["mean_log"] + 
                         self.monthly_stats[self.stations[i]][m]["std_log"] * Z[i] for i in range(self.n_stations)]
                
                records.append(np.maximum(np.exp(log_Q) - self.epsilon, 0.0))
                Z_prev = Z
                
            start_yr = self.data.index[0].year if self.data is not None else 2000
            idx = pd.date_range(f"{start_yr}-01-01", periods=(n_years*12)+warmup, freq="MS")
            self.synthetic_traces[f"trace_{tr:03d}"] = pd.DataFrame(records, index=idx, columns=self.stations).iloc[warmup:]
        return self

    def validate_imputation(self):
        if not hasattr(self, 'pseudo_mask'): return self
            
        obs = self.original_data.values[self.pseudo_mask]
        sim = self.filled_data.values[self.pseudo_mask]
        
        valid = ~np.isnan(obs) & ~np.isnan(sim)
        obs_valid, sim_valid = obs[valid], sim[valid]
        
        if len(obs_valid) > 0:
            rmse = np.sqrt(mean_squared_error(obs_valid, sim_valid))
            nse = 1 - (np.sum((obs_valid - sim_valid)**2) / np.sum((obs_valid - np.mean(obs_valid))**2))
            if self.verbose:
                print("  Validation Results (Pseudo-Missing Test):")
                print(f"    - RMSE: {rmse:.4f}")
                print(f"    - NSE:  {nse:.4f}")
        return self

    def plot_double_mass_curve(self, station, base_stations=None):
        if self.filled_data is None:
            raise RuntimeError("ต้องรัน infill_missing() หรือ run_all() ก่อนพล็อตกราฟ")
            
        if base_stations is None:
            base_stations = [s for s in self.stations if s != station]
            
        target_cumsum = self.filled_data[station].cumsum()
        base_avg = self.filled_data[base_stations].mean(axis=1)
        base_cumsum = base_avg.cumsum()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(base_cumsum, target_cumsum, marker='o', markersize=3, linestyle='-', color='teal', label='Double Mass Curve')
        
        ax.set_title(f"Double Mass Analysis: {station} vs Average of {len(base_stations)} Stations", fontsize=12)
        ax.set_xlabel("Cumulative Mean of Base Stations (m³/s)")
        ax.set_ylabel(f"Cumulative Flow at {station} (m³/s)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        plt.show()
        return self

    def run_all(self, n_years=50, n_traces=5, seed=42, missing_rate=0.1):
        (self.generate_pseudo_missing(missing_rate=missing_rate, seed=seed)
             .infill_missing().compute_statistics()
             .compute_correlation_matrices().compute_generation_matrices()
             .generate_synthetic(n_years=n_years, n_traces=n_traces, seed=seed)
             .validate_imputation())
        if self.verbose: print("\n  ✓ HEC-4 Pipeline Completed.")
        return self
