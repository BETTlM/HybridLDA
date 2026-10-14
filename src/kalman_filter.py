"""
Kalman Filter Analysis Module

Kalman Filter and Continuous State Estimation:
The Kalman Filter is the continuous-state analog of the HMM. It models systems where the 
hidden state evolves according to a linear dynamical system with Gaussian noise. 
In the context of topic modeling, the state represents the true underlying prevalence 
(proportion) of each research topic over time, while the observations are the empirical 
prevalences we calculate from the noisy, finite document samples published each year.

The filter alternates between a Predict step (moving the state forward in time) and an 
Update step (incorporating new observations to correct the prediction). The RTS smoother 
runs backward to provide the best estimate of historical states given all data.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class KalmanFilter:
    def __init__(self, dim_x: int, dim_z: int):
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        self.x = np.zeros((dim_x, 1))      # State vector
        self.P = np.eye(dim_x)             # Uncertainty covariance
        
        self.F = np.eye(dim_x)             # State transition matrix
        self.H = np.zeros((dim_z, dim_x))  # Measurement function
        self.R = np.eye(dim_z)             # Measurement noise
        self.Q = np.eye(dim_x)             # Process noise
        
    def predict(self) -> None:
        """Predict step: updates state and covariance forward in time."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> None:
        """Update step: incorporates a new measurement."""
        y = z - self.H @ self.x            # Innovation / residual
        S = self.H @ self.P @ self.H.T + self.R # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.pinv(S) # Kalman gain
        
        self.x = self.x + K @ y
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P

def rts_smoother(Xs: List[np.ndarray], Ps: List[np.ndarray], F: np.ndarray, Q: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Rauch-Tung-Striebel (RTS) backward smoothing."""
    n = len(Xs)
    Xs_smooth = Xs.copy()
    Ps_smooth = Ps.copy()
    
    for k in range(n-2, -1, -1):
        P_pred = F @ Ps[k] @ F.T + Q
        C = Ps[k] @ F.T @ np.linalg.pinv(P_pred) # Smoother gain
        
        Xs_smooth[k] = Xs[k] + C @ (Xs_smooth[k+1] - (F @ Xs[k]))
        Ps_smooth[k] = Ps[k] + C @ (Ps_smooth[k+1] - P_pred) @ C.T
        
    return Xs_smooth, Ps_smooth

def run_kalman_analysis(years: pd.Series, theta: np.ndarray, topic_words: List[List[str]], n_topics: int) -> Dict[str, Any]:
    """Runs the Kalman filter on yearly topic data (mean theta)."""
    logger.info("Starting Kalman Filter Analysis...")
    
    # Calculate actual yearly mean theta
    df = pd.DataFrame(theta)
    df["year"] = years.values
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    
    if len(df) < 5:
        return {"error": "Not enough temporal data"}
        
    yearly_mean = df.groupby("year").mean().sort_index()
    measurements = [row.values.reshape(-1, 1) for _, row in yearly_mean.iterrows()]
    
    if len(measurements) < 3:
        return {"error": "Sequence too short"}
    
    kf = KalmanFilter(dim_x=n_topics, dim_z=n_topics)
    kf.H = np.eye(n_topics)
    kf.R *= 0.05  # Observational noise
    kf.Q *= 0.01  # Process noise
    kf.x = measurements[0] # Initialize with first observation
    
    Xs = []
    Ps = []
    
    for z in measurements:
        kf.predict()
        kf.update(z)
        Xs.append(kf.x.copy())
        Ps.append(kf.P.copy())
        
    Xs_smooth, Ps_smooth = rts_smoother(Xs, Ps, kf.F, kf.Q)
    
    # Flatten outputs for JSON serialization
    Xs_out = [x.flatten().tolist() for x in Xs]
    Xs_smooth_out = [x.flatten().tolist() for x in Xs_smooth]
    Ps_diag = [np.diag(p).tolist() for p in Ps_smooth]
    
    return {
        'filtered_prevalences': Xs_out,
        'smoothed_prevalences': Xs_smooth_out,
        'uncertainty_variance': Ps_diag,
        'years_modeled': yearly_mean.index.tolist()
    }
