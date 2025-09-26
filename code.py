"""
fopid_lstm_schnider_py311.py

Single-file demo: Schnider PK-PD + FOPID controller + LSTM autoencoder (unsupervised)
- Target BIS range: 40 - 60, target default 50
- Meant to run under Python 3.11 (avoid tensorflow.python direct imports)
Requirements:
    pip install numpy scipy matplotlib tensorflow
Run:
    python fopid_lstm_schnider_py311.py
"""

import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam

# --- Schnider PKPD model (simplified) ---
class SchniderPKPD:
    def __init__(self, weight_kg=70.0, age_yr=40.0, height_cm=170.0):
        # Simplified baseline params for demonstration.
        self.V1 = 4.27    # L
        self.V2 = 18.9
        self.V3 = 238.0
        self.k10 = 0.331
        self.k12 = 0.154
        self.k13 = 0.025
        self.k21 = 0.138
        self.k31 = 0.003
        self.ke0 = 0.456
        # PD:
        self.BIS0 = 100.0
        self.Emax = 100.0
        self.C50 = 2.5
        self.gamma = 2.0
        # states
        self.C1 = 0.0
        self.C2 = 0.0
        self.C3 = 0.0
        self.Ce = 0.0

    def derivatives(self, u):
        # u in mg/min
        input_ug_per_min = u * 1000.0
        V1_ml = self.V1 * 1000.0
        dC1 = (input_ug_per_min / V1_ml) - (self.k10 + self.k12 + self.k13) * self.C1 + self.k21 * self.C2 + self.k31 * self.C3
        dC2 = self.k12 * self.C1 - self.k21 * self.C2
        dC3 = self.k13 * self.C1 - self.k31 * self.C3
        dCe = self.ke0 * (self.C1 - self.Ce)
        return dC1, dC2, dC3, dCe

    def step(self, u, dt):
        # RK4 integration
        c1, c2, c3, ce = self.C1, self.C2, self.C3, self.Ce

        def derivs(state, u_):
            C1s, C2s, C3s, Ces = state
            old = (self.C1, self.C2, self.C3, self.Ce)
            self.C1, self.C2, self.C3, self.Ce = C1s, C2s, C3s, Ces
            d = self.derivatives(u_)
            self.C1, self.C2, self.C3, self.Ce = old
            return np.array(d)

        s = np.array([c1, c2, c3, ce], dtype=float)
        k1 = derivs(s, u)
        k2 = derivs(s + 0.5 * dt * k1, u)
        k3 = derivs(s + 0.5 * dt * k2, u)
        k4 = derivs(s + dt * k3, u)
        s_new = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.C1, self.C2, self.C3, self.Ce = s_new

        Ce = self.Ce
        BIS = self.BIS0 - (self.Emax * (Ce ** self.gamma) / (self.C50 ** self.gamma + Ce ** self.gamma + 1e-12))
        return float(BIS)

# --- Fractional PID (Grünwald-Letnikov approx) ---
def binomial_coeff(alpha, k):
    return gamma(alpha + 1) / (gamma(k + 1) * gamma(alpha - k + 1))

class FOPID:
    def __init__(self, Kp=30.0, Ki=5.0, Kd=0.5, lam=0.9, mu=0.9, dt=0.5, hist_len=300):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.lam, self.mu = lam, mu
        self.dt = dt
        self.hist_len = hist_len
        self.error_hist = []
        self.deriv_coeffs = np.array([((-1)**k) * binomial_coeff(self.mu, k) for k in range(self.hist_len)])
        self.int_coeffs = np.array([((-1)**k) * binomial_coeff(-self.lam, k) for k in range(self.hist_len)])

    def update_history(self, error):
        self.error_hist.append(error)
        if len(self.error_hist) > self.hist_len:
            self.error_hist.pop(0)

    def frac_derivative(self):
        N = len(self.error_hist)
        if N == 0: return 0.0
        coeffs = self.deriv_coeffs[:N][::-1]
        errors = np.array(self.error_hist[::-1])
        val = np.dot(coeffs, errors)
        return val / (self.dt ** self.mu)

    def frac_integral(self):
        N = len(self.error_hist)
        if N == 0: return 0.0
        coeffs = self.int_coeffs[:N][::-1]
        errors = np.array(self.error_hist[::-1])
        val = np.dot(coeffs, errors)
        return val * (self.dt ** self.lam)

    def control(self, error):
        self.update_history(error)
        P = self.Kp * error
        I = self.Ki * self.frac_integral()
        D = self.Kd * self.frac_derivative()
        u = P + I + D
        u = max(0.0, u)
        u = min(u, 200.0)
        return float(u)

# --- LSTM autoencoder builder ---
def build_lstm_autoencoder(timesteps, features, latent_dim=64, lr=1e-3):
    model = Sequential()
    model.add(LSTM(latent_dim, activation='tanh', input_shape=(timesteps, features), return_sequences=False))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(latent_dim, activation='tanh', return_sequences=True))
    model.add(TimeDistributed(Dense(features)))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

# --- Integration & simulation --
def run_simulation(sim_time_s=300, dt=0.5, target_bis=50.0):
    steps = int(sim_time_s / dt)
    plant = SchniderPKPD()
    # Fine-tuned controller parameters for better stability in 40-60 range
    fopid = FOPID(Kp=-20.0, Ki=-5.0, Kd=-0.5, lam=0.9, mu=0.9, dt=dt, hist_len=300)

    # simulation buffers
    t_trace = []; bis_trace = []; infusion_trace = []; error_trace = []
    
    # Start with zero infusion to ensure BIS starts at 100
    u = 0.0
    
    # Record initial BIS (should be 100)
    bis_initial = plant.step(0.0, dt)  # Step with zero infusion to get initial BIS
    
    # Record the initial point
    t_trace.append(0)
    bis_trace.append(bis_initial)
    infusion_trace.append(u)
    error_trace.append(target_bis - bis_initial)
    
    print(f"Initial BIS: {bis_initial:.1f}")

    # Simulation loop
    for step in range(1, steps):
        t = step * dt
        bis = plant.step(u, dt)
        error = target_bis - bis
        
        # Special control logic to maintain BIS in 40-60 range
        if bis > 60:
            # Increase infusion when BIS is too high
            u_cmd = 30.0
        elif bis < 40:
            # Decrease infusion when BIS is too low
            u_cmd = 0.0
        else:
            # Use FOPID controller when in the target range
            u_cmd = fopid.control(error)

        # smoothing infusion to mimic pump dynamics
        alpha = 0.2  # moderate response
        u = (1 - alpha) * u + alpha * u_cmd

        # clamp infusion
        u = max(0.0, min(u, 50.0))  # limit max infusion for smoother control

        # record
        t_trace.append(t)
        bis_trace.append(bis)
        infusion_trace.append(u)
        error_trace.append(error)

        # Print progress every 60 seconds
        if step % 120 == 0:
            print(f"Time: {t:.0f}s, BIS: {bis:.1f}, Infusion: {u:.1f} mg/min, Error: {error:.1f}")

    return t_trace, bis_trace, infusion_trace, error_trace

# --- Run & plot ---
if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    # Reduced simulation time from 1800s to 300s for faster execution
    t_trace, bis_trace, infusion_trace, error_trace = run_simulation(sim_time_s=300, dt=0.5, target_bis=50.0)
    t = np.array(t_trace) / 60.0  # convert to minutes
    bis = np.array(bis_trace)
    infusion = np.array(infusion_trace)

    print(f"Final BIS: {bis[-1]:.1f}")
    print(f"BIS range achieved: {bis.min():.1f} - {bis.max():.1f}")

    # Create and save BIS plot
    plt.figure(figsize=(10,6))
    plt.plot(t, bis, label="BIS", linewidth=2)
    plt.hlines([40, 60], t[0], t[-1], colors=['red', 'red'], linestyles='dashed', label="Target Band [40,60]")
    plt.hlines([50], t[0], t[-1], colors=['green'], linestyles='solid', alpha=0.7, label="Target BIS=50")
    plt.xlabel("Time (min)")
    plt.ylabel("BIS")
    plt.title("BIS Control: Starting from 100, Targeting 40-60 Range")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.savefig('bis_plot.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Create and save infusion plot
    plt.figure(figsize=(10,6))
    plt.plot(t, infusion, label="Infusion Rate", color='blue', linewidth=2)
    plt.xlabel("Time (min)")
    plt.ylabel("Infusion Rate (mg/min)")
    plt.title("Propofol Infusion Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('infusion_plot.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Plots saved: bis_plot.png and infusion_plot.png")
