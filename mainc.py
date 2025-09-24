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

# --- Integration & simulation ---
def run_simulation(sim_time_s=1800, dt=0.5, target_bis=50.0):
    steps = int(sim_time_s / dt)
    plant = SchniderPKPD()
    fopid = FOPID(Kp=30.0, Ki=5.0, Kd=0.5, lam=0.9, mu=0.9, dt=dt, hist_len=300)

    seq_len = 40
    features = 2
    lstm = build_lstm_autoencoder(seq_len, features, latent_dim=64, lr=1e-3)

    # quick lightweight warmup training on simple open-loop sim to let the model have reasonable weights
    # (keeps runtime reasonable while giving the LSTM something to predict)
    warmup_data = []
    u_warm = 20.0
    plant_warm = SchniderPKPD()
    for _ in range(200):
        b = plant_warm.step(u_warm, dt)
        warmup_data.append([b, u_warm])
    arr = np.array(warmup_data)
    # normalize
    mean = arr.mean(axis=0); std = arr.std(axis=0) + 1e-6
    seqs = []
    for i in range(len(arr) - seq_len):
        s = (arr[i:i+seq_len] - mean) / std
        seqs.append(s)
    seqs = np.array(seqs, dtype=np.float32)
    if len(seqs) > 0:
        lstm.fit(seqs, seqs, epochs=3, batch_size=8, verbose=0)

    # simulation buffers
    t_trace = []; bis_trace = []; infusion_trace = []; error_trace = []
    u = 25.0  # initial infusion
    for _ in range(10):
        _ = plant.step(u, dt)

    history_buffer = []

    for step in range(steps):
        t = step * dt
        bis = plant.step(u, dt)
        error = target_bis - bis
        u_cmd = fopid.control(error)

        # sliding buffer for LSTM (raw values)
        history_buffer.append([bis, u_cmd])
        if len(history_buffer) >= seq_len:
            seq = np.array(history_buffer[-seq_len:])
            running_mean = seq.mean(axis=0)
            running_std = seq.std(axis=0) + 1e-6
            seq_norm = (seq - running_mean) / running_std
            x = seq_norm.reshape((1, seq_len, features))
            # predict using LSTM autoencoder (no online weight updates)
            try:
                pred = lstm.predict(x, verbose=0)
                pred_last_bis_norm = pred[0, -1, 0]
                pred_last_bis = pred_last_bis_norm * running_std[0] + running_mean[0]
                bis_trend = pred_last_bis - seq[-1, 0]
                # conservative adaptation
                adapt_delta = -0.04 * bis_trend
                fopid.Kp = max(0.1, min(200.0, fopid.Kp + adapt_delta))
                fopid.Ki = max(0.0, min(50.0, fopid.Ki + (-0.0005 * np.sign(error) * abs(bis_trend))))
            except Exception:
                pass

        # smoothing infusion to mimic pump dynamics
        alpha = 0.2
        u = (1 - alpha) * u + alpha * u_cmd

        t_trace.append(t); bis_trace.append(bis); infusion_trace.append(u); error_trace.append(error)

    return {
        "t": np.array(t_trace),
        "bis": np.array(bis_trace),
        "infusion": np.array(infusion_trace),
        "error": np.array(error_trace),
        "fopid": fopid
    }

# --- Run & plot ---
if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    sim = run_simulation(sim_time_s=1800, dt=0.5, target_bis=50.0)
    t = sim["t"] / 60.0
    bis = sim["bis"]
    infusion = sim["infusion"]
    fopid = sim["fopid"]

    print(f"Final FOPID gains: Kp={fopid.Kp:.3f} Ki={fopid.Ki:.3f} Kd={fopid.Kd:.3f}")

    plt.figure(figsize=(10,6))
    plt.plot(t, bis, label="BIS")
    plt.hlines([40, 60], t[0], t[-1], linestyles='dashed', label="Band [40,60]")
    plt.hlines(50, t[0], t[-1], linestyles='dotted', colors='green', label="Target 50")
    plt.xlabel("Time (minutes)"); plt.ylabel("BIS"); plt.legend(); plt.grid(True)

    plt.figure(figsize=(10,4))
    plt.plot(t, infusion, label="Infusion (mg/min)")
    plt.xlabel("Time (minutes)"); plt.ylabel("Infusion (mg/min)"); plt.legend(); plt.grid(True)

    plt.show()
