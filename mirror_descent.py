import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. Problem dimensions ----------
C = 2                       # channels 0=local, 1/2 = wireless
rates = np.arange(1, 8)     # r = 1..7
actions = [(c, r) for c in range(C + 1) for r in rates]
A = len(actions)

# ---------- 2. Physical & QoE parameters ----------
g_r = np.array([0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85])
Q_r = np.array([32, 34, 36, 38, 40, 42, 44], dtype=float)

f_VR = 3e9
f_ED = 15e9
rho = 200
kappa = 5e-12
eps_tx = 1e-9
eps_rx = 1e-9

# Objective weights
w_L, w_E, w_Q = 1.0, 1e9, 0.5

# Channel-specific throughput parameters (log-normal)
# channel 1: better, channel 2: worse
uplink_mu = [None, np.log(40e6), np.log(20e6)]
uplink_sigma = [None, 0.25, 0.35]
down_mu = [None, np.log(50e6), np.log(25e6)]
down_sigma = [None, 0.25, 0.35]

# ---------- 3. Helper functions ----------
def latency_energy(c, r_idx, S0, Uc=None, Dc=None):
    g = 1 + g_r[r_idx]
    S = g * S0
    if c == 0:
        L = rho * S / f_VR
        E = kappa * f_VR**2 * rho * S
    else:
        L = S / Uc + rho * S / f_ED + S / Dc
        E = eps_tx * S / Uc + eps_rx * S / Dc
    return L, E

def project_simplex(v):
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(len(u)) + 1
    rho_idx = ind[u - cssv / ind > 0][-1]
    theta = cssv[rho_idx - 1] / rho_idx
    return np.maximum(v - theta, 0)

# ---------- 4. Online Mirror Descent ----------
T = 4000
eta0 = 0.25
pi = np.full(A, 1 / A)

rng = np.random.default_rng(7)

for t in range(1, T + 1):
    S0 = rng.lognormal(mean=np.log(2e6), sigma=0.5)

    U = np.array([rng.lognormal(uplink_mu[c], uplink_sigma[c]) for c in range(1, C+1)])
    D = np.array([rng.lognormal(down_mu[c], down_sigma[c]) for c in range(1, C+1)])

    z = np.zeros(A)
    for idx, (c, r) in enumerate(actions):
        r_idx = r - 1
        if c == 0:
            L, E = latency_energy(0, r_idx, S0)
        else:
            L, E = latency_energy(c, r_idx, S0, Uc=U[c-1], Dc=D[c-1])
        z[idx] = w_L * L + w_E * E - w_Q * Q_r[r_idx]

    eta = eta0 / np.sqrt(t)
    pi = project_simplex(pi - eta * z)

# ---------- 5. Heatmap of final policy ----------
pi_matrix = pi.reshape((C + 1, 7))

plt.figure()
plt.imshow(pi_matrix, aspect='auto', cmap='viridis')
plt.colorbar(label='Policy Probability π(c,r)')
plt.xlabel('Encoding Profile r')
plt.ylabel('Channel c')
plt.title('Final Policy Heatmap after Mirror Descent')
plt.xticks(np.arange(7), rates)
plt.yticks(np.arange(C + 1), [str(c) for c in range(C + 1)])
plt.show()
