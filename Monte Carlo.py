import tkinter as tk
from tkinter import ttk
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting as before

###############################################################################
# MONTE CARLO SIMULATION
###############################################################################
def simulate_GBM_paths(S0, r, sigma, T, M, N):
    """
    Simulate geometric Brownian motion paths under the risk-neutral measure.
    Returns:
      S_paths : shape (M, N+1), each row is one path
      t_grid  : shape (N+1,) array of time points
    """
    dt = T / N
    t_grid = np.linspace(0, T, N+1)
    S_paths = np.zeros((M, N+1))
    S_paths[:, 0] = S0
    
    # Draw normal increments (M x N)
    Z = np.random.normal(0.0, 1.0, (M, N))
    
    for i in range(N):
        # log-Euler step for GBM
        S_paths[:, i+1] = S_paths[:, i] * np.exp(
            (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[:, i]
        )
    return S_paths, t_grid


def payoff_european(S_paths, K, r, T, opt_type="CALL"):
    """
    For each path, payoff = max(S_T - K, 0) or (K - S_T, 0), discounted.
    """
    M, steps = S_paths.shape
    S_T = S_paths[:, -1]  # terminal price
    if opt_type.upper() == "CALL":
        payoff = np.maximum(S_T - K, 0.0)
    else:  # "PUT"
        payoff = np.maximum(K - S_T, 0.0)
    
    discounted = np.exp(-r*T)*payoff
    price_est = np.mean(discounted)
    sample_std = np.std(discounted, ddof=1)
    stderr = sample_std/np.sqrt(M)
    return price_est, stderr


def payoff_asian(S_paths, K, r, T, opt_type="CALL"):
    """
    Arithmetic-average Asian CALL or PUT payoff: 
    payoff = max( (avg(S) - K), 0 ) or max( (K - avg(S)), 0 )
    """
    M, steps = S_paths.shape
    avg_price = np.mean(S_paths, axis=1)
    
    if opt_type.upper() == "CALL":
        payoff = np.maximum(avg_price - K, 0.0)
    else:
        payoff = np.maximum(K - avg_price, 0.0)

    discounted = np.exp(-r*T)*payoff
    price_est = np.mean(discounted)
    sample_std = np.std(discounted, ddof=1)
    stderr = sample_std/np.sqrt(M)
    return price_est, stderr


def payoff_barrier(S_paths, K, r, T, opt_type="CALL", barrier_level=80.0, barrier_type="down-and-out"):
    """
    Example Barrier payoff. 
    'down-and-out' call = call payoff but worthless if min(S_path) < barrier_level
    'down-and-in' call  = call payoff only if min(S_path) < barrier_level, else zero
    Similar logic for PUT. This is just a placeholder to show how barrier logic might be added.
    """
    M, steps = S_paths.shape
    
    # Check barrier condition for each path
    path_min = np.min(S_paths, axis=1)  # or path_max, etc. for up-and-out
    
    # For demonstration let's do a "down-and-out" call:
    # if path_min < barrier_level => option knocked out => payoff = 0
    # else payoff = max(S_T - K, 0)
    
    S_T = S_paths[:, -1]
    if opt_type.upper() == "CALL":
        intrinsic = np.maximum(S_T - K, 0.0)
    else:
        intrinsic = np.maximum(K - S_T, 0.0)
    
    if barrier_type == "down-and-out":
        # zero out payoff if path_min < barrier
        knocked_out = (path_min < barrier_level)
        payoff = np.where(knocked_out, 0.0, intrinsic)
    elif barrier_type == "down-and-in":
        # only valid if path_min < barrier
        knocked_in = (path_min < barrier_level)
        payoff = np.where(knocked_in, intrinsic, 0.0)
    else:
        payoff = intrinsic  # default fallback

    discounted = np.exp(-r*T)*payoff
    price_est = np.mean(discounted)
    sample_std = np.std(discounted, ddof=1)
    stderr = sample_std/np.sqrt(M)
    return price_est, stderr


###############################################################################
# Implied volatility function, if needed for 3D surface (vanilla calls)
###############################################################################
def price_vanilla_call_mc(S_paths, r, T, K):
    S_T = S_paths[:, -1]
    payoff = np.maximum(S_T - K, 0.0)
    discounted = np.exp(-r*T)*payoff
    return np.mean(discounted)

def black_scholes_call(S0, K, r, sigma, T):
    if T <= 0:
        return max(S0 - K, 0)
    d1 = (np.log(S0/K) + (r + 0.5*sigma*sigma)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return (S0*st.norm.cdf(d1) - K*np.exp(-r*T)*st.norm.cdf(d2))

def implied_vol_bs_call(price, S0, K, r, T, tol=1e-6, max_iter=100):
    if price < np.exp(-r*T)*max(S0-K,0):
        return 0.0
    if price > S0:
        return np.nan
    low, high = 0.0, 5.0
    for _ in range(max_iter):
        mid = 0.5*(low+high)
        val = black_scholes_call(S0, K, r, mid, T)
        if abs(val - price) < tol:
            return mid
        if val > price:
            high = mid
        else:
            low = mid
    return mid


###############################################################################
# GUI
###############################################################################
class MultiOptionGUI:
    def __init__(self, master):
        self.master = master
        master.title("Monte Carlo Multi-Option Simulation")

        row = 0
        ttk.Label(master, text="Initial Price (S0):").grid(row=row, column=0, sticky=tk.E, padx=5, pady=3)
        self.S0_entry = ttk.Entry(master, width=10)
        self.S0_entry.insert(0, "100.0")
        self.S0_entry.grid(row=row, column=1, padx=5, pady=3)
        row += 1

        ttk.Label(master, text="Strike (K):").grid(row=row, column=0, sticky=tk.E, padx=5, pady=3)
        self.K_entry = ttk.Entry(master, width=10)
        self.K_entry.insert(0, "100.0")
        self.K_entry.grid(row=row, column=1, padx=5, pady=3)
        row += 1

        ttk.Label(master, text="Risk-free rate (r):").grid(row=row, column=0, sticky=tk.E, padx=5, pady=3)
        self.r_entry = ttk.Entry(master, width=10)
        self.r_entry.insert(0, "0.05")
        self.r_entry.grid(row=row, column=1, padx=5, pady=3)
        row += 1

        ttk.Label(master, text="Volatility (sigma):").grid(row=row, column=0, sticky=tk.E, padx=5, pady=3)
        self.sigma_entry = ttk.Entry(master, width=10)
        self.sigma_entry.insert(0, "0.20")
        self.sigma_entry.grid(row=row, column=1, padx=5, pady=3)
        row += 1

        ttk.Label(master, text="Maturity (T in years):").grid(row=row, column=0, sticky=tk.E, padx=5, pady=3)
        self.T_entry = ttk.Entry(master, width=10)
        self.T_entry.insert(0, "1.0")
        self.T_entry.grid(row=row, column=1, padx=5, pady=3)
        row += 1

        ttk.Label(master, text="Number of Paths (M):").grid(row=row, column=0, sticky=tk.E, padx=5, pady=3)
        self.M_entry = ttk.Entry(master, width=10)
        self.M_entry.insert(0, "50000")
        self.M_entry.grid(row=row, column=1, padx=5, pady=3)
        row += 1

        ttk.Label(master, text="Time Steps (N):").grid(row=row, column=0, sticky=tk.E, padx=5, pady=3)
        self.N_entry = ttk.Entry(master, width=10)
        self.N_entry.insert(0, "50")
        self.N_entry.grid(row=row, column=1, padx=5, pady=3)
        row += 1

        # Option type menu
        ttk.Label(master, text="Option Type:").grid(row=row, column=0, sticky=tk.E, padx=5, pady=3)
        self.opt_type_var = tk.StringVar(master)
        # We'll provide "EUROPEAN", "ASIAN", "BARRIER" in the dropdown
        self.opt_type_var.set("ASIAN")
        self.opt_menu = ttk.OptionMenu(
            master,
            self.opt_type_var,
            "ASIAN",
            "EUROPEAN",
            "ASIAN",
            "BARRIER"
        )
        self.opt_menu.grid(row=row, column=1, padx=5, pady=3)
        row += 1

        # Barrier level (if user selects BARRIER)
        ttk.Label(master, text="Barrier Level:").grid(row=row, column=0, sticky=tk.E, padx=5, pady=3)
        self.barrier_entry = ttk.Entry(master, width=10)
        self.barrier_entry.insert(0, "80.0")
        self.barrier_entry.grid(row=row, column=1, padx=5, pady=3)
        row += 1

        # Barrier Type menu
        ttk.Label(master, text="Barrier Type:").grid(row=row, column=0, sticky=tk.E, padx=5, pady=3)
        self.barrier_type_var = tk.StringVar(master)
        self.barrier_type_var.set("down-and-out")
        self.barrier_type_menu = ttk.OptionMenu(
            master,
            self.barrier_type_var,
            "down-and-out",
            "down-and-out",
            "down-and-in"
        )
        self.barrier_type_menu.grid(row=row, column=1, padx=5, pady=3)
        row += 1

        # Simulate Button
        self.sim_button = ttk.Button(master, text="Simulate", command=self.run_simulation)
        self.sim_button.grid(row=row, column=0, columnspan=2, padx=5, pady=10)
        row += 1

        # VolSurface Button (like before)
        self.volsurf_button = ttk.Button(master, text="Generate Vol Surface", command=self.generate_vol_surface_3d)
        self.volsurf_button.grid(row=row, column=0, columnspan=2, padx=5, pady=5)
        row += 1

        # Results label
        self.result_label = ttk.Label(master, text="", foreground="blue")
        self.result_label.grid(row=row, column=0, columnspan=2, padx=5, pady=5)
        row += 1

    def run_simulation(self):
        """
        Read user inputs, run simulate_GBM_paths, compute payoff depending on the
        option type selected, then display final results + 95% CI.
        """
        try:
            S0    = float(self.S0_entry.get())
            K     = float(self.K_entry.get())
            r     = float(self.r_entry.get())
            sigma = float(self.sigma_entry.get())
            T     = float(self.T_entry.get())
            M     = int(self.M_entry.get())
            N     = int(self.N_entry.get())
            opt_choice = self.opt_type_var.get()  # "EUROPEAN", "ASIAN", or "BARRIER"
            barrier_level = float(self.barrier_entry.get())
            barrier_type  = self.barrier_type_var.get()
        except ValueError:
            self.result_label.config(text="Invalid input! Check numeric fields.")
            return

        # 1) Generate paths
        S_paths, t_grid = simulate_GBM_paths(S0, r, sigma, T, M, N)

        # 2) Compute payoff based on user-chosen option type
        if opt_choice.upper() == "EUROPEAN":
            # We'll assume call or put is decided by "CALL" or "PUT" but let's just do a call for demo
            # Or you can add a separate call/put menu. For simplicity we use the "CALL" code. 
            price_est, stderr = payoff_european(S_paths, K, r, T, opt_type="CALL")

        elif opt_choice.upper() == "ASIAN":
            price_est, stderr = payoff_asian(S_paths, K, r, T, opt_type="CALL")

        elif opt_choice.upper() == "BARRIER":
            price_est, stderr = payoff_barrier(S_paths, K, r, T,
                                               opt_type="CALL",
                                               barrier_level=barrier_level,
                                               barrier_type=barrier_type)
        else:
            self.result_label.config(text=f"Option type {opt_choice} not recognized.")
            return

        # 3) Confidence Interval
        ci_lower = price_est - 1.96*stderr
        ci_upper = price_est + 1.96*stderr

        # 4) Display message
        msg = (f"{opt_choice.title()} Option Price: {price_est:.4f}\n"
               f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n"
               f"StdErr: {stderr:.4f}")
        self.result_label.config(text=msg)

        # 5) Plot some sample paths
        plt.figure(figsize=(7,4))
        for i in range(min(5, M)):
            plt.plot(t_grid, S_paths[i,:], alpha=0.7)
        plt.title(f"Sample GBM Paths ({opt_choice.title()} Option)")
        plt.xlabel("Time (years)")
        plt.ylabel("Underlying Price")
        plt.grid(True)
        plt.show()

    def generate_vol_surface_3d(self):
        """
        Generate a simple 3D vol surface for VANILLA calls only, 
        ignoring barrier logic. This part stays as in your previous code.
        """
        try:
            S0    = float(self.S0_entry.get())
            r     = float(self.r_entry.get())
            sigma = float(self.sigma_entry.get())
            M     = int(self.M_entry.get())
            N     = int(self.N_entry.get())
        except ValueError:
            self.result_label.config(text="Invalid input for vol surface.")
            return

        # small grid for K and T
        Ks = np.linspace(80, 120, 5)
        Ts = np.linspace(0.5, 2.0, 5)

        implied_vols = np.zeros((len(Ks), len(Ts)))
        for i, K_ in enumerate(Ks):
            for j, T_ in enumerate(Ts):
                S_paths, _ = simulate_GBM_paths(S0, r, sigma, T_, M, N)
                mc_price = price_vanilla_call_mc(S_paths, r, T_, K_)
                iv = implied_vol_bs_call(mc_price, S0, K_, r, T_)
                implied_vols[i, j] = iv

        Ks_mesh, Ts_mesh = np.meshgrid(Ks, Ts, indexing='ij')
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        X = Ks_mesh
        Y = Ts_mesh
        Z = implied_vols

        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        ax.set_xlabel("Strike (K)")
        ax.set_ylabel("Maturity (T)")
        ax.set_zlabel("Implied Vol")
        ax.set_title("3D Implied Vol Surface (Vanilla Calls)")

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Implied Vol")
        plt.show()
        self.result_label.config(text="3D Vol Surface (Vanilla) done!")


###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    root = tk.Tk()
    gui_app = MultiOptionGUI(root)
    root.mainloop()
