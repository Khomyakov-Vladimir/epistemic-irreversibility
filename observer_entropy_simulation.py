"""
Numerical Example: Observer Entropy for Qubit Coupled to Thermal Bath

This code accompanies the manuscript:
"Epistemic Irreversibility: Observer Entropy and the Emergence of 
Thermodynamics from Quantum Information Loss"

Author: Vladimir Khomyakov
Code availability: Zenodo DOI [to be assigned upon publication]

Requirements: numpy, scipy, matplotlib
Install with: pip install numpy scipy matplotlib
"""

import numpy as np
from scipy.linalg import expm, logm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =============================================================================
# Pauli matrices and operators
# =============================================================================
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_p = (sigma_x + 1j*sigma_y) / 2  # raising operator σ⁺
sigma_m = (sigma_x - 1j*sigma_y) / 2  # lowering operator σ⁻
identity = np.eye(2, dtype=complex)


def dissipator(L, rho):
    """
    Lindblad dissipator superoperator: D[L](ρ) = LρL† - ½{L†L, ρ}
    
    Parameters
    ----------
    L : ndarray
        Lindblad jump operator
    rho : ndarray
        Density matrix
    
    Returns
    -------
    ndarray
        D[L](ρ)
    """
    Ld = L.conj().T
    LdL = Ld @ L
    return L @ rho @ Ld - 0.5 * (LdL @ rho + rho @ LdL)


def lindblad_rhs(t, rho_vec, H, gamma_down, gamma_up):
    """
    Right-hand side of the Lindblad master equation.
    
    dρ/dt = -i[H, ρ] + γ↓ D[σ⁻](ρ) + γ↑ D[σ⁺](ρ)
    
    Parameters
    ----------
    t : float
        Time (not used, included for solver compatibility)
    rho_vec : ndarray
        Flattened density matrix (length 4)
    H : ndarray
        System Hamiltonian
    gamma_down : float
        Decay rate (spontaneous + stimulated emission)
    gamma_up : float
        Excitation rate (absorption)
    
    Returns
    -------
    ndarray
        Time derivative of flattened density matrix
    """
    rho = rho_vec.reshape(2, 2)
    
    # Unitary evolution: -i[H, ρ]
    drho = -1j * (H @ rho - rho @ H)
    
    # Dissipation
    drho += gamma_down * dissipator(sigma_m, rho)
    drho += gamma_up * dissipator(sigma_p, rho)
    
    return drho.flatten()


def von_neumann_entropy(rho):
    """
    Compute von Neumann entropy S(ρ) = -Tr(ρ ln ρ)
    
    Parameters
    ----------
    rho : ndarray
        Density matrix
    
    Returns
    -------
    float
        Von Neumann entropy in natural units (k_B = 1)
    """
    eigvals = np.linalg.eigvalsh(rho)
    # Filter out numerical zeros to avoid log(0)
    eigvals = eigvals[eigvals > 1e-15]
    return -np.sum(eigvals * np.log(eigvals))


def thermal_occupation(omega, T):
    """
    Bose-Einstein occupation number.
    
    Parameters
    ----------
    omega : float
        Frequency
    T : float
        Temperature (in units where k_B = 1)
    
    Returns
    -------
    float
        Mean occupation number n̄
    """
    if T <= 0:
        return 0.0
    x = omega / T
    if x > 700:  # Prevent overflow
        return 0.0
    return 1.0 / (np.exp(x) - 1)


def simulate_qubit_dynamics(omega0, T_bath, gamma0, t_max, n_points=500):
    """
    Simulate qubit coupled to thermal bath under Lindblad dynamics.
    
    Parameters
    ----------
    omega0 : float
        Qubit transition frequency
    T_bath : float  
        Bath temperature (in units where k_B = 1)
    gamma0 : float
        Spontaneous emission rate at T=0
    t_max : float
        Maximum simulation time
    n_points : int
        Number of time points
    
    Returns
    -------
    times : ndarray
        Time array
    S_obs : ndarray
        Observer entropy at each time
    rho_trajectory : list of ndarray
        Density matrix at each time
    """
    # Temperature-dependent decay/excitation rates
    n_bar = thermal_occupation(omega0, T_bath)
    gamma_down = gamma0 * (n_bar + 1)  # emission (spontaneous + stimulated)
    gamma_up = gamma0 * n_bar          # absorption
    
    # System Hamiltonian H_S = (ω₀/2) σz
    H = (omega0 / 2) * sigma_z
    
    # Initial state: |ψ₀⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩ with θ = π/3
    theta = np.pi / 3
    psi0 = np.array([np.cos(theta/2), np.sin(theta/2)], dtype=complex)
    rho0 = np.outer(psi0, psi0.conj())
    
    # Time points
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, n_points)
    
    # Solve Lindblad equation
    sol = solve_ivp(
        lindblad_rhs, 
        t_span, 
        rho0.flatten(),
        args=(H, gamma_down, gamma_up),
        t_eval=t_eval, 
        method='RK45',
        rtol=1e-8,
        atol=1e-10
    )
    
    # Extract density matrices and compute entropy
    times = sol.t
    S_obs = np.zeros(len(times))
    rho_trajectory = []
    
    for i in range(len(times)):
        rho_t = sol.y[:, i].reshape(2, 2)
        rho_trajectory.append(rho_t)
        # For initially pure state: ΔS_obs(t) = S(ρ_S(t)) - S(ρ(0)) = S(ρ_S(t))
        S_obs[i] = von_neumann_entropy(rho_t)
    
    return times, S_obs, rho_trajectory


def equilibrium_entropy(omega0, T_bath):
    """
    Compute equilibrium entropy for qubit at temperature T.
    
    S_eq = -p₀ ln(p₀) - p₁ ln(p₁)
    where p₀ = 1/(1 + exp(-ω₀/T))
    """
    if T_bath <= 0:
        return 0.0
    p0 = 1.0 / (1.0 + np.exp(-omega0 / T_bath))
    p1 = 1.0 - p0
    S = 0.0
    if p0 > 1e-15:
        S -= p0 * np.log(p0)
    if p1 > 1e-15:
        S -= p1 * np.log(p1)
    return S


def extract_temperature(times, S_obs, gamma0, omega0):
    """
    Extract effective temperature from asymptotic dissipation rate.
    
    Uses the relation: β⁻¹ = T_obs ∝ lim_{t→∞} dS_obs/dt
    """
    # Fit to exponential approach to equilibrium: S(t) = S_eq(1 - e^{-Γt})
    # Find asymptotic value
    S_eq = S_obs[-1]
    
    # Extract relaxation rate from late-time behavior
    idx_half = len(times) // 2
    late_times = times[idx_half:]
    late_S = S_obs[idx_half:]
    
    # Linear regression on ln(S_eq - S) vs t
    mask = (S_eq - late_S) > 1e-10
    if np.sum(mask) > 2:
        y = np.log(S_eq - late_S[mask])
        x = late_times[mask]
        slope, _ = np.polyfit(x, y, 1)
        Gamma = -slope
        
        # Temperature from detailed balance: Γ = γ₀(2n̄ + 1) where n̄ = 1/(e^{ω/T} - 1)
        # Solving for T: T = ω₀ / ln((Γ/γ₀ + 1) / (Γ/γ₀ - 1))
        ratio = Gamma / gamma0
        if ratio > 1:
            T_obs = omega0 / np.log((ratio + 1) / (ratio - 1)) / 2
            return T_obs
    
    return None


def main():
    """
    Main function to generate Figure 3 and Table I from the manuscript.
    """
    print("=" * 70)
    print("Observer Entropy Simulation: Qubit + Thermal Bath")
    print("=" * 70)
    
    # Physical parameters (in natural units: ℏ = k_B = 1)
    omega0 = 1.0  # Qubit frequency
    T_bath = 1.0  # Bath temperature (k_B T / ω₀ = 1)
    
    # Coupling strengths to test
    coupling_strengths = [
        (0.01, 'blue', r'$\gamma_0/\omega_0 = 0.01$ (weak)'),
        (0.05, 'red', r'$\gamma_0/\omega_0 = 0.05$ (moderate)'),
        (0.10, 'green', r'$\gamma_0/\omega_0 = 0.10$ (strong)')
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Results storage for Table I
    results = []
    
    for gamma0, color, label in coupling_strengths:
        print(f"\nSimulating with γ₀/ω₀ = {gamma0}...")
        
        # Simulation time scales with thermalization time ~ 1/γ₀
        t_max = 8.0 / gamma0
        
        # Run simulation
        times, S_obs, _ = simulate_qubit_dynamics(
            omega0, T_bath, gamma0, t_max, n_points=500
        )
        
        # Plot (rescale time by γ₀ for comparison)
        ax.plot(times * gamma0, S_obs, color=color, label=label, linewidth=2)
        
        # Extract temperature
        T_obs = extract_temperature(times, S_obs, gamma0, omega0)
        if T_obs is not None:
            rel_error = abs(T_obs - T_bath) / T_bath * 100
            results.append((gamma0, T_bath, T_obs, rel_error))
            print(f"  Extracted T_obs = {T_obs:.4f} (error: {rel_error:.2f}%)")
    
    # Equilibrium entropy line
    S_eq = equilibrium_entropy(omega0, T_bath)
    ax.axhline(S_eq, color='gray', linestyle='--', linewidth=1.5, 
               label=r'$S_{\rm eq}$')
    
    # Formatting
    ax.set_xlabel(r'$\gamma_0 t$', fontsize=14)
    ax.set_ylabel(r'$\Delta S_{\rm obs} / k_B$', fontsize=14)
    ax.set_title(r'Observer Entropy Growth ($k_B T_{\rm bath}/\omega_0 = 1$)', 
                 fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 0.75)
    
    plt.tight_layout()
    plt.savefig('observer_entropy_figure.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('observer_entropy_figure.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved to 'observer_entropy_figure.pdf'")
    
    # Print Table I
    print("\n" + "=" * 70)
    print("TABLE I: Temperature Extraction Validation")
    print("=" * 70)
    print(f"{'γ₀/ω₀':>10} | {'k_B T_bath/ω₀':>14} | {'k_B T_obs/ω₀':>13} | {'Rel. Error':>10}")
    print("-" * 55)
    for gamma0, T_bath, T_obs, rel_error in results:
        print(f"{gamma0:>10.2f} | {T_bath:>14.1f} | {T_obs:>13.3f} | {rel_error:>9.1f}%")
    
    # Additional validation: different temperatures
    print("\n" + "=" * 70)
    print("Extended Validation: Different Bath Temperatures")
    print("=" * 70)
    gamma0_fixed = 0.01
    for T_test in [0.5, 1.0, 2.0]:
        t_max = 10.0 / gamma0_fixed
        times, S_obs, _ = simulate_qubit_dynamics(
            omega0, T_test, gamma0_fixed, t_max, n_points=500
        )
        T_obs = extract_temperature(times, S_obs, gamma0_fixed, omega0)
        if T_obs is not None:
            rel_error = abs(T_obs - T_test) / T_test * 100
            print(f"T_bath = {T_test:.1f}: T_obs = {T_obs:.3f} (error: {rel_error:.1f}%)")
    
    plt.show()
    
    return results


if __name__ == "__main__":
    results = main()
