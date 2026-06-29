# _____________________________________________________________________________
#
# Proyecto:       Repair-in-a-Dynammic-Discrete-Choice-Model
#
# Script:         scripts/config.py
# Objetivo:
#
# Autor:          Rodrigo Antonio Aldrette Salas
# Correo(s):      raaldrettes@colmex.mx
#
# Fecha de 
# creación:       29/06/2026
#
# _____________________________________________________________________________


from dataclasses import dataclass
import jax.numpy as jnp

@dataclass(frozen=True)
class simconfig:

    # Simulation params
    CAR_UTILITY: jnp.ndarray = jnp.array([30, 20, 10])   # Utility of each car type
    NEW_PRICES: jnp.ndarray = jnp.array([200, 120, 80])   # Price of new cars
    SCRAP_PRICES: jnp.ndarray = jnp.array([20, 12, 8])   # Price of scrap cars
    MAINT_COSTS: jnp.ndarray = jnp.array([5, 3, 2])   # Maintenance costs of each car type

    TC_B: jnp.ndarray = jnp.array([10, 5, 4])   # Transition costs for each car type when buying
    TC_S: jnp.ndarray = jnp.array([10, 5, 3])   # Transition costs for each car type when selling

    # State space
    J: int = len(CAR_UTILITY)   # Number of car types
    MAX_A: int = 25   # Max number of years
    E_STATES: int = 4   # Number of quality states
    NSTATES: int = 1 + J * MAX_A * E_STATES   # Total number of states


    # Economic params
    BETA: float = 0.95   # Discount factor
    MU: jnp.ndarray = jnp.array([0.1, 0.3, 0.5])   # Wealth effect
    TYPEMASS: jnp.ndarray = jnp.array([0.5, 0.3, 0.2])   # Mass of each type of people
    Utypes: int = len(MU)   # Number of types of people


class estimconfig:

    # Estimation params
    NITER: int = 1000   # Number of iterations for estimation
    TOL: float = 1e-8   # Tolerance for convergence
    SEED: int = 42   # Random seed for reproducibility






