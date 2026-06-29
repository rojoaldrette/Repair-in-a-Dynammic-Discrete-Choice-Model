from dataclasses import dataclass
import jax.numpy as jnp

@dataclass(frozen=True)
class simconfig:
    # State space
    J: int = 3   # Number of car types
    MAX_A: int = 25   # Max number of years
    E_STATES: int = 4   # Number of quality states


    # Economic params
    BETA: float = 0.95   # Discount factor
    MU: jnp.ndarray = jnp.array([0.5, 0.3, 0.2])   # Wealth effect
    Utypes: int = len(MU)   # Number of types of people



