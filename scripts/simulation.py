# _____________________________________________________________________________
#
# Proyecto:       Repair-in-a-Dynammic-Discrete-Choice-Model
#
# Script:         scripts/simulation.py
# Objetivo:
#
# Autor:          Rodrigo Antonio Aldrette Salas
# Correo(s):      raaldrettes@colmex.mx
#
# Fecha de 
# creación:       29/06/2026
#
# _____________________________________________________________________________

import jax
import jax.numpy as jnp
from jaxopt import FixedPointIteration
from jaxopt import AndersonAcceleration

# Procedure --------------------
# 1. Define the utility for each state
# 2. Define the dynamics to get a transition matrix
# 3. Do the value functions with dynammics
# 4. Get fixed point
# 5. Solve for excess demand and get price vector







