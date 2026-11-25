# Simulation size
N_BUSES = 1000
N_PERIODS = 60
N_REPETITIONS = 60

# True cost parameters
REPLACEMENT_COST = 15.0750
MAINTENANCE_COST = 0.0005293

# Discount factors
BETA_AGENT = 0.8
BETA_MODEL = 0.8

# Mileage grid
MILEAGE_STEP = 5000
MAX_MILEAGE = 350000

# Probabilities of mileage increases
P_ZERO = 0.3919
P_ONE_STEP = 0.5953
P_TWO_STEPS = 1 - P_ZERO - P_ONE_STEP