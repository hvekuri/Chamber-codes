# Used for saving results and plots
experiment = "leikkuu"

# Meta data with experiment, collar name, start and end times for each measurement, collar heights
metadata_path = "JF_example_files/JF_metadata.csv"

# Folder(s) where licor and rpi files are
folder_licor = "JF_example_files/Data/"
folder_rpi = "JF_example_files/Data/"

# Constants for fixing gas analyzer drift
drift_path = "JF_example_files/DriftFix.xlsx"

# Quality control
# If PAR sd is greater than this, flux is not used for light response fitting, FEEL FREE TO CHANGE THE LIMITS
par_sd_limit = 150

# Check rmse/nrmse. If nrmse, the code checks first the magnitude of the flux. If the flux is small (absolute value of slope < 0.1, nrmse is not checked.)
check = "nrmse"

# If check = 'nrmse', if slope>0.1 and if NRMSE > nrmse_limit, flux is not used for LR fitting, FEEL FREE TO CHANGE THE LIMITS. Can be None, if check = 'rmse'.
nrmse_limit = 0.05

# If check = 'rmse', if RMSE > rmse_limit, flux is not used for LR fitting, FEEL FREE TO CHANGE THE LIMITS. Can be None, if check = 'nrmse'.
rmse_limit = None

# Plotting
plotting = True  # If False, no plots are made

# Use exponential or linear fits for light response curves, "exponential"/"linear"
use_fit = "exponential"

# Where to put results
results_path = ""

# Collar area in m-2
collar_area = 0.348
# Gap in collar in m
gap = 0.035

# Bounds for parameters: min, max
alpha_bounds = -0.1, -0.00000001
GPmax_bounds = -5, -0.0000001

GPP_light_levels = [100, 290, 680, 910, 1090, 1310, 1150, 550, 250, 90, 50]
