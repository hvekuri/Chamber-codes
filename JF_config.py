# Used for saving results and plots
experiment = 'leikkuu'

# Meta data with experiment, collar name, start and end times for each measurement, collar heights (and LAI if used for GPP)
metadata_path = "/JF_example_files/JF_metadata.csv"

# Folder(s) where licor and rpi files are
folder_licor = "/JF_example_files/Data/"
folder_rpi = "/JF_example_files/Data/"

# Constants for fixing gas analyzer drift
drift_path = "/JF_example_files/DriftFix.xlsx"

# Scale GPP with LAI? True/False
use_LAI = False

# Quality control
# If PAR sd is greater than this, flux is not used for light response fitting, FEEL FREE TO CHANGE THE LIMITS
par_sd_limit = 150
rmse_limit = 1  # If RMSE is greater than this, flux is not used for LR fitting, FEEL FREE TO CHANGE THE LIMITS
#nrmse_limit = 0.05

# Plotting
plotting = True  # If False, no plots are shown

# Use exponential or linear fits for light response curves, "exponential"/"linear"
use_fit = "exponential"

# Where to put results
results_path = "/JF_example_files/"

# Collar area in m-2
collar_area = 0.348
# Gap in collar in m
gap = 0.035
