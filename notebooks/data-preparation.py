# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tasks performed in this notebook
# * Read raw data
# * Transform to time-series data per individual in long format
# * Normalize time-points to hourly values
# * Save time-series data in parquet format


# %%
# Import dependencies
import pandas as pd
import os
import shutil
import logging
from tqdm import tqdm


# %%
# Setup logging
logger = logging.getLogger('data-preparation-logger')
logger.setLevel(logging.DEBUG)

if logger.hasHandlers():
    # If the logger already has handlers, remove them
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

if not logger.hasHandlers():
    # Create a directory for logs if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.getcwd()), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Create a file handler
    log_file = os.path.join(log_dir, 'data-preparation.log')
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)

    # Create a console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.CRITICAL)

    # Create a formatter and set it for both handlers
    log_format = '%(asctime)s - %(name)s - %(levelname)s:\n\t\t%(message)s'
    formatter = logging.Formatter(log_format)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


# %%
# Preperations:
# Boolean determining whether the script is run on the full data or
# on a subset residing in subfolders with suffix -test
# If the testdata doesn't exist, it will be created.
# This is for debugging purposes only. The test data created is platform
# dependent because of the use of os.scandir(), hence the results
# are not reproducible.
test = True

# A list of all the time series variables to consider
LIST_VARIABLE_TS = [
    'ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin',
    'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
    'HCO3', 'HCT', 'HR', 'ICUType', 'K', 'Lactate', 'MAP', 'MechVent',
    'Mg', 'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets',
    'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine',
    'WBC', 'pH'
]

# A list of static variables
LIST_VARIABLE_STATIC = [
    'Age', 'Gender', 'Height', 'Weight'
]

# A list of keys:
# RecordID indexes the individual times series
# Hour is the time stamp to be used -- note that we will have to normalize
# the variable Time which is in format hh:mm after admission to hourly
# variables
LIST_VARIABLE_KEYS = [
    'RecordID', 'Hour'
]

# A list of all variables, including time series and static variables
LIST_VARIABLES = LIST_VARIABLE_KEYS + LIST_VARIABLE_STATIC + LIST_VARIABLE_TS

# fix types for the different variables
TYPE_STRING = ['RecordID']
TYPE_INTEGER = ['Hour']
TYPE_CATEGORICAL = ['Gender']
TYPE_FLOAT = (LIST_VARIABLE_TS + LIST_VARIABLE_STATIC).remove('Gender')

# Fix a threshold for zero
LOG_ZERO_OFFSET = 1e-6


# %%
# Fix path for data
DATA_PATH = os.path.join(os.path.dirname(os.getcwd()),
                         "physionet.org/files/challenge-2012/1.0.0/")
DATA_TRAINING = os.path.join(DATA_PATH, "set-a")
DATA_VALIDATION = os.path.join(DATA_PATH, "set-b")
DATA_TESTING = os.path.join(DATA_PATH, "set-c")
dataPaths = [DATA_TRAINING, DATA_VALIDATION, DATA_TESTING]

# if we want to run the code on a test subset, generate test data
if test:
    for dataPath in dataPaths:
        try:
            newFolder = dataPath + "-test"
            os.mkdir(newFolder)
            i = 0
            for entry in os.scandir(dataPath):
                if i < 100 and entry.name.endswith(".txt"):
                    i += 1
                    newPath = os.path.join(newFolder, entry.name)
                    shutil.copy(entry.path, newPath)
        except FileExistsError:
            logger.info(
                "Test data folder "
                f"{os.path.basename(os.path.normpath(dataPath))}-test "
                "already exists. Using existing data."
            )
            pass
    dataPaths = [p + "-test" for p in dataPaths]


# %%
# Here we define the functions used for the data transformation


# Constructor for the static dictionary:
# We produce a long format data frame per patient, where the static variables
# are treated as constants. Hence we keep track of them while producing the
# data frame and fill them in the end.
# - keys are the names of the static variables
# - initialized with values pd.NA
# Input:    - None
# Output:   - a dictionary with keys as defined in LIST_VARIABLE_STATIC
def initializeStaticDict():
    staticDict = {key: pd.NA for key in LIST_VARIABLE_STATIC}
    return staticDict


# Constructor for the patient data frame
# Initializes an empty data frame with a column per variable as defined before
# Input:    - None
# Output:   - a pandas.DataFrame with columns according to LIST_VARIABLES,
#             i.e., the list of variables defined above.
def initializeDataFrame():
    # Create an empty DataFrame with the specified columns
    df = pd.DataFrame(columns=LIST_VARIABLES)
    # we fix types so that we can safely initialize using pd.NA
    df = df.astype({
        col: (
            pd.StringDtype() if col in TYPE_STRING else
            pd.Int64Dtype() if col in TYPE_INTEGER else
            pd.CategoricalDtype() if col in TYPE_CATEGORICAL else
            pd.Float64Dtype()
        ) for col in df.columns
    })
    return df


# Constructor for a new row in the data frame
# Initializes a new row with pd.NA for all variables
# except the hour variable which is set to the given hour
# Input:    - hour = timestamp associated with the new row
# Output:   - a pandas.DataFrame new_row with a single row
#             initialized with values pd.NA and in the same
#             format as the output of initialize DataFrame for
#             concatenation.
def initializeNewRow(hour):
    # Create a new row with pd.NA for all variables
    new_row = pd.DataFrame(
        [[pd.NA] * len(LIST_VARIABLES)],
        columns=LIST_VARIABLES
    )
    # we fix types so that we can safely initialize using pd.NA
    new_row = new_row.astype({
        col: (
            pd.StringDtype() if col in TYPE_STRING else
            pd.Int64Dtype() if col in TYPE_INTEGER else
            pd.CategoricalDtype() if col in TYPE_CATEGORICAL else
            pd.Float64Dtype()
        ) for col in new_row.columns
    })
    new_row['Hour'] = hour
    return new_row


# We define a custom exception for data format errors; cf. preprocessLine
# This is used to catch cases where the input data is not provided
# in the expected format.
class DataFormatException(Exception):
    pass


# Supposedly every line in a data file has three comma-separated values:
#   - a time stamp in format hh:mm
#   - a variable name
#   - a value for the variable
# This function checks the format of a line and returns the split values. It
# raises a DataFormatException if the line does not contain exactly three
# values
# Input:    - A string
# Output:   - A list of three strings:
#             [timeStamp, variableName, value]
def preprocessLine(line):
    lineSplit = line.strip().split(",")
    if len(lineSplit) != 3:
        raise DataFormatException()
    return lineSplit


# extract the hour value from a time string
# Input:    - timeString in format hh:mm
# Output:   - hour as an integer, rounded up if minute > 0
def processTimeStamp(timeString):
    # timeString is in format hh:mm
    hour = int(timeString.split(":")[0])
    minute = int(timeString.split(":")[1])
    # normalize to hour, rounding up
    if minute > 00:
        hour += 1
    return hour


# We build the data frame in long format using one row per hour.
# Variables might not be available for all hours and we also allow
# the input data to be unordered.
# This function expands the data frame to include all missing hours
# between the last timestamp and the current hour.
# Input:    - hour = the current hour to be processed
#           - timestamp = the last processed timeStamp
#           - df = the data frame to be expanded
# Output:   - the expanded data frame with all hours from the last
def expandDataFrame(hour, timestamp, df):
    while timestamp < hour:
        # Check if the hour is already in the data frame
        condition = not (df['Hour'] == hour).any()
        if condition:
            timestamp += 1
            if not (df['Hour'] == timestamp).any():
                # Create a new row for the current timestamp
                new_row = initializeNewRow(timestamp)
                # Append the new row to the data frame
                df = pd.concat([df, new_row], ignore_index=True)
    return df


# create a data frame per patient
# Input:    - Path to an individual record's data
# Output:   - a pandas.DataFrame with the time series data for this individual
#             in long format with one row per hour
def patientDataFrame(dataPath):
    df = initializeDataFrame()
    data = open(dataPath)
    # first line is a header
    next(data)
    # read the remainder of the file line-by-line
    timestamp = -1
    # We first save the static variables in a dict
    # and fill up the rows later
    staticDict = initializeStaticDict()
    # now we parse the file line by line
    fileName = os.path.basename(dataPath)
    RecordID = ""
    # we keep track of the line number for logging purposes
    lineNumber = 1  # we ignore the header
    for line in data:
        lineNumber += 1
        try:
            lineSplit = preprocessLine(line)
            hour = processTimeStamp(lineSplit[0])
            df = expandDataFrame(hour, timestamp, df)
            timestamp = hour
            if lineSplit[1] in staticDict.keys():
                # This is a static variable, save it
                staticDict[lineSplit[1]] = lineSplit[2]
            elif lineSplit[1] == "RecordID":
                # This is the RecordID, save it
                RecordID = lineSplit[2]
            else:
                # This is a time-series variable, save it
                try:
                    value = float(lineSplit[2])
                except ValueError:
                    # If the value cannot be converted to float, skip it
                    message = (
                        f"Value in file {fileName} in line {lineNumber}"
                        f" cannot be converted to float: {lineSplit[2]}"
                    )
                    logger.warning(message)
                    continue
                # add the value to the data frame
                # Note: some values might have multiple observations within
                # one hour; we take the last one since we don't have enough
                # information to decide how to aggregate them
                df.loc[df["Hour"] == hour, lineSplit[1]] = value
        except DataFormatException:
            message = (
                f"Data format error in file {fileName}"
                f" in line {lineNumber}"
            )
            logger.warning(message)
            pass
    # add static variables
    for key, value in staticDict.items():
        df[key] = value
        # add RecordID and FileName
        df['RecordID'] = RecordID
    return df


# Create df in long format for each folder in dataPaths
# It is assumed that each folder contains multiple files, one txt
# file per patient, containing all the measured variables.
# Input:    - dataPath = path to the folder containing the patient
# Output:   - a pandas.DataFrame with the time series data for all patients
#             in dataPath using long format with one row per hour
def dataToLongFormat(dataPath):
    df = initializeDataFrame()
    for file in tqdm(os.listdir(dataPath)):
        if file.endswith(".txt"):
            # Create a data frame for the patient
            patient_df = patientDataFrame(os.path.join(dataPath, file))
            # Add the patient data frame to the main data frame
            df = pd.concat([df, patient_df], ignore_index=True)
    return df


# %%
# Transform data to frames in long format and save as parquet
# We produce one data frame per set according to the split into
# training, validation, and testing
for dataPath in dataPaths:
    logger.info(
        "Processing data in "
        f"{os.path.basename(os.path.normpath(dataPath))}..."
    )
    # Create the long format data frame
    df = dataToLongFormat(dataPath)
    # save data to parquet format
    outputFileName = os.path.basename(dataPath) + "_long_format.parquet.gzip"
    outputFilePath = os.path.join(dataPath, outputFileName)
    df.to_parquet(outputFilePath, index=False)
    logger.info(f"Saved {outputFileName} with {len(df)} rows.")
print("Data preparation completed successfully.")
logger.info("Data preparation completed successfully.")
