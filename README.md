# CSC4444-stock
Stock market predictor for AI/reinforcement learning class

Data comes from finnhub.io. Free API key was required to download. The data is the 1 minute OHLC(volume) data one the free tier. The data range from 1992 throught fevruary 2026. (march data has not yet been released as of 4.12.26)
Since this data is going to be used for multiple stock market projects, it lives in a folder at equal level to the root of the project. This will allow the same data to function for multiple projects easily. 
Container Folder
|- Project Folder
|- OHLC 1 minute data
  |- extracted files
    |- 1992
    |- 1993
    ...
    |- 2026
      |- 2026-01
      |- 2026-02
        |- TICKER.csv

The actual file name for TICKER.csv is formatted as the ticker in capital leters. For example Apple's ticker is AAPL, so the file for Apple's monthly level CSV is AAPL.csv. 

###Starting Steps###

1. Prerequisites: Ensure that Python 3.12-13 is installed.
2. Environment Setup: After unzipping the folder, navigate to the directory and set up your virtual environment. From your terminal:
  a. Create the environment using Python: "py -3.12 -m venv .venv"
  b. Activate it: ".\.venv\Scripts\activate"
  c. Install dependencies: "pip install -r requirements.txt"
3. Data Configuration: The system expects OHLC 1-minute data stored in a specific directory.
  a. Create a .env file in the root directory.
  b. Add the path to your extracted data (if you do not do this, it will use a default path): "DATA_PATH="C:/Users/YourName/Path/To/OHLC 1 minute data/extracted_files""
4. Running the System: The project is controlled via the main_runner.py script. You can toggle between Training and Testing modes by using the configuration switch located on line 23.
5. Training the Brain: In main_runner.py:
  a. Verify the dates you wish to use on lines 150 and 151.
  b. Verify that TRAINING_MODE = True.
  c. Run the script: "python main_runner.py".
  d. This will generate challenger_model.pth.
6. Testing the Brain: In main_runner.py:
  a. Verify the dates you wish to use on lines 154 and 155. Note these dates should be after your training dates.
  b. Verify that TRAINING_MODE = False.
  c. Ensure challenger_model.pth is in the root folder.
  d. Run the script: "python main_runner.py".
