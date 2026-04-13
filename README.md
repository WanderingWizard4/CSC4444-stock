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
