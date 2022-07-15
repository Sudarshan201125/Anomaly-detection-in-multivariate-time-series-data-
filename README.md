# Anomaly-detection-in-multivariate-time-series-data-
###### Internship project at Coupa Software Pvt Ltd 
------

### OVERVIEW
-------
CPU_model.py - CNN model for CPU metrics (need to change input csv file location) \
MEM_model.py - CNN model for Memory metrics (need to change input csv file location)\
editcsv.py - To delete rows of CPU with condition and select first n rows \
CSVtoAVRO.py - To convert CSV file to AVRO file (but schema here is defined to convert CPU file) \
readAVRO.py - To read AVRO file as pandas dataframe 

### INSTALLATION 
-------------
pip install msda\
pip install pandas\
pip install numpy\
pip install matplotlib\
pip install datetime\
pip install statistics\
pip install torch\
pip install seaborn\
pip install sklearn\
pip install scipy\
pip install shap\
pip install keras\
pip install ipywidgets

### IMPORTANT LINKS
-----------
Research Paper [here](https://ieeexplore.ieee.org/document/8581424) \
Medium Blog [here](https://towardsdatascience.com/explainable-ai-xai-design-for-unsupervised-deep-anomaly-detector-6bd1275ed3fc)
