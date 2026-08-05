import numpy as np 
import pandas as pd 
houseprice = pd.read_csv('C:\Users\SW02\Desktop\Wine-Quality-Dataset\ML-03\Lab03\lab03\submission_house_price.csv')
sub  = pd.read_csv('C:\Users\SW02\Desktop\Wine-Quality-Dataset\ML-03\Lab03\lab03\submission.csv')
from pycaret.classification import *
houseprice.head()