from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]  %(message)s"
logging_dir = "logs"
os.makedirs(logging_dir, exist_ok=True)
logging.basicConfig(filename = os.path.join(logging_dir,"running_logs.log"), level=logging.INFO, format=logging_str, filemode='a')

def main(data, eta, epochs, filename, plotFileName):
    
    df = pd.DataFrame(data)
    logging.info(f"this the actual data frame {df}")

    X , y = prepare_data(df)
    
    model_and = Perceptron(eta = eta, epochs=epochs)
    model_and.fit(X,y)

    _ = model_and.total_loss()

    save_model(model_and, filename=filename)
    save_plot(df,file_name=plotFileName, model=model_and)


if __name__ == '__main__':
    AND ={
        "x1":[0,0,1,1],
        "x2":[0,1,0,1],
        "y" :[0,0,0,1]
    }
    ETA = 0.01
    EPOCH = 100
    try:
        logging.info("<<<<<<<<<< started traing >>>>>>>>>>>")
        main(data=AND, eta=ETA, epochs=EPOCH,filename= "and.model",plotFileName= "and.png")
        logging.info("<<<<<<<<<< training done >>>>>>>>>>>\n")
    except Exception as e:
        logging.exception(e)
        raise e
