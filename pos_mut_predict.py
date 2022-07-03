import os
import sys
import copy
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description='predict rm position or mutation.')

parser.add_argument('-data_file', type=str,default='./test_data_to_predict.csv',
                    dest="ifile",help='CSV file with the data to predict. Position prediction 611 columns, mutation prediction 803 columns')
parser.add_argument('-output_dir', type=str,default='./',
                    dest="output_dir",help='Directory where to export the predictions.')
parser.add_argument('-models_dir', type=str,default='./pos_mut_models/',
                    dest="models_dir",help='directory where the models mut and pos are stored.')
parser.add_argument('-batch_size', type=int,default=32,dest="batch_size",help='32')
parser.add_argument('-th', type=str,default="10",dest="th",help='RM threshold to predict')
parser.add_argument('-prediction_type', type=str,default="pos",dest="pos_mut",help='Prediction type. position or mutation.')
parser.add_argument('-file_separator', type=str,default=",",dest="sep",help='Character used as separation in the csv file')
parser.add_argument('-first_column_is_index', type=int,default=0,dest="index_first_col",help='set 1 if the first column is the index')
args = parser.parse_args()

def load_model(ij,ih,compile=False,learning_rate=0.01):
  """
  ij: input json model file
  ih: input h5 model file

  """
  from keras.models import model_from_json
  with open(ij, 'r') as jf:
    loaded_model_json = jf.read()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights(ih)
  if compile==True:
    from keras.optimizers import Adam,Adadelta
    optimizer = Adam(learning_rate,beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    loaded_model.compile(optimizer=optimizer,loss="categorical_crossentropy")
  return loaded_model

def load_model_file(load_model_file):
  model = None
  if len(load_model_file)>0:
    load_model_file = load_model_file
    load_model_file = load_model_file.replace(".h5","").replace(".json","")
    ih = load_model_file+".h5"
    ij = load_model_file+".json"
    if os.path.isfile(ih) and os.path.isfile(ij):
      print("Loading model.")
      load_default_model = False
      model = load_model(ij,ih,compile=True,learning_rate=0.01)
    else:
      print("Model not found. Exit.",load_model_file)
  return model

class ModelPath:
    def __init__(self,main_dir):
        md = main_dir
        if md[-1]!="/":
            md+="/"
        self.pos_d = md+"pos_pred_1_5_10_15/"
        self.mut_d = md+"mut_pred_5_10_15/"
        self.pos_model_paths={}
        self.mut_model_paths={}
    def search_models(self):
        ths = "1 5 10 15".split(" ")
        self.pos_model_paths = self.search_model_paths(self.pos_d,ths)
        ths = "5 10 15".split(" ")
        self.mut_model_paths = self.search_model_paths(self.mut_d,ths)
    def search_model_paths(self,idir,ths):
        #pos
        od = {}
        for th in ths:
            d = f"{idir}th{th}/"
            if os.path.isdir(d):
                h5_and_json_present = 0
                for f in os.listdir(d):
                    if f.endswith(".h5"):
                        h5_and_json_present+=1
                        p = f[:-3]
                    if f.endswith(".json"):
                        p = f[:-5]
                        h5_and_json_present+=1
                if h5_and_json_present==2:
                    f = d+p
                    od[th]=f
                else:
                    od[th]=""
            else:
                od[th]=""
        return od.copy()

def search_model_file(models_dir="./",pred_pos_or_mut = "mut",th="1"):
    pred_options = "position pos mut mutation".split(" ")
    if pred_pos_or_mut not in pred_options:
        print("Invalid prediction option, select between:",pred_options)
        return None
    if not os.path.isdir(models_dir):
        print(models_dir,"Is not a directory or does not exist")
        return None
    mp = ModelPath(models_dir)
    mp.search_models()
    if pred_pos_or_mut in pred_options[:2]:
        model_path = mp.pos_model_paths.get(th,"")
    else:
        model_path = mp.mut_model_paths.get(th,"")
    if len(model_path)==0:
        print("Invalid prediction threshold",th,"or model directory not present.")
        return None
    print("Loading model",model_path)
    model = load_model_file(model_path)
    return model

def collect_summary(i):
    global model_print
    model_print+=f"{i}\n"


def load_data(iargs,imodel):
    if not os.path.isfile(iargs.ifile):
        print(ifile,"does not exist")
        return None  
    index_col = None
    if bool(iargs.index_first_col):
        index_col=0
    dfx = pd.read_csv(iargs.ifile,sep=iargs.sep,index_col=index_col)
    data_index = dfx.index
    X = dfx.values
    xs = X.shape
    if len(xs)==1:
        X = np.expand_dims(X,0)
    xs = X.shape
    required_cols = np.array(imodel.input.shape)[-1]
    if xs[-1]!=required_cols:
        msg = f"Model require input of {required_cols}. But {xs[-1]} rows were passed from file {iargs.ifile}"
        print(msg)
        return None
    #
    total_rows = xs[0]
    batches = total_rows//iargs.batch_size
    o = []
    print("Prediction started")
    for batch in tqdm(range(batches+1)):
        sb = batch*iargs.batch_size
        eb = sb + iargs.batch_size
        if sb<total_rows:
            xb = X[sb:eb,:]
            px = imodel.predict(xb)
            o.append(px)
    px = np.vstack(o)
    ppx = np.expand_dims(px.argmax(1),-1)
    cols = [ f"{iargs.th}_{_}" for _ in "prob_0 prob_1 y_pred".split(" ")]
    d = np.hstack([px,ppx])
    odf = pd.DataFrame(data=d,columns=cols,index=data_index).copy().round(3)
    #
    od = args.output_dir
    #
    if od[-1]!="/":
        od+="/"
    #
    os.makedirs(od,exist_ok=True)
    f = iargs.ifile.split("/")[-1]
    of = f"{od}{f}_prediction_{iargs.pos_mut}_th{iargs.th}.csv"
    odf.to_csv(of,index=bool(iargs.index_first_col))
    print("Prediction saved in ",of)
    return "complete"

def main():
    modelx = search_model_file(models_dir=args.models_dir,pred_pos_or_mut = args.pos_mut,th=args.th)
    if type(modelx)==type(None):
        sys.exit(1)
    load_data(args,modelx)

if __name__ == '__main__':
    main()