import os
import sys
import copy
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description='predict rm position or mutation from nucleotide sequence.')

parser.add_argument('-sequence_file', type=str,default='',
                    dest="ifile",help='A plain file with sequences with the format TGAAGTTGCTAGC>C. The last character is the mutation of nucleotide of the middle')
parser.add_argument('-sequence', type=str,default='',
                    dest="iseq",help='A sequence with the format TGAAGTTGCTAGC>C. The last character is the mutation of nucleotide of the middle')
parser.add_argument('-output_dir', type=str,default='./',
                    dest="output_dir",help='Directory where to export the predictions.')
parser.add_argument('-models_dir', type=str,default='./pos_mut_models/',
                    dest="models_dir",help='directory where the models mut and pos are stored.')
parser.add_argument('-batch_size', type=int,default=32,dest="batch_size",help='32')
parser.add_argument('-th', type=str,default="10",dest="th",help='RM threshold to predict')
parser.add_argument('-prediction_type', type=str,default="pos",dest="pos_mut",help='Prediction type. position or mutation.')

args = parser.parse_args()

def txt_seq_to_df_dic(t,w=6):
    o = {}
    for p in range(-w,w+1):
        tp = p+w
        for cn in "AGCT":
            l = f"P{p}_{cn}"
            if p == 0:
                l = f"P{p}{cn}>"
            v = 0
            if t[tp]==cn:
                v=1
            o[l]=v
    for cn in "AGCT":
        l = f"P0>{cn}"
        v = 0
        if t[-1]==cn:
            v=1
        o[l]=v
    return o.copy()

def txt_seq_to_df_row_data(itxt,valid_length=15):
    def get_random():
        nucs = [n for n in "AGCT"]
        rseq = np.random.choice(nucs,valid_length-2)
        rseq = "".join(rseq)
        _ = nucs[:]
        rseq +=">C"
        return rseq
    error_return = {}
    nucs = [n for n in "AGCT"]
    txt = itxt.upper()
    if len(txt)!=valid_length:
        rseq = get_random()
        print(txt,f"not valid length {len(txt)}. Valid sequence should be like:",rseq)
        return error_return
    #Check no valid characters
    valid_chars = "AGCTU>"
    not_valid_chars = [ c for c in set(txt) if c not in valid_chars]
    if len(not_valid_chars)>0:
        print("Invalid characters in sequence:",",".join(not_valid_chars),"Valid sequence has only:",valid_chars)
    #check presence of >
    if ">" not in txt:
        rseq = get_random()
        print("Invalid format. > not present. Format should be like:",rseq)
        return error_return
    txt = txt.replace("U","T")
    od = txt_seq_to_df_dic(txt)
    return od.copy()
    
def txt_lines_to_df(ifile="",itxt="",valid_length=15):
    """
    file ignore text
    """
    #
    error_return = pd.DataFrame()
    lines_to_seq = []
    if len(ifile)>0:
        print("Processing file",ifile)
        if os.path.isfile(ifile):
            with open(ifile,"r") as f:
                lines_to_seq = f.read().split("\n")
        else:
            print("File",ifile,"Does not exist")
            return error_return
    else:
        if len(itxt)>0:
            lines_to_seq = [itxt]
        else:
            print("No sequence nor file found in input.")
            return error_return
    #
    data = []
    for itxt in lines_to_seq:
        #print(itxt)
        o = txt_seq_to_df_row_data(itxt,valid_length)
        if len(o)>0:
            data.append(o.copy())
            see_cols = [ o[f"P0>{n}"] for n in "AGCT"]
            #print(see_cols)
    if len(data)<1:
        print("No valid data produced.")
        return error_return
    else:
        ocols ="P-6_A,P-6_G,P-6_C,P-6_T,P-5_A,P-5_G,P-5_C,P-5_T,P-4_A,P-4_G,P-4_C,P-4_T,P-3_A,P-3_G,P-3_C,P-3_T,P-2_A,P-2_G,P-2_C,P-2_T,P-1_A,P-1_G,P-1_C,P-1_T,P0>A,P0>G,P0>C,P0>T,P1_A,P1_G,P1_C,P1_T,P2_A,P2_G,P2_C,P2_T,P3_A,P3_G,P3_C,P3_T,P4_A,P4_G,P4_C,P4_T,P5_A,P5_G,P5_C,P5_T,P6_A,P6_G,P6_C,P6_T,P0A>,P0G>,P0C>,P0T>"
        ocols = ocols.split(",")
        df = pd.DataFrame(data)[ocols]
    return df.copy()

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
    dfx = txt_lines_to_df(ifile=iargs.ifile,itxt=iargs.iseq,valid_length=15)#1,56
    #
    if dfx.shape[0]<1:
        print("No valid data produced")
        sys.exit(0)
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
    odf = pd.DataFrame(data=d,columns=cols).copy().round(3)
    #
    od = args.output_dir
    #
    if od[-1]!="/":
        od+="/"
    #
    os.makedirs(od,exist_ok=True)
    f = iargs.ifile.split("/")[-1]
    of = f"{od}{f}_prediction_{iargs.pos_mut}_th{iargs.th}.csv"
    odf.to_csv(of,index=False)
    print("Prediction saved in ",of)
    return "complete"

def main():
    modelx = search_model_file(models_dir=args.models_dir,pred_pos_or_mut = args.pos_mut,th=args.th)
    if type(modelx)==type(None):
        sys.exit(1)
    load_data(args,modelx)

if __name__ == '__main__':
    main()