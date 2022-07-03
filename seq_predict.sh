python3 rm_predict_from_seq.py -sequence_file "seq_file.txt" \
-models_dir="./sequence_predict_models/" \
-output_dir="./output_dir/" \
-batch_size=32 \
-th="5" \
-prediction_type="mutation"

python3 rm_predict_from_seq.py -sequence "TACAAACCTTTTC>G" \
-models_dir="./sequence_predict_models/" \
-output_dir="./output_dir/" \
-batch_size=32 \
-th="5" \
-prediction_type="mutation"
