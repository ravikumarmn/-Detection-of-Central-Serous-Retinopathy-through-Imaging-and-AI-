BASE_DIR =  "/home/ravikumar/freelance/computer_vision/"
MODEL_STR = 'cnn'
DATASET_NAME = "fundus" #'fundus' # macular 
WORKING_DIR = BASE_DIR + "CSR_detection/"
NORMAL_DATA_STR = "normal"#"OCTID_NORMAL" #normal"#"OCTID_NORMAL"
DISEASE_DATA_STR = "fundus"#"OCTID_MH" # fundus"#"OCTID_MH"
TEST_SPLIT_SIZE = 10
VAL_SPLIT_SIZE = 0.2
RANDOM_RESIZED_CROP = 224
DEVICE =  "cuda" #"cpu" # cuda
EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 32
PATIENCE = 5
WEIGHT_DECAY = 0.01#1e-3
RESULT_FILE = F"results/{DATASET_NAME}/performance_analysis.json"
# precision,recall,f1-score,confustion matrix,