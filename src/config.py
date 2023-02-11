BASE_DIR =  "/home/Ravikumar/freelance/computer vision/"
MODEL_STR = 'cnn'
DATASET_NAME = "macular" #'fundus' # macular
WORKING_DIR = BASE_DIR + "CSR_detection/"
NORMAL_DATA_STR = "normal"#"OCTID_NORMAL"
DISEASE_DATA_STR = "csc"#"OCTID_MH"
TEST_SPLIT_SIZE = 10
VAL_SPLIT_SIZE = 0.2
RANDOM_RESIZED_CROP = 224
DEVICE = "cpu" # cuda
EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 8
PATIENCE = 5
WEIGHT_DECAY = 1e-5
RESULT_FILE = F"results/{DATASET_NAME}/performance_analysis.json"
# precision,recall,f1-score,confustion matrix,