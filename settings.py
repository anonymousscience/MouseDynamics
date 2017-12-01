# CASE = 'test' # training - evaluation is performed on a validation set which is created from the training data; test - evaluation is performed on the test part of the Balabit dataset

##############################
# INTERPOLATION: NO,LINEAR,POLINOMIAL,SPLINE
##############################
INTERPOLATION_TYPE='NO'
FREQUENCY = 0.05

##############################
# SESSION_CUT = 1  # modeling unit - sequence of actions
##############################
NUM_ACTIONS = 20
# OFFSET = NUM_ACTIONS//4 # integer division
OFFSET = NUM_ACTIONS

##############################
# SESSION_CUT = 2 # modeling unit - action
##############################
SESSION_CUT = 2
EVAL_TEST_UNIT = 0 #  0: test data are evaluated by session (class probabilities are averaged for all the actions belonging to the test session); 1 -- test data are evaluated by actions (class probability is computed for each action);
NUM_EVAL_ACTIONS = 30 # how many actions are used for decision - only for EVAL_TEST_UNIT = 1

##############################
# GENERAL settings
##############################

BASE_FOLDER = 'C:/_DATA/XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX/_Mouse-Dynamics-Challenge-master/'
TRAINING_FOLDER = 'training_files'
TEST_FOLDER   = 'test_files'
PUBLIC_LABELS = 'public_labels.csv'
TRAINING_FEATURE_FILENAME  = 'output/balabit_features_training.csv'
TEST_FEATURE_FILENAME  = 'output/balabit_features_test.csv'

##############################
# ACTIONS settings
##############################
GLOBAL_DELTA_TIME = 10       #DO NOT CHANGE
GLOBAL_MIN_ACTION_LENGTH = 4 #DO NOT CHANGE!!!
GLOBAL_DEBUG = False
CURV_THRESHOLD = 0.0005 # threshold for curvature

# action codes: MM - mouse move; PC - point click; DD - drag and drop
MM = 1
PC = 3
DD = 4

# temporary file
ACTION_CSV_HEADER = "type_of_action,traveled_distance_pixel,elapsed_time,direction_of_movement,straightness,num_points,sum_of_angles,mean_curv,sd_curv,max_curv," \
                    "mean_omega,sd_omega,max_omega,largest_deviation,dist_end_to_end_line,num_critical_points,"+\
                    "mean_vx,sd_vx,max_vx,mean_vy,sd_vy,max_vy,mean_v,sd_v,max_v,mean_a,sd_a,max_a,mean_jerk,sd_jerk,max_jerk,a_beg_time,n_from,n_to"+\
                    "\n"
ACTION_FILENAME = 'output/balabit_actions.csv'


##############################
# Raw data preprocessing
##############################
# RDP - Remote Desktop Window leaving
X_LIMIT=4000
Y_LIMIT=4000