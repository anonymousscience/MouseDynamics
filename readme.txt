Python 3.6.1 |Anaconda custom (64-bit)


The main.py file contains two functions. Before running check the settings.py for proper settings

main_ACTION()
-modeling unit: action
-EVAL_TEST_UNIT = 0 #  test data are evaluated by session (class probabilities are averaged for all the actions belonging to the test session);
-EVAL_TEST_UNIT = 1 #  test data are evaluated by actions (class probability is computed for each action);


main_ACTION_SEQUENCE()
-modeling unit: action sequence
-number of actions used in a sequence - settings.py: NUM_ACTIONS
-offset for considering consecutive sequences - - settings.py: OFFSET




Balabit Data Set Plots - Actions in session files.

Training sessions plots:
https://drive.google.com/open?id=1uqI5t8N_PBTV52_pOyvZeDP591nySdhO

Test session plots:
https://drive.google.com/open?id=1eMIeMX8t5AZwo5NOBOl9HFiLtU189Pcj



