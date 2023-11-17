''' Main pipeline script for model training 
Reads previously created training data inventory
Instantiates model and loads any pre-existing weights
Configures hyperparameters and optimization strategy according to config
Saves checkpoints, training log, and any other specified callbacks
Creates data generators for training/validation
Saves intermediate/final training model to output location
Optional running of validation stats (may require separate file)'''