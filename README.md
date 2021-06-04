# robustness_and_reliability_in_weak_supervision

Requirements can be installed using e.g. pip: python -m pip install -r requirements.txt

##For simple noise experiment:
The data used in the paper can be found under “experiments/toy_example/data”. 

Experiment without label noise:
Run python -m experiments.toy_example.toy_experiment --model_dir experiments/toy_example/models --data_dir experiments/toy_example/data --early_stopping

Experiment with label flip probability 0.2:
Run python -m experiments.toy_example.toy_experiment --model_dir experiments/toy_example/models --data_dir experiments/toy_example/data --early_stopping --label_noise 0.2

##For overfitting experiments:

The noisy targets used in the experiments can be found under experiments/overfitting_experiment/data/MNIST/targets. The shuffle order used for splitting the MNIST train data set into train and validation is saved in experiments/overfitting_experiment/data/training_indices.npy. Note that both files are dependent on the order in which the data is loaded. 

Training of CCE models: 

Run python -m experiments.overfitting_experiment.cce_experiment --model_dir experiments/overfitting_experiment/models --data_dir experiments/overfitting_experiment/data --lr 0.005 --label_noise 0.0

Evaluation of CCE models:

Run python -m experiments.overfitting_experiment.cce_evaluation --model_dir experiments/overfitting_experiment/models --data_dir experiments/overfitting_experiment/data --label_noise 0.0

Training of MAE models:

Run python -m experiments.overfitting_experiment.mae_experiment --model_dir experiments/overfitting_experiment/models --data_dir experiments/overfitting_experiment/data --lr 0.005 --num_epochs 5000 --save_interval 10 --label_noise 0.0

Evaluation of MAE models:

Run python -m experiments.overfitting_experiment.mae_evaluation --model_dir experiments/overfitting_experiment/models --data_dir experiments/overfitting_experiment/data --num_epochs 5000 --save_interval 10 --label_noise 0.0

For initialising a model from a previously trained model, specify the directory to the initial model using the “init_model” argument, for instance

Run python -m experiments.overfitting_experiment.mae_experiment --model_dir experiments/overfitting_experiment/models --data_dir experiments/overfitting_experiment/data --lr 0.005 --num_epochs 5000 --save_interval 10 --label_noise 0.0 --init_model experiments/overfitting_experiment/models/standard_discriminative_model_noise_0.0

Run python -m experiments.overfitting_experiment.mae_evaluation --model_dir experiments/overfitting_experiment/models --data_dir experiments/overfitting_experiment/data --num_epochs 5000 --save_interval 10 --label_noise 0.0 --init_model experiments/overfitting_experiment/models/standard_discriminative_model_noise_0.0

In all cases, change the “label_noise” argument (a value between 0 and 1) to train and evaluate models with label noise in the training (and validation) data. 


