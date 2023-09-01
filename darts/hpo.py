from nni.experiment import Experiment

search_space = {
    'lr': {'_type': 'loguniform', '_value': [0.00001, 0.001]},
    'num_iter': {"_type": "choice", "_value": [50, 75, 100, 125, 150, 175, 200]},
    'buffer_size': {"_type": "choice", "_value": [25, 50, 100, 150, 200]},
}


experiment = Experiment('local')
experiment.config.trial_command = 'python main.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
# experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 1

experiment.run(8080)

# experiment.stop()