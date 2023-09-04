
search_space = {
    'lr': {'_type': 'uniform', '_value': [0.01, 0.15]},
    'buffer_sise': {'_type': 'normal', '_value': [700, 150]},
    'patience': {'_type': 'normal', '_value': [200, 50]},
}

from nni.experiment import Experiment
experiment = Experiment('local')

experiment.config.trial_command = 'python model.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 2

experiment.run(8889)