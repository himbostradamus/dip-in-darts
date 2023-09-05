
from nni.experiment import Experiment
import torch
torch.cuda.empty_cache()

search_space = {
    'lr': {'_type': 'uniform', '_value': [0.01, 0.15]},
    'buffer_sise': {'_type': 'choice', '_value': [300, 400, 500, 600, 700, 800, 900]},
    'patience': {'_type': 'choice', '_value': [50, 100, 150, 200, 250, 300]},
    'weight_decay': {'_type': 'loguniform', '_value': [5e-8, 1e-6]},
}

experiment = Experiment('local')

experiment.config.trial_command = 'python model.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

# experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 6
experiment.config.assessor.name = 'Medianstop'
experiment.config.assessor.class_args = {
            'start_step': 20
}
experiment.config.tuner.class_args = {
            'optimize_mode': 'maximize',
            # 'start_step': 5
        }

experiment.run(8889)