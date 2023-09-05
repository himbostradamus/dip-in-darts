
from nni.experiment import Experiment
import torch
torch.cuda.empty_cache()

search_space = {
    'lr': {'_type': 'uniform', '_value': [.08,.16]},
    'burnin_iter': {'_type': 'choice', '_value': [400, 500, 600, 700, 800]},
    'max_iter': {'_type': 'choice', '_value': [550, 700, 850]},
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