#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='of0b68pg'
export NNI_SYS_DIR='/home/joe/nas-for-dip/C:/Users/Public/nni-experiments/of0b68pg/trials/sdacy'
export NNI_TRIAL_JOB_ID='sdacy'
export NNI_OUTPUT_DIR='/home/joe/nas-for-dip/C:/Users/Public/nni-experiments/of0b68pg/trials/sdacy'
export NNI_TRIAL_SEQ_ID='0'
export NNI_CODE_DIR='/home/joe/nas-for-dip'
export CUDA_VISIBLE_DEVICES='0'
cd $NNI_CODE_DIR
eval '/home/joe/.cache/pypoetry/virtualenvs/nas-test-OHy8kATa-py3.8/bin/python -m nni.retiarii.trial_entry py' 1>/home/joe/nas-for-dip/C:/Users/Public/nni-experiments/of0b68pg/trials/sdacy/stdout 2>/home/joe/nas-for-dip/C:/Users/Public/nni-experiments/of0b68pg/trials/sdacy/stderr
echo $? `date +%s%3N` >'/home/joe/nas-for-dip/C:/Users/Public/nni-experiments/of0b68pg/trials/sdacy/.nni/state'