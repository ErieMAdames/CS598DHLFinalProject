Your contribution: SleepWakeTask (standalone task) for the existing DREAMT dataset in PyHealth.

Files to create/modify:

pyhealth/tasks/sleep_wake_task.py — core task implementation
docs/api/tasks/pyhealth.tasks.sleep_wake_task.rst — RST doc file
docs/api/tasks.rst — add your task to the index
examples/dreamt_sleep_wake_task_lstm.py — ablation/example script
tests/test_sleep_wake_task.py — tests using synthetic data


What the task needs to do (based on the paper):

Inherit from PyHealth's base task class
Segment each participant's overnight recording into 30-second epochs
Define input features from E4 signals (BVP, ACC, EDA, TEMP, HR, IBI)
Extract binary wake/sleep labels from PSG annotations (wake = positive class)
Drop epochs labeled "Missing"
Expose participant-level train/val/test splits (no subject leakage)
Return ordered time-series so temporal models (LSTM) can consume them


Ablation study in your examples script — you already planned two great ones:

Signal subset ablation: ACC-only vs BVP/HRV-only vs EDA+TEMP-only vs all signals
Label granularity: binary wake/sleep → 5-class (Wake/REM/N1/N2/N3)


Tests must:

Use synthetic/pseudo data only (2–5 fake patients, small tensors)
Run in milliseconds
Cover: sample processing, label generation, feature extraction, edge cases (e.g. missing labels, artifact epochs)





DREAMT dataset https://pyhealth.readthedocs.io/en/latest/api/datasets.html

repo https://github.com/WillKeWang/DREAMT_FE