---------------------INSTALLATION---------------------

Note: python should be any version of 3.10 to be sure all dependencies work.


[1] register the package:

    $ pip install -e .


[2] install requirements

    $ bash requirements.sh


---------------------RUNNING EXPERIMENTS FROM PAPER---------------------

From the master_thesis/ folder run:
(Note: these lines can also simply be added after "srun" in the provided train.sh file to be run on the Snellius cluster.)

[1] basic uncertainty weights 

    python -u train.py \
        --slr 0.0001 \
        --c 0.0025 \
        --train_metrics \
        --seed 1.0 \ # change seed for different runs
        --num_epochs 50 \
        --id _weights_1 \ # change for different runs
        --uncertainty basic_weights \
        --save_state > out_basic_weights_1.txt

[2] class normalized uncertainty weights

    python -u train.py \
        --slr 0.0001 \
        --c 0.0025 \
        --train_metrics \
        --seed 1.0 \ # change for different runs
        --num_epochs 50 \
        --id _class_weights_1 \ # change for different runs
        --uncertainty class_weights \
        --save_state > out_class_weights_1.txt


[3] baseline

    python -u train.py \
        --slr 0.0001 \
        --c 0.0025 \
        --train_metrics \
        --seed 1.0 \ # change for different runs
        --num_epochs 50 \
        --id _baseline_1 \ # change for different runs
        --save_state > out_baseline_1.txt


[4] uncertainty based stochastic sample drawing

    python -u train.py \
        --slr 0.0001 \
        --c 0.0025 \
        --train_metrics \
        --seed 1.0 \ # change for different runs
        --num_epochs 50 \
        --id _stochastic_1 \ # change for different runs
        --train_stochastic \
        --save_state > out_stochastic_1.txt