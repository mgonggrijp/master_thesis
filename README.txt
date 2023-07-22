---------------------INSTALLATION---------------------

Note: python should be any version of 3.10 to be sure all dependencies work.

[1] set ROOT variable at the top of segmenter.py, train.py and data_helpers.py to 
    path in your local system to the master_thesis folder. F.e. : 
        
            ROOT = /home/mats/master_thesis/

[2] register the package:

    $ pip install -e .


[3] install requirements

    $ bash requirements.sh


[4] download the data:

    for the base folders you can run

        $ python download_pascal.py

    which will automatically download and unzip the data. Then you have to get the augmented Pascal labels from 

        https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0

    Put the SegmentationClassAug folder in the data folder of Pascal such that the local path is:
        
        " datasets/pascal/data/SegmentationClassAug "


    (Additional data)

        Even though no experiments are ran on COCO, it can still be done as:

            $ download_coco.py

        Note that the annotations link doesn't work in this script and has to be downloaded manually from:

            https://github.com/nightrome/cocostuff#downloads --> Downloads --> stuffthingmaps_trainval2017.zip 

        And unzipped into:
        
                 " datasets/coco/data/ "

        To run with COCO simply add a flag to the bash command "$ python -u ... "

            --dataset coco



---------------------RUNNING EXPERIMENTS FROM PAPER---------------------

From the master_thesis/ folder run:
(Note: these lines can also simply be added after "srun" in the provided train.sh file to be run on the Snellius cluster.)

[1] basic uncertainty weights 

    python -u train.py \
        --slr 0.0001 \
        --c 0.0025 \
        --train_metrics \
        --seed 1.0 \                # change seed for different runs
        --num_epochs 50 \
        --id _weights_1 \           # change for different runs
        --uncertainty basic_weights \
        --save_state > out_basic_weights_1.txt

[2] class normalized uncertainty weights

    python -u train.py \
        --slr 0.0001 \
        --c 0.0025 \
        --train_metrics \
        --seed 1.0 \                # change for different runs
        --num_epochs 50 \
        --id _class_weights_1 \     # change for different runs
        --uncertainty class_weights \
        --save_state > out_class_weights_1.txt


[3] baseline

    python -u train.py \
        --slr 0.0001 \
        --c 0.0025 \
        --train_metrics \
        --seed 1.0 \                # change for different runs
        --num_epochs 50 \
        --id _baseline_1 \          # change for different runs
        --save_state > out_baseline_1.txt


[4] uncertainty based stochastic sample drawing

    python -u train.py \
        --slr 0.0001 \
        --c 0.0025 \
        --train_metrics \
        --seed 1.0 \                 # change for different runs
        --num_epochs 50 \
        --id _stochastic_1 \         # change for different runs
        --train_stochastic \
        --save_state > out_stochastic_1.txt