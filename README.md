# Knowledge Graph Completion

This repository holds the code and datasets for experiments in the paper (Please, cite these publication if you use our source codes and datasets)

There are three kinds of inference of missing data in a knowledge graph: 1) missing entity, 2) missing relation type and 3) missing entity type. Our code implement a E2T model and a E2T_TRT model to improve missing entity type prediction performance.

Author: Adam Zhang (anxiangz@andrew.cmu.edu)

Dependency Requirements: Python==3.7

The source codes are modified versions from original source code from 

Usage instructions:

\1. Parameter Setup

    

* 1.1 Related files

        1.1.1 ./run_E2T_FB.py (for our E2T model on FB15k dataset)

        1.1.2 ./run_E2T_TRT_FB.py (for our E2T-TRT model on FB15k dataset)

        1.1.3 ./run_E2T_YAGO.py (for our E2T model on YAGO43k dataset)

        1.1.4 ./run_E2T_TRT_YAGO.py (for our E2T-TRT model on YAGO43k dataset)

	1.1.5 ./config/*/config.json 

		All the files in the config directory are related to the parameter setup for the very model. For example, file "./config/FB15k/E2T/config.json" is for the parameter setup of the E2T model on the YAGO43k dataset.

* 1.2 Parameter setup

	All the files in the "config" directory contains the default parameter setup we use to train the model. For each parameter, the explanation is following:

- nbatch: number of batches to train in one epoch. Generally, we use vectorization mechanism to speed up the training process. If you increase this parameter, then it means you train more batches in one round of training. Consequently, you have to simultaneously calculate a larger matrix in on batch.(For all our model, the default value is 20.)
- learning_rate: the learning rate of the training model. (For all our model, the default value is 0.1.)
- entity_dim: the dimension of the entity/relation vector.
- type_dim: the dimension of the type vector.
- norm: the kind of norm we use to implement the loss function. (For all our model, we use L2 norm.)
- margin: Margin for loss function.
- evaluator: the kind of evaluator to use when evaluating the model. (We only implement type prediction evaluator, which is "Type_Evaluator")
- evaluator_time: number of epochs to train before one evaluation of the whole test sample. 

  \2. Inference of missing entity type (ETE model)

* 2.1 Related datasets

        2.1.1 Freebase: ../data/FB15k/origin/

        2.1.2 YAGO: ../datasets/YAGO/YAGO43k/

        2.1.3 ../datasets/YAGO/YAGO43k_Entity_Types/

* 2.2 How to run

	Before we run the model, we have to firstly clean the original dataset because the train/valid/test entity type data is constructed in a way that has some errors. Firstly, we should clean the entity type train data by erasing all the samples in the training set, in which the entity does not appear in the triplet training set. Second, we erasing all the samples in the valid/test set, in which the entity or entity type dose no appear in the entity type training set. Also, this would generate the type-relation-type dataset, which is needed for running E2T_TRT model.

	

        2.2.1 Prepare the dataset.

	    python ./data/FB15k/prepare_data_FB.py

	    python ./data/YAGO/prepare_data_FB.py

	2.2.2 Run the program

            python ./run_E2T_FB.py

	    python ./run_E2T_TRT_FB.py

	    python ./run_E2T_YAGO.py

	    python ./run_E2T_TRT_YAGO.py




