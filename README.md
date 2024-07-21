<!-- --># Original Datasets
<!-- -->We conduct all experiments on four trajectories datasets: NYC and Tokyo which were collected from Foursquare and Los Angeles and Houston, which were extracted from Gowalla.

# Document description
We have three folders: data, model, and embedding.
In the data folder, we have uploaded examples to conduct case studies.  
In the embedding folder, we have the embedding file which we have generated. In the model folder we have model files, comparative learning embedding generation files, and other. Under current files, there are files for trainer files and data loader files, etc.

# Usage PAN
1. run 'pip3 install -r requirements.txt' to install dependencies
2. To achieve PAN model, run PAN_run.py --gamma=0.5 to start training and evaluation
3. The metrics will be stored in 'results' folder
4. for example: python PAN_run.py --gamma=0.5

