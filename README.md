## Installation Instructions 

Folders in the root directory that begin with 'py' are independent packages that need to be installed. The packages must be installed in the following order - pybasicbayes, pyhsmm, pyhsmm-autoregressive. To install, run the following command in each package folder
: ```python setup.py install```

Any additional package dependencies you run into can be installed with pip or conda. The file traj_seg_requiremens.txt contains the required packages and can be imported directly into a new virual environment.

## Model Implementations and Hyper-parameters
The implementations for all models used can be found in `trajectory_segmentation/models.py`. All hyper-parameters are set within the model classes themselves and do not need to specified anywhere else.

## Running the Experiments
The only script required to generate the results presented in the report is `trajectory_segmentation/main.py`. The lines to actually run the experiments are commented out because they take many many hours to run. However, the code to generate plots is left uncommented. 