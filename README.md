[![Build Status](https://travis-ci.com/rg314/autoballs.svg?token=BCkcrsWckKEnE7AqL2uD&branch=main)](https://travis-ci.com/rg314/autoballs) 
[![codecov](https://codecov.io/gh/rg314/autoballs/branch/main/graph/badge.svg?token=35L8J85XO9)](https://codecov.io/gh/rg314/autoballs)

<p align="center">
  <img src="https://user-images.githubusercontent.com/35999546/109693171-44312080-7b81-11eb-812a-2659d07cd632.png" alt="Sublime's custom image"/>
</p>

Pipeline for analysis of ex vivo explants

Please note there are a few bugs I need to fix... for example when there's an eyeball and no axons or more than 1 eyeball in the FOV. This is work in progress.


## Installation
While writing this package I tried my best not to reinvent the wheel for that reason there are a number of dependencies on [Fiji](https://imagej.net/Fiji). When installing the package please ensure that you create a conda env with relatvent [openjdk8](https://openjdk.java.net/install/) and [Maven](https://maven.apache.org/guides/getting-started/maven-in-five-minutes.html). Furthermore, due to constant updates of Fuji please use ```get_fiji_version.sh``` script to download the working version.

```git clone git@github.com:rg314/autoballs.git```

Go into the autoballs firectory get the working Fiji version

```
source scripts/get_fiji_version.sh 
```

Create conda env

```
conda create -n autoballs python=3.8
```

Install autoballs in editable mode

```
pip install -e .
```


## File structure
Please note you will need to follow the following file conventions for this to work properly. I’ve not yet included the AFM section but I’m planning on adding this part in as well so there will be a later dependency. 

```
├── 20210226 Cam Franze
│   ├── afm
│   │   ├── data
│   │   │   ├── e01_nt_fg1_1force-save-2021.02.27-11.46.03.003.jpk-force
│   │   │   ├── ...
│   │   │   └── ve1_fg1_1force-save-2021.02.27-11.34.31.418.jpk-force
│   │   ├── datatxt
│   │   │   ├── e01_nt_fg1_1force-save-2021.02.27-11.46.03.003.txt
│   │   │   ├── ...
│   │   │   └── ve1_fg1_1force-save-2021.02.27-11.34.31.418.txt
│   │   └── run.py
│   ├── e01_nt_fg1_1
│   │   └── series.nd2
│   ├── e01_nt_fg2_1
│   │   └── series.nd2
│   └── metadata.txt
```

## Metadata
You will need to have a metadata file that store info about the frogs and gels for example 

```
{
'gel': 
{
'e01':'Elastic 100 Pa',
'e1':'Elastic 1 kPa',
've1':'Viscoelastic 1 kPa',
'glass':'Glass',
},
'frog': 
{
'fg1':'Upstairs outside box 176b 13:30 17/20',
'fg2':'Downstairs inside box 176b 13:30 17/20',
'fg3':'Downstairs outside box 182b 13:30 17/20',
'fg4':'Upstairs inside box 176b 13:30 17/20',
}
}
```

Note the subfolders need to have the same naming convention i.e. if ```e01_nt_fg1_1``` is processed the frog and gel metadata will be identified as

1) Upstairs outside box 176b 13:30 17/20
2) Elastic 100 Pa

## Usage

Check that the config is set-up correctly in ```scripts/pipeline_cnn.py```

You'll need to ensure that the path is pointing to the correct folder where each subdir has the file structure outlined above. 

```
def config():
    configs = dict()
    
    if sys.platform == "linux" or sys.platform == "linux2":
        path = '/media/ryan/9684408684406AB7/Users/ryan/Google Drive/TFM Cambridge/2021/Frogs'
    elif sys.platform == "win32":
        path = 'C:\\Users\\ryan\\Google Drive\\TFM Cambridge\\2021\\Frogs'
    elif sys.platform == 'darwin':
        path = '/Users/ryan/Google Drive/TFM Cambridge/2021/Frogs'
    

    configs['path'] = path
    configs['sample'] = '20210305 Cam Franze' #'20210226 Cam Franze'
    configs['metadata_file'] = f"{configs['path']}/{configs['sample']}{os.sep}metadata.txt"
    configs['metadata'] = biometa(configs['metadata_file'])
    configs['frog_metadata'] = configs['metadata']['frog']
    configs['gel_metadata'] = configs['metadata']['gel']
    configs['sholl'] = True
    configs['create_results'] = True
    configs['results_path'] = 'results' + os.sep + configs['sample'] + '_results'
    configs['seg'] = True
    configs['headless'] = True
    configs['step_size'] = 5
    configs['device'] = 'cuda'
    configs['best_model'] = './best_model_1.pth'
```

If you have all path envs and file structures set up correctly you can run the pipeline as 

```
python scripts/pipeline_cnn.py
```

A resuls folder will created and stats will be performed on the sample i.e. 

```
(autoballs) ryan@ryan:~/Documents/GitHub/autoballs$ python scripts/pipeline_cnn.py 
          Multiple Comparison of Means - Tukey HSD, FWER=0.05          
=======================================================================
    group1         group2      meandiff p-adj    lower    upper  reject
-----------------------------------------------------------------------
 Elastic 1 kPa Elastic 100 Pa   -36.075 0.6065 -129.2556 57.1056  False
 Elastic 1 kPa          Glass -108.3469 0.0091 -192.3387 -24.355   True
Elastic 100 Pa          Glass  -72.2719  0.103 -156.2637   11.72  False
-----------------------------------------------------------------------
          Multiple Comparison of Means - Tukey HSD, FWER=0.05          
=======================================================================
    group1         group2      meandiff p-adj    lower    upper  reject
-----------------------------------------------------------------------
 Elastic 1 kPa Elastic 100 Pa   -36.075 0.6065 -129.2556 57.1056  False
 Elastic 1 kPa          Glass -108.3469 0.0091 -192.3387 -24.355   True
Elastic 100 Pa          Glass  -72.2719  0.103 -156.2637   11.72  False
-----------------------------------------------------------------------

```
![image](https://user-images.githubusercontent.com/35999546/110236403-941b3900-7f2d-11eb-8769-5c23c5cc3b16.png)

## Example
![image](https://user-images.githubusercontent.com/35999546/110234828-8f9e5280-7f24-11eb-86ae-512017c80779.png)
