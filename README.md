[![Build Status](https://travis-ci.com/rg314/autoballs.svg?token=BCkcrsWckKEnE7AqL2uD&branch=main)](https://travis-ci.com/rg314/autoballs)

[![codecov](https://codecov.io/gh/rg314/autoballs/branch/main/graph/badge.svg?token=35L8J85XO9)](https://codecov.io/gh/rg314/autoballs)

# autoballs
Pipeline for analysis of ex vivo explants


## Installation
While writing this package I tried my best not to reinvent the wheel for that reason there are a number of dependencies on [Fiji](https://imagej.net/Fiji). When installing the package please ensure that you create a conda env with relatvent [openjdk8](https://openjdk.java.net/install/) and [Maven](https://maven.apache.org/guides/getting-started/maven-in-five-minutes.html). Furthermore, due to constant updates of Fuji please use ```get_fiji_version.sh``` script to download the working plugin of Sholl analysis (v4.0.1).

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


## FIle structure
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

## Usage


## Example
pics/example_proc.png![example_proc](https://user-images.githubusercontent.com/35999546/109683001-d67ff700-7b76-11eb-980a-da1e136b2bb5.png)
