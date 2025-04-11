# SFU CMPT 419 Project Template -- Replace with project title
This repository is a template for your CMPT 419 course project.
Replace the title with your project title, and **add a snappy acronym that people remember (mnemonic)**.

# Members

- Armaan Singh Chahal; 301559489
- Ekamleen Maan; 301555970
- Md Rownak Abtahee Diganta; 301539632
- Harry Gabbi; 301575215
- Henrik Sachdeva; 301563572



## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/EQtkw0aRKKxLv6Z9HjbeheMBTgQUvSGsXA66OyNGmjm8ZQ?e=aCWdmb) | [Slack channel](https://app.slack.com/client/T0866LNE29J/C086CRN5210) | [Project report](https://www.overleaf.com/project/676b837f0be019e9fe90e430) |
|-----------|---------------|-------------------------|


- Timesheet: Link your timesheet (pinned in your project's Slack channel) where you track per student the time and tasks completed/participated for this project/
- Slack channel: Link your private Slack project channel.
- Project report: Link your Overleaf project report document.

- [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/EQtkw0aRKKxLv6Z9HjbeheMBTgQUvSGsXA66OyNGmjm8ZQ?e=aCWdmb)
-  [Slack channel](https://app.slack.com/client/T0866LNE29J/C086CRN5210)
- [Project report](https://www.overleaf.com/project/676b837f0be019e9fe90e430)


## Video/demo/GIF
Record a short video (1:40 - 2 minutes maximum) or gif or a simple screen recording or even using PowerPoint with audio or with text, showcasing your work.


## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

This project has been tested on CSIL Workstation (for frontend and backend) and Ubuntu 20.04 EC2 instances (for training and testing) using the following exact versions:

### Front-end, Backend  and Final Output

- Flask==3.1.0  
- torch==2.6.0  
- torchvision==0.21.0  
- Pillow==11.1.0  
- reportlab==4.3.1  
- numpy==2.2.4  
- opencv-python==4.11.0.86  
- albumentations==2.0.5  
- tqdm==4.67.1  
- matplotlib==3.10.1  
- pandas==2.2.3  
- boto3==1.37.31  
- scipy==1.15.2  
- PyYAML==6.0.2  
- sqlalchemy==2.0.40  
- alembic==1.15.2  
- colorlog==6.9.0  
- typing-extensions==4.13.1  
- typing-inspection==0.4.0  
- packaging==24.2  
- pyparsing==3.2.3  
- python-dateutil==2.9.0.post0  
- MarkupSafe==3.0.2  
- six==1.17.0  
- filelock==3.18.0  
- networkx==3.4.2  
- fsspec==2025.3.2  
- stringzilla==3.12.3  
- simsimd==6.2.1  
- albucore==0.0.23  
- pydantic==2.11.3  
- pydantic-core==2.33.1  
- annotated-types==0.7.0  
- tzdata==2025.2  
- pytz==2025.2  
- Jinja2==3.1.6  
- Werkzeug==3.1.3  
- itsdangerous==2.2.0  
- click==8.1.8  
- blinker==1.9.0  
- setuptools==78.1.0

### Training and Testing on EC2 
- Python xx
- torch >= 1.13
- torchvision
- albumentations >= 1.3.0
- opencv-python
- numpy
- matplotlib
- tqdm
- boto3
- optuna

3. [Reproducing this project](#repro)


4. [Guidance](#guide)


<a name="demo"></a>
## 1. Example demo

A minimal example to showcase your work

```python
from amazing import amazingexample
imgs = amazingexample.demo()
for img in imgs:
    view(img)
```

### What to find where

Explain briefly what files are found where

```bash
repository
├── src                          ## source code of the package itself
├── scripts                      ## scripts, if needed
├── docs                         ## If needed, documentation   
├── README.md                    ## You are here
├── requirements.yml             ## If you use conda
```

<a name="installation"></a>

## 2. Installation

Provide sufficient instructions to reproduce and install your project. 
Provide _exact_ versions, test on CSIL or reference workstations.

```bash
git clone $THISREPO
cd $THISREPO
conda env create -f requirements.yml
conda activate amazing
```

<a name="repro"></a>
## 3. Reproduction
Demonstrate how your work can be reproduced, e.g. the results in your report.
```bash
mkdir tmp && cd tmp
wget https://yourstorageisourbusiness.com/dataset.zip
unzip dataset.zip
conda activate amazing
python evaluate.py --epochs=10 --data=/in/put/dir
```
Data can be found at ...
Output will be saved in ...

<a name="guide"></a>
## 4. Guidance

- Use [git](https://git-scm.com/book/en/v2)
    - Do NOT use history re-editing (rebase)
    - Commit messages should be informative:
        - No: 'this should fix it', 'bump' commit messages
        - Yes: 'Resolve invalid API call in updating X'
    - Do NOT include IDE folders (.idea), or hidden files. Update your .gitignore where needed.
    - Do NOT use the repository to upload data
- Use [VSCode](https://code.visualstudio.com/) or a similarly powerful IDE
- Use [Copilot for free](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal)
- Sign up for [GitHub Education](https://education.github.com/) 
