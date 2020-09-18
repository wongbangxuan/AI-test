# AI-test

![](https://gallery.mailchimp.com/f98d5ac0a3fbbdcdda35136ab/images/c1c0c438-0576-42ce-84ea-2fa1d1d6cde2.jpg)
# Threats-Mitigation Using Semi-Supervised LSTM model
This readme has 2 target audience: <span style="color:LightGreen"> Sponsor Team </span> (a) and <span style="color:magenta">PI Team</span> (b). Where there are specific instructions, refer to the relevant section. If there are some broken image links, please refer to the `/assets/images` folder to view the images.

# Table of Contents

1a. [Quick Start <span style="color:LightGreen">Sponsor Team</span>](#quick-start-sponsor)  
1b. [Quick Start <span style="color:magenta">PI Team</span>](#quick-start-pi)  
2. [Project Overview](#overview)  
3. [Configuration](#config)   
4. [Supplementary Notebooks](#notebooks)  
5. [FAQs](#faq)  
6. [Possible Future Research Directions](#future-directions)

<a name="quick-start-sponsor"></a>
# 1a. Quick Start (<span style="color:LightGreen">Sponsor Team</span>)

This section aims to quickly enable the user to train/run the model. However, additional steps will be required to transform the data from client operational network to the desired form required by this prototype.
Deeper discussions of the methodology, findings can be found in other sections of this README document.   

Scripts and modules belonging to this repository has been tested on the following VM configuration:
+ Azure NC6v2 with P100 GPU, Ubuntu 18.04.3 LTS (comes with 112GiB RAM)

It assumes the following dependencies are installed within the VM:
+ Nvidia GPU drivers
+ Docker
+ Nvidia-docker

Refer to the [FAQ section](#faq) on installing these dependencies.

### Setup
Git clone the [repository](https://bitbucket.ai3.aisingapore.org:9443/projects/TOFFTM/repos/threats-mitigation-base/browse) to the Linux machine that the experimentation would be conducted on. The `threats-mitigation-base` repo folder __must__ be renamed as `threats-mitigation-base-v20`.
```bash
git clone <repo>
cd scripts
bash toffs_build_train.sh
bash toffs_build_infer.sh
```  
The bash scripts will build the necessary Docker containers, `toffs_train:1.0` and `toffs_infer:1.0`. Necessary libraries will be read in from `config/requirements.txt` during docker build.

__`To train model:`__  
Requirements:  
+ __1 single csv file__ containing __only__ the subdomain-level traffic logs to be trained on

It is recommended to have 2-3 days of training data for the subdomain. The data __should not__ contain any known DDOS traffic. As a start, we suggest training the model once a week.    

Columns that are necessary:  
1. timestamp
2. remote_addr
3. request_uri
4. status
5. http_user_agent
6. country_code

Create a folder with unique `folder_name` under the `/data` folder and put the `.csv` training file in this folder.

<!-- Place the (big batch) `.csv` file in the host machine's cloned repo, under the `/data` folder.
Change the following settings of the configuration file, `/src/config.yaml`:
1. Change the `uniqueID` and `name` under `metadata`. The output data would be saved in a new folder named `uniqueID`.
2. Change `path` under `datapath` to be the datapath to the `.csv` file. -->

Then run the following command from the root of the `threats-mitigation-base` repo folder:
```bash
bash scripts/toffs_run_train.sh [folder_name] [training_file_name]
```
The bash scripts will call Docker to run the `toffs_train` container and do a folder bind of the repository in the VM to the relevant location in the Docker container. A new folder name `artefact` would be created to save the trained model.

__`To run inference model:`__  
Requirements:  
The algorithm __should not__ be run during periods of normal operations because it will always rank users by suspicion level, and thus false positives will always be generated. During DDOS period, send `.csv` files of the logs (suggested: 5 minute intervals) to the previously created `data/folder_name` folder of the repo in the host machine.

<!-- The algorithm __should not__ be run during periods of normal operations because it will always rank users by suspicion level, and thus false positives will always be generated. During DDOS period, send `.csv` files of the logs (suggested: 5 minute intervals) to the `data/inference` folder of the repo in the host machine. The model assumes the timestamps are in order, i.e. entries that should be in the earlier log files are included in the later file. Currently, assumption is made for the absence of duplicates in the log files. The log files must be named in the following format to enable the model to easily infer the oldest file to read in order. Reading the files out of order could result in wrong intepretation of results:

`yyyy-mm-dd-hhmmss_to_yyyy-mm-dd-hhmmss.csv`
(where the timestamp refers to the oldest timestamp in the file) For instance, `2019-08-13-220000_to_2019-08-13-220500.csv` -->

Columns that are necessary:  
1. timestamp
2. remote_addr
3. request_uri
4. status
5. http_user_agent
6. country_code

Note: Only the trained subdomain should be in the logs as the model does not filter out any unwanted hosts.  

Then, run the following from the root of the threats-mitigation-base folder:
```bash
bash scripts/toffs_run_infer.sh [folder_name] [training_file_name] [attack_file_name]
```  
Note that the `training_file_name` is only needed  to preprocess the attack file and not for another round of training.

The bash script will call Docker to run the `toffs_infer` container and do a folder bind of the repository in the VM to the relevant location in the Docker container. If it does not yet exist, a folder named `result` will be created to save the output files. The output scores would be saved in a `.csv` file named `PScore_[attack_file_name]`.
<!-- The code will continuously loop thru any files found in the `/data/inference` folder, moving the `.csv` files that has been processed to `/data/inference/processed`. This is a single thread process loop. -->

<!-- A screenshot of how the inference folder files looks like during internal testing is as such:  

<img src="assets/images/1-inference-folder.png" width="300" align="middle"/> -->

<!-- User scores, based on the latest processed batch of data, can be found under `/artefacts/inference`. -->
The scores reflect the latest assessment of each `IP` and thus the scores is expected to change over time.  

__Sample User Score:__  

<!-- (Interpretation) -->
<!-- *EDITED after last meeting on 17th Dec, a new column, user_score_percentile has been added.*  

| remote_addr | last_seen             | score | user_score_percentile |
|-------------|-----------------------|-------|-----------------------|
| 56.2.156.2  | 2019-08-12 12:01:05   |  0.3  |   0.3                 |
| 52.2.155.1  | 2019-08-12 12:02:03   |  0.6  |   0.5                 |
| 52.2.15.9   | 2019-08-12 12:02:03   |  0.9  |   0.95                 |    -->

| IP | P |
|-------------|-------|
| 56.2.156.2  |  0.3  |
| 52.2.155.1  |  0.6  |
| 52.2.15.9   |  0.9  |

<!-- There should not be any duplicate `remote_addr` in each `.csv`.   -->
<!-- User scores are float values between 0 and 1, where 0 is closer to __normal__ and 1 is __abnormal__. We recommend using default of score <= 0.5 to decide if the user is normal or abnormal. Users that are not seen in the last 1 hour will be dropped from user scores.   
It is not guaranteed to have a wide spread of scores from 0 to 1, hence, we included an additional float column, user_score_percentile, that ranks the remote_addresses from 1 to 0, 1 being __most abnormal__, and 0 the __least abnormal__, such that the operator can decide to filter out users using percentile. -->

User scores are float values between 0 and 1, where 0 is closer to __abnormal__ and 1 is __normal__. Lower score means that the data is more likely to be an attack

<!-- <img src="assets/images/2-user-score.png" width="900" align="middle"/>   -->

Comparing Model Performance with Operator Actions  
Currently, we understand that there are human monitoring and interventions by operators. Hence, one way to assess the performance of the model is to assess what is the user score for the manually flagged blacklisted addresses, and whether or not the algorithm ranks them <0.5, and also to check if the top ranked remote_addrs by the model is logical to the human operator. For post-mortem feedback, we suggest collecting the manually marked addresses (which can serve as labels) together with the training data and inference data and sending it over to the PI team as additional datasets to their approaches.  

Ways to improve model:   
Provide more training data to the model. (and should it not work, it would also help the PI team by giving them more training data to train their subsequent algorithms).    

<!-- Pictorial Summary to Load Data into Model:  
<img src="assets/images/27-load-data-to-vm-toffs.png" width="900" align="middle"/>   -->

<a name="quick-start-pi"></a>
# 1b. Quick Start (<span style="color:magenta">PI Team</span>)

Scripts and modules pertaining to this section has also been tested on the following VM configuration:
+ Azure NC6v2 with P100 GPU, Ubuntu 18.04.3 LTS (comes with 112GiB RAM)

It assumes the following dependensices are installed within the VM:
+ Nvidia GPU drivers
+ Docker
+ Nvidia-docker

Refer to the [FAQ section](#faq) on installing these dependencies.

The repository can be located here [repository](https://bitbucket.ai3.aisingapore.org:9443/projects/TOFFTM/repos/threats-mitigation-base/browse)  

Data and metadata to be given to the model can be located in the Toffs Azure blob storage container "transfer-to-pi"  
<img src="assets/images/3-azure-container.png" width="500" align="middle"/>

### Setup
```bash
git clone <repo>
cd threats-mitigation-base/scripts
bash PI_build_evaluation.sh
```
The bash scripts will build the necessary Docker container: `toffs_pi:1.0`. Necessary libraries will be read in from `config/requirements.txt` during docker build.



### For experimentation by PI team
Due to size, the data repository can be found in the Azure Toffs blob container under the following folders:
1. `all_raw_from_toffs` (the zipped csvs with multiple domains in each csv. Refer to readme.txt in folder)
2. `Parquet Files` (the parquet files used for experimentation)  

The parquet files should be used for experimentation:
1. To run an experiment, put a parquet file inside the `./data/evaluation` folder of the VM and specify the relevant filename in `configuration.yml`
2. Download `/metadata` folder from Azure blob and put it in the `threats-mitigation-base` root folder as such:
<pre>
 |- threats-mitigation-base
    | - data
        |- metadata
           | - attack_timestamps.json
           | - GS_actual_label.csv
           | - 888.abcb11.com_actual_labels.csv
</pre>
3. Modify `./config/model_config.yml` (config file):
  a. `evaluation:filename`
  b. `evaluation:training_period`
  c. `evaluation:inference_period`  
  d. `evaluation:output_dir`
  e. If the parquet is not one of the labeled datasets, please use 'pgt' in `evaluation:label` instead of 'agt'.
5. In the repo's root folder, run the following:
```bash
bash scripts/PI_run_evaluation.sh -c <path to model_config.yml>
```
(for instance `./config/model_config.yml`)

While the `.yml` config file can have other names, it must be located in the `/configuration` folder. The code will mount the repo folder to Docker container and run. It will also output the runtime messages (logs) to console.
The bash scripts will call Docker run on the `toffs_train` container and do a folder bind of the repository in the VM to the relevant location in the Docker container.

The following outputs are generated from a successful run:  
1. traffic.html
2. user_level_metrics using prod method.csv
3. metrics.csv
4. prc_roc.png
5. scores.png
6. model.h5
7. run_evaluation.log
8. outputs.pickle (if debug is enabled under `model_config.yml`)  

If model parameters are to be edited, please refer to the section [3. Configuration](#config).   
Pictorial Summary to Load Data into Model:  
<img src="assets/images/28-load-data-to-vm-pi.png" width="900" align="middle"/>  
<a name="overview"></a>
# 2. Project Overview  

## Project Background

### Abstract
In order to manage the disruptive effects of a DDOS attack on website availability, it is proposed that a deep neural network model be created to identify and prioritize web traffic users. Using a semi-supervised LSTM approach to detect anomalous traffic, the team was able to achieve 85% average F-1 scores on 1 dataset and 71% average F-1 scores on another. A MVP was developed for the client to perform limited testing the model against operational traffic.

To be delivered at end of project:  
A deep neural network model that enables the classification of users as either malicious attackers or normal web users during an attack.

### Data Available
Overview of the datasets collected to date  
<img src="assets/images/4-dataset-overview.png" width="900" align="middle"/>  

2 datasets are labeled. `GS_55555964.com_stage3.parquet` and `June_888.abcb11.com_stage3.parquet`
These 2 datasets were among 3 datasets chosen, because according to psuedo ground truth, "GS" has more normal reqt than attack and the "Jun" dataset has overwhelmingly more attack requests than normal request. Thus we choose these 2 contrasting datasets to label manually. A 3th dataset CF13888.com which was a "balanced" dataset, was initially chosen but not labeled due to lack of time.

In the proposal, Web Application Firewall (WAF) logs were mentioned but Toffs mentioned that they do not normally activate WAF even during attacks, so there is no WAF logs to supplement the dataset.

For PI Team, it is recommended to use the parquet files because the raw csv files are very large and Pandas sometimes throws error ingesting the files. Databricks was used to ingest the raw files and subdomain specific parquet files are generated. A point to note is that the raw data rows have UTC timezones, but for the sake of simplicity in analysis and understanding, we shifted the timestamps to SGT (+8 UTC) when generating the parquets. This is because the client generated the data from 00:00hrs SGT  to 23:59:59hrs SGT when providing the dataset.

### Labels
To allow us to judge the performance of the models, labels are required for batch training and inference on historical data. At the start of the project, a pseudo ground truth method was used to label the users. Since we know the approximate period of attack (refer to `./data/metadata/attack_timestamps.json`), a rule was set that remote_addr outside of the attack periods are normal users;consequently, new users that are only seen during the attack are considered attackers.
Subsequently, a decision was made to manually label some datasets to give greater confidence to the inference, as some attacker patterns are captured in the "normal" data and we might end up classifying bots as normal.
We decided to label the data manually; an on-site discussion with toffs was conducted to find out how Toffs operations staff analyze and blacklist web traffic based on visiting patterns, ie, repeated visit.

### Approach Hypothesis
Anomaly detection + sequence to sequence as the basis for flagging out suspicious requests. We define users at the remote address level.   
The model is an unsupervised model that is trained on "normal period" data and thus malicious requests are less likely to be predicted well compared to normal user requests.
__Caveat:__ IP level is as granular as the model could go, however, it is possible that the IP address could represent a group of users sharing the same gateway IP address.  

### Model
Model employed in this project is an LSTM model. The model will take in a sequence of requests from each IP and predicts the contents of the next request from the same user. The difference between the next request and the predicted request generates a "score" for each request.

It takes in 4 features from raw data:
1. `http_user_agent`
2. `status_code`
3. `country_code`
4. `request_uri`
and 1 derived feature:
5. `time_diff` (time between requests coming from the same IP)

It predicts characteristics of the next request:
1. `status_code`
2. `country_code`
3. `request_uri`
4. `time_diff` (i.e. when the next request will arrive)

### Loss Function
- `country_code`: sparse categorical cross entropy  
- `http_user_agent`: sparse categorical cross entropy  
- `request_uri`: sparse categorical cross entropy  
- `status code`: categorical cross entropy  
- `time_difference`: mean squared error  

Sparse categorical cross entropy is used to minimise memory consumption, because the data is typically transformed using one-hot representation, which generates a sparse array. However, Keras requires the dense array for categorical cross entropy. So Instead, we use sparse categorical cross entropy to compare the dense output against integer labels.

### Request Scoring
Each incoming request after 10 observations has a corresponding prediction, the difference between the actual and predicted forms the basis to categorise the request as a malicious or normal request.

### Aggregation to User Score  
<img src="assets/images/5-aggregation.png" width="600" align="middle"/>

The section above classifies requests as malicious or normal request, however, the objective is to classify users. For each `remote_addr`, normalize the request score (refer to `src/metrics.py`, `aggregate_user_traffic_metrics_prod`) and we take the product of the 1st predicted request to the *n*th predicted request.    

### Deliverables
High level overview of pipelines for evaluation (PI team) and deployment (PS team):
<img src="assets/images/6-architecture-pic.png" width="900" align="middle"/>  

The two pipelines are intended to be used together in the following manner:  
<img src="assets/images/7-config-use.png" width="500" align="middle"/>

<a name="config"></a>
# 3. Configuration (`config.yaml`)
The screenshot below is an overview of the `config.yaml` that allows user to tweak the model. Toffs team do not necessarily need to make changes to `config.yaml`. However, for PI team, editing the configuration is highly likely a need.

<img src="assets/images/8-configure-screenshot.png" width="700" align="middle"/>

<!-- There are 3 sections (split into color sections) which governs the following:  
1. Blue   : Parameters pertaining to the model itself.  
2. Yellow : Where the data folder and outputs directories for the Toffs' MVP is to be found.
3. Green  : Where the data folder, output folders, labelling, plotting options are found.    -->

Notes about `config.yaml`:    
Lines starting with # is a comment. There are no block comments symbols.  
The configuration file is read by functions in `src.evaluation.py`, `src.train.py`, `src.inference.py` as a Python nested dictionary (with some lists inside). Thus be careful with the use of spacing.
<!-- There are 6 sections in this configuration file, i.e. keys in dictionary:  
1. data
2. model
3. deployment_path
4. evaluation
5. plotting
6. others -->

Rows with colons (`:`) are keys in a dictionary.   
For instance "agentHashRange: 50" agentHashRange is key, 50 is value.

<pre>key:  
  subkey:   
    sub_subkey: value  
</pre>

<!-- Rows with a "-" are part of a Python List.   
For instance 'timestamp' and 'remote_addr' are entries in a list, and the key to access the list is 'base'  

<img src="assets/images/9-configdata.png" width="1200" align="middle"/>  
<img src="assets/images/10-configmodel.png" width="1200" align="middle"/>
<img src="assets/images/11-configeval.png" width="1200" align="middle"/>
<img src="assets/images/12-configdeployment.png" width="1200" align="middle"/>
<img src="assets/images/13-configplot.png" width="1200" align="middle"/>
<img src="assets/images/14-configothers.png" width="1200" align="middle"/> -->

## Deployment User Guide
Overview of the evaluation model.  
<img src="assets/images/15-walkthrough-pi.png" width="800" align="middle"/>
Envisaged use case: to facilitate the PI team in recreating our reported results, as well as help to batch train/infer future datasets.  

Overview of the toffs batch train/mini batch infer model.  
<img src="assets/images/16-walkthrough-toffs.png" width="800" align="middle"/>
Envisaged use case: to train the model on a specified subdomain, and conduct inference on attack period data.

### Detailed view of deployment environment  
The main processes, as well as inputs to and artefacts generated by the pipeline are shown in the following diagrams:  
<img src="assets/images/17-vm-evaluation.png" width="800" align="middle"/>
<img src="assets/images/18-vm-deployment.png" width="800" align="middle"/>  
A closer look at the inference artefacts:  
<img src="assets/images/19-artefacts-inference-1.png" width="800" align="middle"/>   
<img src="assets/images/20-artefacts-inference-2.png" width="800" align="middle"/>

### Code Documentation and Demo

Overview of functions called in the evaluation, train and inference pipelines:  
<img src="assets/images/21-sequence-diagram.png" width="1500" align="middle"/>

For more details, refer to the diagrams generated by the `PyCallGraph` library here:  
`src.evaluation.run_evaluation`
<img src="assets/images/22-pycallgraph-evaluation.png" width="1500" align="middle"/>
`src.train.train`
<img src="assets/images/23-pycallgraph-train.png" width="1500" align="middle"/>
`src.inference.inference`
<img src="assets/images/24-pycallgraph-inference.png" width="1500" align="middle"/>

Relationships between main classes used:
<img src="assets/images/25-UML.png" width="900" align="middle"/>

#### LSTM_v2 model architecture
The LSTM_v2 model uses the Keras functional API and instantiates model inputs and outputs based on 3 expected column types:
- Numeric
- Categorical (one-hot encoded)
- Categorical (label encoded)

This is specified as a dictionary in the `col_types` argument of the function.

The following diagram shows the way the inputs/outputs are structured for each of the 3 column types. This includes the dimensions of the initial input, whether or not it is followed by an embedding layer before concatenation, the activation function and loss function.

<img src="assets/images/30-LSTM_v2-architecture.png" width="900" align="middle"/>

For the categorical features, in order to generate inputs/outputs of the right dimensions, the argument `col_dims` specifying the dimensions of each column is required.

In `SequentialPipeline_v2`, `col_dims` is generated automatically using the `_cat_dim_attrs` dictionary, which contains column names as keys and the values are the attribute of the corresponding transformer to get the column dimension from. `_cat_dim_attrs` is also generated automatically during the parsing of the config file, based on a pre-defined dictionary. However, if the transformers used are new and not yet mapped, `_cat_dim_attrs` will have to be specified manually.

For more details on the modules/functions/classes, refer to the docstrings in the source code.

<a name="notebooks"></a>
# 4. Supplementary Notebooks
The Azure blob contains Jupyter notebooks showing:
- Demo notebooks for experimentation with the pipeline functions
  - `demo_pi.ipynb`, `demo_ps.ipynb`
- Findings from exploratory data analysis
- Data preparation

The notebooks can be located in the Azure Blob. Please note that not all the notebooks are runnable; some require the use of specialised environments (Databricks/PySpark) and the file link references could be wrong due to the fact that the notebooks are created in another environment (Polyaxon) where the folder structure is different. We will highlight the file structure at the start of the notebook.  

<a name="faq"></a>
# 5. FAQs

### What other essentials are needed for setting up a (Azure NC6v2) VM?
The newly provisioned VM needs some packages to be installed.
The graphics drivers can be added via the following. There is no need to download CUDA because Tensorflow enabled Docker images are used which has CUDA included.  
```bash
sudo apt install build-essential
sudo apt-get install linux-headers-$(uname -r)
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
```  
reboot the vm before installing the downloaded package (just select "install" from the default installation options):  
```bash
sudo reboot
sudo sh cuda_10.2.89_440.33.01_linux.run
```  
Install Docker:
```bash  
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
```  
Lastly,
```bash
 sudo usermod -aG docker XXX
```  
where XXX is the username used to login the VM. Logout and log back in for the settings to take effect.   
Install Nvidia-docker  
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

<!-- ### How to add a new dataset for evaluation?
Unless there is generated labels for the new dataset, there will be a need to add entries to the model medata.  
1. Generate parquet file.  
  The parquet files were generated in Databricks using Pyspark due to the challenges of dealing with large, unfiltered csv from client. A sample notebook will be added in the /notebooks folder.
2. Inside data/metadata folder, attack_timestamps.json. Add an entry to the json, in the format:  
  "filename.parquet" : [["start_time_of_attack"," end_time_of_attack"]],  
  For example:
  "WN_mobile.beike188.com_stage3.parquet": [["2019-07-27 14:45:00", "2019-07-27 16:13:00"]]
  For multiple attacks, refer to 1155jc.com entry in the attack_timestamps. -->

### Facing errors in trying to run bash scripts?
If you're running the script and encounter cryptic errors in the bash script such as this:  
```bash
PI_run_evaluation.sh: line 1: syntax error in conditional expression
PI_run_evaluation.sh: line 1: syntax error near `]]
PI_run_evaluation.sh: line 1: `if [[ "$(pwd)" == *threats-mitigation-base ]]
```
This is due to the replacement of unix LF with Windows style CRLF endings which will generate hard to interprete errors. This could happen if the repository is git cloned on Windows machine, then manually transferred to Ubuntu VM, or an incorrectly configured Git.
In vim editor, run the following command to see the line endings:  
```bash
:e ++ff=unix
```  
if there are CRLF endings such as these:  (the "^M" at the end of each line)  
<img src="assets/images/26-line-endings-debug.png" width="500" align="middle"/>
run the following command (install with <sudo apt install dos2unix> if the VM does not have the utility)
```bash
dos2unix pathto/script
```
which will automatically remove the CRLF endings.

### About the Azure Blob storage: (section copied from readme.txt in the Blob Storage)  
There are multiple blob containers which will be deleted at a later date. The container that stores the information to be handed over to PI team is the "transfer to PI" container.  
There are 4 folders in this Blob Container:   
1. `Correspondence with client` - contains Q&A with client (one time), email notification from client's staff about dataset uploads, as well as the information about the domains under attack, and the timestamp.  
2. `metadata` -	copy this folder to inside data folder of the repo. (refer to README.md quick start PI section)
3. `parquet files` - this folder contains all the parquet files, with only 1 specific subdomain inside, for evaluation purposes.
4. `raw data upload from client` - contains uploaded raw files from client, split into different groups, ie, KS, GS, IVI  
5. `notebooks` - due to size, we will be uploading the Jupyter notebooks into Azure blob instead.
During the uploads, the staff would email us the affected domains and it is up to us to extract the relevant domain from the upload (because the uploads include other unaffected domains)
For Jun dataset (AccessLog (June dataset), there was no email from client stating the attacked domains. Instead, they uploaded 2 `.zip` files (`Normal 16-06-2019.zip`, `attack_16-06-2019.zip`). From the attack.zip, we are able to infer that there are multiple domains attacked on 16th June, however, a plot of the traffic logs revealed that there were only 9 subdomains that were subject to volumetric attacks. These subdomains are prefixed with "Jun" inside `metadata/attack_timestamps.json`.
As such, we chose those 9 domains as the subdomains for the June dataset. We identified the period of attack from `attack.zip` for these domains, BUT, extracted the data from `normal.zip` (This is because the staff said that there were overlaps in the rows in attack and normal).
For Outwit dataset, there were no attack period given, but looking at the data, it was only a 2 hour period so we assumed that they only kept the attacked period. As such, we did not use this dataset at all.
For Toucai, the timestamps are written in a `.txt` in the folder.

<a name="future-directions"></a>
# 6. Possible Future Research Directions
Future research directions that we suggest the PI team to investigate if they want to continue using this model:

### Account for Seasonal Trends
Currently, the biggest data set we have in terms of duration of data coverage is the Jun dataset, and we only have at most 2 weeks of the data (Jun) datasets. There is insufficient data to conclusively determine if there is any seasonality trends (ie, extract seasonality based features) that could be used to help enhance the performance of the model. Currently, the features used in the model are request-level and user-level features.

### Integration with DDOS detection
Currently the scope of the project does not involve detecting the event of a DDOS attack itself. It was established early on with the PI that the detection of the attack event was not the priority objective and the delivered model will be "activated" upon 3th party detection (using tools like Greylog) of the event. Furthermore it is not recommended to employ the model during normal operations because of false positives (Likelihood of flagging some normal users as malicious).

### Feature engineering
- Exploring other data columns
- Using the `count_per_user` and `unique_per_user` functions in `src.feature_eng`
  - These showed potential in terms of distinguishing attackers and normal users (refer to `eda_feature_eng.ipynb`). E.g. number of unique http_user_agents in past 5 min (the graph title indicates 1 min, which is a typo)
  <img src="assets/images/31-feature-eng-http-user-agent.png" width="300" align="middle"/>
  - But being numeric features, they have to be scaled properly (probably between 0 and 1) to be used in the model. Otherwise they will overwhelm the other features and also cause the training loss to explode.
- Parsing string values in different ways (e.g. simplifying http_user_agent)
- Hyperparameter tuning
- Sequence length  
  - Because there are multiple uris to 1 page load, 10 uris might not reflect 1 page worth of uris. However, we don’t have visibility over the actual page structure as a working heuristic.

### Training period  
Only have limited labelled data (GS only) and it is generally observed that the more data fed to model, the performance improves. However, there might be a tipping point.

### Quantifying impact of model in production  
More clarification on the workflow of applying captchas to suspected malicious users is needed before we can simulate and better quantify the actual amount of traffic let through

### Integration Challenges  
Based on our discussion with Toffs on 18th Dec, due to latency between Nginx and the logging servers, some of the Nginx nodes might pass on their request logs late to the DB (potentially to 1 hr late). Different Nginx might serve the same user due to load balancing. Hence during inference, it is possible that requests from even the same user could arrive out of order in different csv batches to the model.   
<img src="assets/images/29-toffs-architecture-discussion.png" width="900" align="middle"/>  
