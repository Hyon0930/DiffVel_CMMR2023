# DiffVel: Note-Level MIDI Velocity Estimation for Piano Performance by A Double Conditioned Diffusion Model



This DiffVel project is based on DiffRoll(https://github.com/sony/DiffRoll) and it's a modeification for note level MIDI velocity estimation. 
The main idea is to insert double conditioning audio and score; MIDI note frame inforamtion and predict the velocity of each note. 
This paper about this project is accepted by CMMR2023, Tokyo, Japan.
Any questions or comments, please contact to;
hyon.kim@upf.edu

The flow of the blocks of code is followiong; 
1. Train the model with MAESTRO dataset.
2. Inferece the model with SMD dataset and create pair of the model estimation and ground truth of note level MIDI velosity in npy files.
3. Based on the error profile created in the step2, export final results in csv file. 

To reporduce the results, please follow the instruction below.

## Enviroment set up 
Set up your python virtual enviroment and install the required packages. 

```
pip install -r requirements.txt
```

## Re-Training
Set path for training data, MAESTRO in config/spec_roll.yaml. 
Place your directory path containing MAESTRO v2.0.0. 

```
data_root: set/your/path/here
```

Training is covered by train_spec_roll.py to reproduce the DiffVel model. If you do not have the MAESTRO dataset, set download=True.  
``` 
python train_spec_roll.py gpus=1 model.args.kernel_size=9 model.args.spec_dropout=0 dataset=MAESTRO dataloader.train.num_workers=8 epochs=2000 download=False
```


## Use a trained Model for Inference for SMD dataset and Creating Error Profile npy files. 
The SMD is processed in h5 file for each excerpts, containing audio and score information (MIDI note frame roll).
Here is the trained model used in the paper (model folder); https://drive.google.com/drive/folders/1Eu96UOpwe8sdXP_ZWftTHp7KvoYvzAQM?usp=drive_link
SMD Testset used in the paper is here (test_data); https://drive.google.com/drive/folders/1Eu96UOpwe8sdXP_ZWftTHp7KvoYvzAQM?usp=drive_link

Set path for SMD dataset in config/inference.yaml.

```
Place your trained model ckpt file at "comp_modelckpt" 
Set your path to output inference result at "inference_result_path"
Set the SMD dataset in h5 format at "SMD_data_root"
Set the path to output error profile from the inference at "pair_pkl_dir:" 
```

Run following command to inference the model and create error profile npy files which contains the estimation from model and ground truth from SMD dataset. 

```
python run_test.py
```


## Post processing for removing remained gaussian noise. 

In order to perform the post processing for removing the remained gaussian noise after the inference, please follow the instruction below.
Here is the estimation results and ground truth pairs in npy file used in the paper: https://drive.google.com/drive/folders/1Eu96UOpwe8sdXP_ZWftTHp7KvoYvzAQM?usp=drive_link


```
python post_processing_gnoise.py --output_dir "path/to/your/output/dir/for/results" --pair_pkl_dir "path/to/your/pair/pkl/dir" 

```
This script outputs final estimation results of the model in csv format for each excerpts focusing on MIDI velocity estimation. 


## License: MIT


## Citation

```
@inproceedings{HyonDiffvel2023,
   title={DiffVel: Note-Level MIDI Velocity Estimation for Piano Performance by A Double Conditioned Diffusion Model},
   author={Kim, Hyon; Serra, Xavier.},
   booktitle={Proceedings of the 16th International Symposium on Computer Music Multidisciplinary Research},
   pages={},
   year={2023},
   address={Tokyo, Japan},
   url={http://hdl.handle.net/10230/57790}
}

```

