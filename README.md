
# Image Super-resolution via Dual-state Recurrent Neural Networks (CVPR 2018)
### [[Paper Link]](https://arxiv.org/pdf/1805.02704.pdf)

### Citation

	@inproceedings{han2018image,  
		title={Image super-resolution via dual-state recurrent networks},
		author={Han, Wei and Chang, Shiyu and Liu, Ding and Yu, Mo and Witbrock, Michael and Huang, Thomas S},
		booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
		year={2018}
	}
  
### Dependencies
- Common python dependencies can be installed via `pip install -r requirements.txt`
- Lingvo (for inference only), see linvgo project page for installation instructions.

### Data
There is a very helpful repo [collected](https://github.com/jbhuang0604/SelfExSR#datasets) download links for all the training and test sets needed here. 
### Training 
The training data is specified by a file list of HR images. No futher pre-processing is needed as we perform downsampling and augmentation on-the-fly.

Use `train.py` and the model specification file `model_recurrent_s2_u128_avg_t7.py` to start a training job. 
### Inference
We release our models in tensorflow [lingvo](https://github.com/tensorflow/lingvo) format such that the models are self contained for inference tasks. Each model consists of by a `inference_graph.pbtxt` and a checkpoint file. 

To run the inference with provided pre-trained models on an image, use provided `predictor.py`:
Example:

	`python predictor.py --checkpoint=models/x3/ckpt-00754300 --inference_graph=models/x3/inference.pbtxt --image_path=./cat.png --output_dir=./`


The script will write super-resolved images to `output_dir`.
### Evaluation
Use `evaluate.py` to compute average PSNR on a test set after saving all the model predicted images. Eval set is also specified by a file list.
Example:

	`python evaluate.py --hr_flist=flists/set5.list --prediction_dir=${your_pred_dir}`

### Acknowledgement
This code is partly based on a previous work from our group [[here]](https://github.com/ychfan/sr_ntire2017)


