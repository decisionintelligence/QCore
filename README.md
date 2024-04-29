# QCore

QCore: Data-Efficient, On-Device Continual Calibration for Quantized Models (under review)

How to run the model:
 * Train the full-precision model and generate QCore. Run [main.py](main.py) by specifying the datasets for training and streaming. For example:
 `python main.py --bits 4 --core_size 20 --data_source activities --dataset 1 --stream_dataset 2`
 * Baselines can be found in the `/baselines/` folder. To run them, execute [main.py](baselines/utils/main.py) while specifying the appropriate model and the original and target domains. For instance, after accessing the folder using `cd baselines`, a possible command would be:
 `python ./utils/main.py --bits 4 --dataset har --lr 0.01 --buffer_size 20 --data_source activities --model er --data_in 1 --stream_dataset 2`
 * The data is managed in the [dataloader.py](utils/dataloader.py) file. 
 * Time-series data can be found in the `/data/` folder. The results will be inserted into a local database. 
 * Detailed information about all the parameters can be found in each execution file.

 # Citation

If you use the code, please cite the following paper:

<pre>  
@article{pvldb/Ca24,
  author    = {David Campos and Bin Yang and Tung Kieu and Miao Zhang and Chenjuan Guo and Christian S. Jensen},
  title     = {{QCore: Data-Efficient, On-Device Continual Calibration for Quantized Models}},
  journal   = {{PVLDB}},
  volume    = {17},
  year      = {2024}
}
</pre> 
