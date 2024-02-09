# Mi-CGA
This code is about a framework, Mi-CGA, handle incomplete multimodal learning in conversational contexts. It consists of two main components including Incomplete Multimodal Representation (IMR) and Cross-modal Graph Attention Network (CGA-Net). 

## Paper


## Run Mi-CGA

### Prerequisites
- Python 3.8
- CUDA 10.2
- pytorch ==1.8.0
- torchvision == 0.9.0
- torch_geometric == 2.0.1
- fairseq == 0.10.1
- transformers==4.5.1
- pandas == 1.2.5
(all used package in our environment is list in env.txt)

### Run example
```bash
python main.py --numLabel 4 --output test.txt --E 100 --lr 0.003 --weight_decay 0.00001 --seed 1001 --crossModal --usingGAT --missing 66 --numTest 1 --wFP --rho 0.1 --reconstructionLoss kl
```



## Dataset 
[IEMOCAP](https://drive.google.com/drive/u/1/folders/1o4fvksJfIfUTsbe37izf3bWDS-morOZt)

## Pretrained model

## Files

> `attentionModule.py`: contain cross Attention module
>
> `utils.py`: handy functions.
>
> `dataloader.py`: build graph, generate missing masks and preprocess data into suitable iterator for training/testing.
>
> `main.py`: main function to run model.
>
