# Mi-CGA

## Run example
```bash
python main.py --numLabel 4 --output test.txt --E 100 --lr 0.003 --weight_decay 0.00001 --seed 1001 --crossModal --usingGAT --missing 66 --numTest 5 --wFP --rho 0.1 --reconstructionLoss kl
```

## Dataset 
[IEMOCAP](link)

## Files

> `data.py`: prepare train/test dataset
>
> `utils.py`: some handy functions for model training etc.
>