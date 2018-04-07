# BIO465LeafPrediction
Code necessary to predict plant-disease pair based on image of leaf.

Image(s) to be predicted need to be placed inside the ```images```
folder. Also the ```classify.py``` script needs to be modified to point the script to the new image as of right now.


To make replication of this experiment easier, we have used a VirtualBox environment. In order to use this environment you will need to install
the appropriate VirtualBox version for the host system.  These can be found here [VirtualBox.org](https://www.virtualbox.org/ "VirtualBox.org Website")

The environment with the software necessary to run the code in this repository can be found [here](http://edwardportela.com/bio465/Ubuntu_17.10-VB-64bit.7z "VirtualBox Environment")

The username and password for this environment are both osboxes.org


This code has been tested using python3 so it is recommended that it be used.
Example usage:
```python3 classify.py```


Code for ```deploy.prototxt``` was copied from the following [repository](https://github.com/salathegroup/plantvillage_deeplearning_paper_analysis "PlantVillage Deep Learning Analysis").
It was copied from ```googLeNet/color-80-20/finetune```

## Downloading caffe models and snapshots

Due to the size of the generated models and snapshots, they have not been included with this repository. However a script to download them has.
In order to get this information run the following commands.
```bash
bash get_snapshots.sh
```
Once this is done you can run the example with the following command.
```bash
python3 classify.py
```
