# Deep Learning
## Project: Image Classifier
In this project, I'll train an image classifier to recognize different species of flowers. I have built a deep neural network with a Pytorch application that can train an image classifier on a dataset, then predict new images using the trained model. 
### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- Pytorch
- Torchvision (for pretrained models)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)
*Note:* You need to have GPU available on your system for training section.

### Code
**Notebook:**
The code is provided in the `image_classifier_project.ipynb` notebook file. You will also be required to use the included `cat_to_name.json` which is a mapping from category label to category name and `workspace-utils.py` files. The data of 102 flower categories can be downloaded [here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz).

**Python Script:**
The code is provided in in two python files. The `train.py` will train a network on dataset and save the model as checkpoint. The `predict.py` uses a trained network to predict the class for an input image. You will also be required to use the included `cat_to_name.json` and `workspace-utils.py` files and the same data mentioned above.
### Run
**Notebook:**
In a terminal or command window, navigate working directory to `image_classifier_project.ipynb` and run one of the following commands:

```bash
ipython notebook image_classifier_project.ipynb
```  
or
```bash
jupyter notebook image_classifier_project.ipynb
```
This will open the iPython Notebook software and project file in your browser.

**Python Script:**

You should note that train file will ask you to imput some information like directory to save checkpoint, pretrained architecture, hyperparameters (learning rate, hiddenunits, epochs), etc through command line. Therefore, it is better to run them with -h or --help to get familliar with arguments and input variables. In a terminal or command window, navigate working directory to `train.py` and run one of the following commands. 

```bash
python train.py -h
```  
then
```bash
python predict.py -h
```




