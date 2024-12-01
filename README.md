# Samsara GAN - README

## Prerequisites
+ Python 3.5.2
+ TensorFlow 1.1.0
+ Ubuntu 16.04 LTS

## Getting Started
### Clone Project
<pre><code>git clone https://github.com/hdnse4798/Test</code></pre>

### Data Preparation
+ Download Dataset
<pre><code>https://goo.gl/QDxAAi</code></pre>

### Directory structure
+ Move dataset into <code>data</code> folder (e.g. data/horse2zebra).
<pre><code>
├── data/                    # Dataset
│   ├── apple2orange
│   ├── horse2zebra
│   └── ...
├── main/              # Main Project
│   ├── densenet.py
│   ├── discriminator.py
│   ├── generator.py
│   ├── load_train_data.py
│   ├── model.py            # Create Model(Network)
│   ├── ops.py              # Model operations
│   ├── test_model.py       # Testing Entry Point
│   ├── train.py            # Training Entry Point
│   └── utils.py            # Utility Functions
└── README.md
</code></pre>


## Train
<pre><code>python train.py</code></pre>

+ If you want to change some default settings, you can type these to the command line:
<pre><code>python train.py --dataset_dir="apple2orange"</code></pre>

+ Check TensorBoard to see training progress and generated images.
<pre><code>tensorboard --logdir=./checkpoints/{folder name}</code></pre>

+ If you halt the training process and want to continue training, then you can set the <code>load_model</code> parameter:
<pre><code>python train.py --dataset_dir="dataset" --load_model="{folder name}"</code></pre>

+ The list of arguments:
<pre><code>usage: train.py [-h] [--batch_size BATCH_SIZE] [--image_size IMAGE_SIZE]
                [--norm NORM] [--lambda1 LAMBDA1] [--lambda2 LAMBDA2]
                [--learning_rate LEARNING_RATE] [--beta1 BETA1]
                [--pool_size POOL_SIZE] [--ngf NGF] [--load_model LOAD_MODEL]
                [--dataset_dir DATASET_DIR] [--class_num CLASS_NUM]
                [--iteration ITERATION] [--initial_step INITIAL_STEP]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        batch size, default: 1
  --image_size IMAGE_SIZE
                        image size, default: 256
  --norm NORM           [instance, batch] use instance norm or batch norm,
                        default: instance
  --lambda1 LAMBDA1     weight for forward cycle loss (X->Y->X), default: 10.0
  --lambda2 LAMBDA2     weight for backward cycle loss (Y->X->Y), default:
                        10.0
  --learning_rate LEARNING_RATE
                        initial learning rate for Adam, default: 0.0002
  --beta1 BETA1         momentum term of Adam, default: 0.5
  --pool_size POOL_SIZE
                        size of image buffer that stores previously generated
                        images, default: 50
  --ngf NGF             number of gen filters in first conv layer, default: 32
  --load_model LOAD_MODEL
                        folder of saved model that you wish to continue
                        training (e.g.
                        horse2zebra_batchsize=1_LR=0.0002_20180206-1441),
                        default: None
  --dataset_dir DATASET_DIR
                        path of the dataset, default: horse2zebra
  --class_num CLASS_NUM
                        number of classes, default: 2
  --iteration ITERATION
                        number of iteration, default: 200000
  --initial_step INITIAL_STEP
                        number of initial step, default: 500</code></pre>

### Note
Due to cross platform, the results may have minor difference.

## Acknowledgments
+ Code borrows from [CycleGAN-TensorFlow](https://github.com/vanhuyz/CycleGAN-TensorFlow) and [densenet-tensorflow](https://github.com/YixuanLi/densenet-tensorflow).
+ Datasets are collected from [CycleGAN](https://github.com/junyanz/CycleGAN) and [MSCOCO 2017](http://cocodataset.org/#home).
