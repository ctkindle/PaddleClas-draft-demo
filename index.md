```python
# !unzip -qa paddlepaddle-PaddleClas-dygraph.zip
```

# 一、写作项目的一些基本设定
## 1.项目的目标是帮助阅读者深入使用pcl框架
我觉得用户对paddle套件的使用能分为三个层次：1）通过yaml使用框架。2）能够在框架上添加用户代码，定制使用框架。3）非常熟悉框架能贡献代码、改进框架。本次项目的目的是帮助读者达到第二甚至第三层的应用框架能力，而非只是通过yaml定制使用。通过yaml使用的教程[这里的文档](https://paddleclas.readthedocs.io/zh_CN/latest/index.html)已经写得很清楚了～～,原理解释得也很清楚了。所以本项目从以下方便着手：
整体介绍pcl文件布局，包括文件夹、文件功能。
介绍主要文件，比如reader、train、loss、optimizer文件里各个函数的功能及调用关系。
从train文件开始，以yaml文件设置为基础，介绍参数加载、模型训练过程。
## 2.通过一个项目例程来展示pcl的使用和修改
主要展示如何根据实际需求进行数据处理、模型修改。例子不一定加入最先进的论文模型，主要是展示如何将新的数据处理代码、新模型加入到框架中使用。这两天正好有大佬同学在ai studio上发了个脊椎网络spinalnet，本教程就演示将这个网络加入pcl使用吧。
## 3.项目使用ipynb格式
可以在ai studio上编写、执行，方便导出为markdown格式发布到其他地方。现在ai studio的官方环境是1.84。自己能装2.0beta。可能正式版还会有所区别（比如开头的那个动态图的开关设置）。
## 4.总结
#我理解这个项目***前半部分是带着读者读代码，理清脉络，后半部分是通过一个项目展示如何加代码、改代码。***
# 二、项目流程流程
## 1.数据处理
包括规范数据格式、数据标准化、数据增强
## 2.模型选择、修改
介绍包括理清pcl模型库的代码流程，以及如何修改已有模型的结构、创建新的更先进的模型加入模型库。
## 3.模型训练、调参
调参部分、模型蒸馏部分我觉得框架做得已经很好了。也可以介绍下用paddle怎么实现、修改，但项目的重点还是介绍pcl的使用。毕竟模型蒸馏是pcl将交叉商损失，魔该成gan的JS散度损失的独门SSLD秘技，实在改不动了～～
## 4.预测部署
模型的动静态图转换，服务器上、本地pc上的部署。这部分官方文档介绍得也比较全面了。我觉得重点还是数据处理和模型选择修改部分。
# 三、需要考虑到的问题
* 新版动态图基础上的pcl函数结构应该和老版本大致一致，还有就是现在只能用2.0beta版，正式项目是不是得用2.0正式版本？
* 写项目中遇到的一些问题（坑与经验）要不要写入项目，还是以教程为主。比如我在数据处理时层遇到一个问题：我用地址拼接的方法给reader提供图片存储地址，结果发现pcl的reader读不出数据（我自己写的reader可以读出）。在读了pcl的reader源码后发现，pcl的reader是通过穷举目录下文件的方式读取数据，所以图片文件夹下不能有子文件夹。
* 类、函数的调用往往是树形的，用顺序的文本表达，有没有什么比较流畅的好方法？

---------------------【以上是项目介绍，下面是项目实现。其中第四部分是解读项目文件，第五部分是例程。】---------------------

# 四、pcl整体项目解读
## 1.项目结构（这里介绍那些文件夹、文件及文件里的类和函数需要权衡，都写感觉有点工程太大了～）
解压后，PaddleClas文件夹下有三个文件，五个文件夹：

![](https://ai-studio-static-online.cdn.bcebos.com/620e5be7a3dc449eb7383561a1954f8894dff8ed6a4a4ea987e4e34cce92dc68)

* requirements.txt文件用于使用pip安装PaddleClas的依赖项。
* tools文件夹下存放用于模型下载、训练、预测和部署的脚本。
* dataset文件夹下存放处理训练数据的脚本。
* docs文件夹下存放中英文文档。
* ppcls文件夹下存放PaddleClas框架主体。
* configs文件夹下存放训练脚本和验证脚本的yaml配置文件。

其中configs、tools、ppcls三个文件夹最为关键，我们分别介绍：

### 1）configs文件夹

configs文件夹下按模型类别分别存放了yaml初始化文件，用于设置超参。框架为各种模型各个型号都预设了适合的超参，真是贴心。当然，使用时还是要根据自己项目实际情况对超参进行调整。比如归一化训练集数据的超参mean、std就需要根据自己的数据集重新计算。yaml文件结构如下：（下面是一个从ai studio项目上摘过来的例子，实际项目中会根据后面介绍的例程重写）
```
mode: 'train' # 当前所处的模式，支持训练与评估模式
ARCHITECTURE:
    name: 'ResNet50_vd' # 模型结构，可以通过这个这个名称，使用模型库中其他支持的模型
pretrained_model: "" # 预训练模型，因为这个配置文件演示的是不加载预训练模型进行训练，因此配置为空。
model_save_dir: "./output/" # 模型保存的路径
classes_num: 102 # 类别数目，需要根据数据集中包含的类别数目来进行设置
total_images: 1020 # 训练集的图像数量，用于设置学习率变换策略等。
save_interval: 1 # 保存的间隔，每隔多少个epoch保存一次模型
validate: True # 是否进行验证，如果为True，则配置文件中需要包含VALID字段
valid_interval: 1 # 每隔多少个epoch进行验证
epochs: 20 # 训练的总得的epoch数量
topk: 5  # 除了top1 acc之外，还输出topk的准确率，注意该值不能大于classes_num
image_shape: [3, 224, 224] # 图像形状信息


LEARNING_RATE: # 学习率变换策略，目前支持Linear/Cosine/Piecewise/CosineWarmup
    function: 'Cosine'
    params:
        lr: 0.0125

OPTIMIZER: # 优化器设置
    function: 'Momentum'
    params:
        momentum: 0.9
    regularizer:
        function: 'L2'
        factor: 0.00001

TRAIN: # 训练配置
    batch_size: 32 # 训练的batch size
    num_workers: 4 # 每个trainer(1块GPU上可以视为1个trainer)的进程数量
    file_list: "./dataset/flowers102/train_list.txt" # 训练集标签文件，每一行由"image_name label"组成
    data_dir: "./dataset/flowers102/" # 训练集的图像数据路径
    shuffle_seed: 0 # 数据打散的种子
    transforms: # 训练图像的数据预处理
        - DecodeImage: # 解码
            to_rgb: True
            to_np: False
            channel_first: False
        - RandCropImage: # 随机裁剪
            size: 224
        - RandFlipImage: # 随机水平翻转
            flip_code: 1
        - NormalizeImage: # 归一化
            scale: 1./255.
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - ToCHWImage: # 通道转换

VALID: # 验证配置，validate为True时有效
    batch_size: 20 # 验证集batch size
    num_workers: 4  # 每个trainer(1块GPU上可以视为1个trainer)的进程数量
    file_list: "./dataset/flowers102/val_list.txt" # 验证集标签文件，每一行由"image_name label"组成
    data_dir: "./dataset/flowers102/" # 验证集的图像数据路径
    shuffle_seed: 0 # 数据打散的种子
    transforms:
        - DecodeImage:
            to_rgb: True
            to_np: False
            channel_first: False
        - ResizeImage:
            resize_short: 256
        - CropImage:
            size: 224
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - ToCHWImage:
```

### 2）tools文件夹

tools文件夹下主要子文件夹和文件：

![](https://ai-studio-static-online.cdn.bcebos.com/e8fa8f7388154fd3b923aee704d3faf3033ff4c3298c4b8fb2fc9259e47e86ba)

* download.py是下载预训练模型的脚本。
* train.py和eval.py是训练和验证模型的脚本。
* ema.py和ema_clean.py是计算和清除指数移动均值的脚本，用于提升训练效果。
* export_model.py和export_serving_model.py是导出训练模型和服务部署模型的脚本。
* infer文件夹下放的是用于预测、推理的脚本。
* program.py是组装训练过程的脚本。train过程由此脚本组装好后运行。
* ......
 
### 3）ppcls文件夹

ppcls文件夹是框架的主体，负责运行模型训练过程：

![](https://ai-studio-static-online.cdn.bcebos.com/53e8053d45cf4c529981ea0651adde6e5e2656e72067473aaf29246657539105)

* data文件夹存下放处理数据的脚本，包括模型读取数据的reader和数据增广处理脚本。
* modeling文件夹下存放模型结构及loss计算脚本。
* optimizer文件夹下存放优化器超参（包括优化器种类，正则化项设置）和学习率设置脚本（包括学习率warmup和各种decay策略）。
* utils文件夹下存放其他功能脚本（包括脚本参数解析、参数检查、模型存取、日志记录脚本）

（这部分大致介绍框架文件结构，下面流程部分再展开介绍各个用到的关键类和函数）

## 2.项目流程介绍

### 1）数据处理流程

训练流程开始，我们通过执行 python -m paddle.distributed.launch 调用tools 文件夹下的 train.py 脚本（预测流程调用eval。py脚本时的数据处理流程是大致相同的，只是不进行shuffle和数据增强），这个过程首先需要将训练集数据经过处理后送入模型。train.py 脚本中是这样使用数据读取脚本reader.py的：

```
...
from ppcls.data import Reader # 导入Reader类
...
train_dataloader = Reader(config, 'train', places=place)() # 声明 train_dataloader 对象，config为脚本参数，‘train’是数据读取模式（train模型进行数据增广和shuffle）， places指定执行资源是cpu还是gpu
...
program.run(train_dataloader, config, net, optimizer, lr_scheduler, epoch_id, 'train') # 训练过程被封装在了program脚本的run函数里
...
```

在run函数中则用下面的语句调用train_dataloader对象：
```
...
for idx, batch in enumerate(dataloader()): # dataloader 是 train.py 脚本中调用 program.run()时传过来的train_dataloader 对象
...
```
接下来我们看看Reader类如何实现分布式分batch读取数据。先看Reader类源码：

```
class Reader:
    """
    Create a reader for trainning/validate/test

    Args:
        config(dict): arguments
        mode(str): train or val or test
        seed(int): random seed used to generate same sequence in each trainer

    Returns:
        the specific reader
    """

    def __init__(self, config, mode='train', places=None):
        try:
            self.params = config[mode.upper()]
        except KeyError:
            raise ModeException(mode=mode)

        use_mix = config.get('use_mix')
        self.params['mode'] = mode
        self.shuffle = mode == "train"

        self.collate_fn = None
        self.batch_ops = []
        if use_mix and mode == "train":
            self.batch_ops = create_operators(self.params['mix'])
            self.collate_fn = self.mix_collate_fn

        self.places = places

    def mix_collate_fn(self, batch):
        batch = transform(batch, self.batch_ops)
        # batch each field
        slots = []
        for items in batch:
            for i, item in enumerate(items):
                if len(slots) < len(items):
                    slots.append([item])
                else:
                    slots[i].append(item)

        return [np.stack(slot, axis=0) for slot in slots]

    def __call__(self):
        batch_size = int(self.params['batch_size']) // trainers_num

        dataset = CommonDataset(self.params)

        if self.params['mode'] == "train":
            batch_sampler = DistributedBatchSampler(
                dataset,
                batch_size=batch_size,
                shuffle=self.shuffle,
                drop_last=True)
            loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=self.collate_fn,
                places=self.places,
                return_list=True,
                num_workers=self.params["num_workers"])
        else:
            loader = DataLoader(
                dataset,
                places=self.places,
                batch_size=batch_size,
                drop_last=False,
                return_list=True,
                shuffle=False,
                num_workers=self.params["num_workers"])

        return loader
```

Reader类源码解析：

初始化函数__init__()初始化了Reader对象的参数，包括param（训练参数集合）、use_mix（是否使用mixup数据增广）、shuffle（是否打乱数据顺序）。如果参数的设置是在train（训练）模式下，采用的mixup数据增广方法，则还需指定collate_fn（整理函数）和batch_ops（针对整个batch的操作）。因为其他数据增广方法如变形、切割、遮挡、噪声等是针对单个数据进行，而像mix up、sample pair这样的增强方式需要对整个batch的数据进行操作，所以要对整个batch的数据进行处理。

python中类似于重载括号的__call__（）方法可以是类对象像函数一样被使用，所以Reader的__call__（）函数的作用就是输出一个batch的数据。首先将通过继承DataSet类创建的CommonDataset()类来创建数据集对象dataset，然后用DataLoader类创建loader对象并将其返回。如果在训练模型下loader对象将由DistributedBatchSampler类创建的batch_sampler对象创建，用于分布式载入数据。如果是非训练模式，则直接用dataset对象创建普通载入器。

......

### 2）训练流程

......

### 3）预测流程

......

# 五、数据处理部分的项目思路、样例

## 1.数据定制处理（包括添加自定义数据增广处理）
*引入imgaug库的数据增广方法到pcl中使用*
......

## 2.模型处理（修改或使用自定义模型）
*加入脊椎网络*
......

(以上是个项目的大体思路、骨架。)
