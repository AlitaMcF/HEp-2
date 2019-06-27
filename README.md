HEp-2 Cell Classification Project
====

`ResNet50`  `Transfer Learning`  `PyTorch`

* The neural network for classifying the HEp-2 cells.<br>
* This is a 50 levels ResNet with Transfer Learning.
* Here are the dataset urls:
    * dataset of HEp-2 2016: [hep2016.zip](https://pan.baidu.com/s/1iP7ZS79ICae1miu_pbWTVA "https://pan.baidu.com/s/1iP7ZS79ICae1miu_pbWTVA")
      * 提取码：ms3z
    * dataset of HEp-2 2012: [hep2012.zip](https://pan.baidu.com/s/1WTHMiEKsMdpMZSmjAvCk-g "https://pan.baidu.com/s/1WTHMiEKsMdpMZSmjAvCk-g")
      * 提取码：2brd
    
## **Description**
_Old Code是前期迭代版本，建议以New Code为主_
* **New Code**
   * **[ResNet50_2012.py](/new_code/ResNet50_2012.py)**: 训练HEp2012彩色图像的网络
   * **[ResNet50_2016.py](/new_code/ResNet50_2016.py)**: 训练HEp2016灰色图像的网络
   * **[Pre_Train.py](/new_code/Pre_Train.py)**: 预训练
   * **[Nor_Train.py](/new_code/Nor_Train.py)** 预训练之后的迁移训练
   * **[MaskImg.py](/new_code/MaskImg.py)**: 依据遮罩切割图像感兴趣区域
   * **[MakeNewDataset.py](/new_code/MakeNewDataset.py)**: 生成预训练数据集，本项目中并非HEp2012数据，按需自行更改代码即可
   * **[GenerateTest.py](/new_code/GenerateTest.py)**: 生成测试数据集
   * **[ExpandTrainingSet.py](/new_code/ExpandTrainingSet.py)**: 数据扩容

* **Old Code**
   * **[Cuda_ResNet50_HEp-2_2016.py](/old_code/Cuda_ResNet50_HEp-2_2016.py)**：可利用GPU加速训练过程
   * **[ResNet50_HEp-2_2016.py](/old_code/ResNet50_HEp-2_2016.py)**：只能用CPU运行
   * **[ResNet50_Failed.py](/old_code/ResNet50_Failed.py)**：失败的网络，暂且留着
   * **[GenerateTest.py](/old_code/GenerateTest.py)**：按20%比例随机生成测试集
   * **[MaskImg.py](/old_code/MaskImg.py)**：切割图像
