# Pseudo--efficientNet
Pytorch--使用伪标签训练efficientNet模型
# 来源
针对于伪标签的讲解，国内有简书与CSDN博客介绍，分别如下：
（https://www.jianshu.com/p/fd4cae0d0e85）
（https://blog.csdn.net/leadai/article/details/81518830）.
但是对于源码，一直搜索不到，所以就手撸了一个伪标签训练的模型，开源出来。
代码若有问题，请联系我（email:xjy333@mail.nwpu.edu.cn）
# 使用前注意
1、如果您的pandas,torch版本过低，可能会报错，需要针对于某个函数，找对应版本针对性改动。  
2、对于pseudo 的数据集图片，不能有.png结尾，否则会导致程序报错，因为PIL不是RGB三通道。  
3、该工程与上述简书、博客区别在于未将val的数据化为pseudo，所以会出现三种类型的数据集。因为在该项目运行时，val数据集保证了百分百的正确。  
4、针对于3的问题，若想做到如简书、博客上的流程，只需把val的部分加到pseudo即可。  
# 用法
*由于需要实时变更训练集，故选择重写Dataloader，根据每次训练结果重组并筛选DataFrame进行加载训练。  
*数据集目录层次需要如下：  
    """
    ├─train  
    │  ├─0  
    │  ├─1    
    ...    
    ├─val  
    │  ├─0  
    │  ├─1    
    ...  
    ├─pseudo  
    │  ├─0  
    │  ├─1    
  
    """ 其中0，1...为分类的目录，用于存放各个分类的图片  
* path_t_base，path_v_base，pseudo_path分别对应于train、val、pseudo的目录  
* 其中pseduo的目录，可以选择不进行分类  
* 该目录的层次，保证了源码的运行。但若读者自行构建三个DataFrame亦可正常跑通  
* DataFrame的列分别为img_loc，type，label，p_label  
# Tips  
若该工程对您有用，麻烦给个小星星。  

