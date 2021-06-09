###
超参：expl_amount，horizon,model_lr,batch_length

expl_amount=0.05
horizon=15
model_lr=6e-4
batch_length=15
grad_clip=100
deter_size=200
没调出来

先调单环境的。
expl_amount=0.05
horizon=15
model_lr=1e-4
batch_length=20
grad_clip=10
deter_size=300

先调learning-rate

nohup python3 dreamer.py --seed 90 --logdir ./logdir/gridworld/dreamer/log_attn3 >log_attn3.txt &
#NOTE: sometimes infinite grad norm appears.
各尝试了2个seed，model_lr分别是2\3\4e-4

div:KL-divergence，表示model先验与后验的一致性

观察现象：
看了下图，发现model没学好，可能不是lr的问题
解决方案：
多加一些sample，给img_step的网络结构再复杂一些。

现象：
attn3曲线比较一致，先kill掉
model还是没学好，调一下weight吧（也就是KL-scale）
lr固定为保守一点的2e-4，把subnetwork调简单点，好学。

曲线：curve_kl
KL-scale调到5之后model似乎学得还不错.

再加大learning rate 跑两个seed试试。

曲线：curve_rw
imgloss开始回升，可能是div的系数太大了，rw系数可能也有点大.
model比较难训，不如加大一下训练的频率。

5.22
发现origin的std居然能到150左右，加一个tanh约束下

教训：
1. 不要质疑原论文。
2. 好好看论文。
3. 先从小模型开始。
4. 先调的参数：lr,各个loss的比例，模型大小。

5.25
目前实验结果总结：
在不加coin的情况下，泛化能力不错。
加了coin后model会变得难学，加了water后会更难学，体现在div会变大。
从当前策略来看没有学会吃coin.
注意到当前extra-step是stay，所以实际上只是根据desc学会了action。

下一步：
1. 用cpc_loss，先把单环境的water和coin学明白了。
2. 分别尝试固定coin和water的位置,但是让init state random.

5.25
很好，work了，甚至不固定位置也可以work，看来cpc loss确实有用。

下一步：
1. 调大model，现在多coin学得还不太好。
2. 完全不固定位置。
3. 可以尝试多个desc了.

5.26
dynamics的log太慢了，是没有flush的缘故？研究一下tf.summary.
6coin的reward被卡在10,策略是直接忽略coin去找goal，应该是exploration不够，policy net可能也有点简单.

分析6coin结果发现
reward loss较高，cpc_loss较高，认为是dynamic没学好而不是action net的问题。
增加exploration，增加action net的层数+1（因为先吃code的策略毕竟变难了）(2->4)。

同时训一个desc的baseline作为对比，验证model复杂度是可以训出来的。


5.26

目前结论：1. cpc loss好. 2.expl确实可以解决更大的问题。

解决了dynamics的log问题；发现exploration确实可以让cpc_loss和reward_loss下降地更好，而且会让agent顺路主动吃一些coin。
在进一步实验之前，首先找到模型合适的大小是重要的。（或者，与其说是合适的参数）
另外，能不能还是训一下reconstruction loss?这样可视化出来会更方便一些。
还有一件事，内存占用有点大(20G+)，我就先不管了。
还有一件事，batch_length和img_horizon要不要变？暂时不用。model学得准之后可以变长。

所以下一步想做的事情：
1. 在单环境的多coin和多water上先调出来，找到一个模型合适的规模（现在的exploration很可能还不够）。（能对desc泛化的模型一定比单一模型更大）
2. 考虑训reconstruction loss，当然可能要调一下比例。
3. 看一下dreamerv2的trick。

调参:（调跟RL相关的参数）
batch_length:最重要的是学和goal相关的信息（比如扔掉没有reward的trajectory），expl。要把短的trajectory保留下来。

把所有工作连成一个核心的故事。毕业一定要知道自己的时间节点。
先跟着师兄师姐发一篇论文，心里就踏实了。
相信自己的直觉，但是techique 的skill向大家学习。

合作，合作。

有俩实验没跑完。主要是在继续加大expl解决6coin问题，以及研究actionnet的影响。