#LAPGAN-improved

## 系统要求
* Mac OS X or Linux
* NVIDIA GPU with compute capability of 3.5 or above.

## 安装torch深度学习框架和依赖包
* Install [Torch](http://torch.ch)
* Install the nngraph and tds packages:

```
luarocks install tds
luarocks install nngraph
```
## 参数选择 option
```
  --save8           (default "logs8")      subdirectory to save logs
  --save16           (default "logs16")      subdirectory to save logs
  --save32           (default "logs32")      subdirectory to save logs
  --saveFreq         (default 30)           save every saveFreq epochs
  --network8         (default "")          reload pretrained network of scale 8
  --network16        (default "")          reload pretrained network of scale 16
  --network32        (default "")          reload pretrained network of scale 32
  -p,--plot                                plot while training
  -r,--learningRate  (default 0.02)        learning rate
  -b,--batchSize     (default 128)         batch size
  -m,--momentum      (default 0)           momentum
  --coefL1           (default 0)           L1 penalty on the weights
  --coefL2           (default 0)           L2 penalty on the weights
  -t,--threads       (default 4)           number of threads
  -g,--gpu           (default -1)          gpu to run on (default cpu)
  --K                (default 1)           number of iterations to optimize D for
  -w, --window       (default 3)           windsow id of sample image
  --hidden_G8        (default 64)         number of channels in hidden layers of G in 8-scale
  --hidden_D8        (default 64)         number of channels in hidden layers of D in 8-scale
  --hidden_G16       (default 64)         number of channels in hidden layers of G in 16-scale
  --hidden_D16       (default 64)         number of channels in hidden layers of D in 16-scale
  --hidden_G32       (default 64)         number of channels in hidden layers of G in 32-scale
  --hidden_D32       (default 64)         number of channels in hidden layers of D in 32-scale
  --ifwbupsampleD1     (default 0)
  --ifwbupsampleG1     (default 0)
  --ifwbupsampleD2     (default 0)
  --ifwbupsampleG2     (default 0)
  --mode              (default 1)
  --seed              (default 1)
 ```
## 运行命令
### 训练过程 train process

```
 th scripts/train_cifar_coarse_to_fine_classcond-8-16-32-fir.lua
```
* 如果需要设置不同参数，则可以在命令里设置参数 比如：
```
th scripts/train_cifar_coarse_to_fine_classcond-8-16-32-fir.lua -mode 16 -save16 un_sc16_s1_k2 -K 2 -g 1 

```
注： fir 和 sec 两个版本主要是8*8层的结构不同，其他均相同：

### 图片重构过程 reconsitution process
```
default cmd : th sampling/main.lua

```