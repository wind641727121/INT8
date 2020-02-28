# INT8

This project is based on [NiuTensor](https://github.com/NiuTrans/NiuTensor). Mainly focused on using integer to speed up the Feedforward Neural Network Language Mode. 
---

## Catalog
Catalog | Content
---|---
/brown |                     Data set
/source |                    Source code of FNNLM
/source/INT8.cpp(INT8.cpp.h) |	                   MY work 
fnn.model |	                    Train model
fnn.prob |	                    Test result
---



使用教程(英文储备不够了)
---
* 安装[NiuTensor](https://github.com/NiuTrans/NiuTensor)
* 将`INT8.cpp`和`INT8.h`添加到`/source`目录下 
* 在`source\tensor\XTensor.h`里添加 `#include "../INT8.h"`

## 函数
函数名 | 作用
---|---
`void MySum(const XTensor &a, const XTensor &b, XTensor &c)` |      补充X_INT型Tensor的加法
`void Mymul(const XTensor &a, const XTensor &b, XTensor &c,int corenum=1)`|      补充X_INT型Tensor的乘法
`void Mymul_c(const XTensor &a, const XTensor &b, XTensor &c, int corenum = 1)` |	    补充X_INT8型Tensor的乘法
`void MYMatrixMul(const XTensor &a, const XTensor &b, XTensor &c,int corenum=1)` | a,b为FLOAT型Tenosr 在该函数中完成对浮点的映射,整型和比例矩阵的乘法,corenum表示线程大小(下同)
`void Show(const XTensor &x, const XTensor &s, int flag = 0)` |	通过整型和比例矩阵打印原矩阵的值 flag=0 按行映射 flag =1 按列映射
`void Mapping(const XTensor &a, XTensor &x, XTensor &s) `| 将a映射为x和s x和s的大小与a相同
`void UnMapping(XTensor &a, const  XTensor &x, const  XTensor &s)`| 用x和s反映射为a x和s的大小与a相同
`XTensor Match(XTensor &x, XTensor &s, int flag)`| 压缩s为乘法做准备 返回的一维向量
`void Match(XTensor &x1, XTensor &s1, XTensor &x2, XTensor &s2)`|调整s1和s2为加法做准备
`void  Mymul_int8(const XTensor &a, const XTensor &b, XTensor &c,const XTensor &sa, const XTensor &sb, XTensor &sc, int corenum)`|x与s的乘法
`void Myadd_int8(const XTensor &a, const XTensor &b, XTensor &c,const XTensor &sa, const XTensor &sb, XTensor &sc)`|x与s的加法
`void Mapping(const XTensor &a, XTensor &x, XTensor &s, int corenum)`|	 带上了多线程的上面内容
`void Match(XTensor &x1, XTensor &s1, XTensor &x2, XTensor &s2, int corenum)`|	 带上了多线程的上面内容
`XTensor Match(XTensor &x, XTensor &s, int flag,int corenum)`|	 带上了多线程的上面内容
`void UnMapping(XTensor &a, const  XTensor &x, const  XTensor &s,int corenum)`|	 带上了多线程的上面内容
`void InitMaptensor(XTensor &a, XTensor &x, XTensor &s,int corenum=0)`|通过a对x和s快速初始化
`void MYHardTanH(const XTensor &xa, const XTensor &sa, XTensor &xb, XTensor &sb, int corenum)`|b=hardtanh(a)
`void Initinput(const XTensor &a,  XTensor &x,XTensor &s, int corenum)`|对input初始化
 `void InitMap_Row(const XTensor &a, XTensor &x, XTensor &s, int corenum)`|快速初始化并且对s按行压缩
`void InitMap_1(const XTensor &a, XTensor &x, XTensor &s, int corenum)`|快速初始化并且对s按行列压缩
`void InitMap_Col(const XTensor &a, XTensor &x, XTensor &s, int corenum)`|快速初始化并且对s按列压缩
`XTensor Myadd_int8(const XTensor &a, const XTensor &b, const XTensor &sa,int corenum)`|x与s的加法
---

## [MKL库的安装说明(未用上)](https://blog.csdn.net/qq_39128349/article/details/104484808)
## [部分函数的测试说明](https://blog.csdn.net/qq_39128349/article/details/104314286)

 如果想要直接体验INT8 将`\source\sample\fnnlm\FNNLM.CPP\void Forward(XTensor inputs[], XTensor &output, FNNModel &model, FNNNet &net)`里的注释取消 并且将自带的乘法和加法注释掉
 
 感谢林野学姐,李炎洋学长,单韦乔学长等学长学姐在实习过程中对我的帮助 
