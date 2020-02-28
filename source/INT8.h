#ifndef __MYSUM_H__
#define __MYSUM_H__
#include"./tensor/XTensor.h"
#include <thread>
#define MAXNDATANUM 10000000 ////乘法所需的临时数组大小
namespace nts { // namespace nts(NiuTrans.Tensor)
	
	void MySum(const XTensor &a, const XTensor &b, XTensor &c); //朴素的int加法
	void Mymul(const XTensor &a, const XTensor &b, XTensor &c,int corenum=1);// int乘法
	void Mymul_c(const XTensor &a, const XTensor &b, XTensor &c, int corenum = 1);//int_8 乘法
	void Mymul_f(const float *a, const float *b, const int &leni, const int &lenj, const int &lenk, float  *c, int corenum = 1);//float乘法
    /*溢出判断*/
	bool OverAddChar(const char &a, const char &b, const char &c);
	bool OverAddInt(const int &a, const int &b, const int &c);
	bool OverMulChar(const char &a, const char &b, const char &c);
	bool OverMulInt(const int &a, const int &b, const int &c);
	void MYMatrixMul(const XTensor &a, const XTensor &b, XTensor &c,int corenum=1);//带映射的int乘法
	void Show(const XTensor &x, const XTensor &s, int flag = 0); //输出反映射后的结果,s为一维向量
	///////////////
	void Mapping(const XTensor &a, XTensor &x, XTensor &s); //将a映射为x和s a,s为float型 x为int_8型 
	void UnMapping(XTensor &a, const  XTensor &x, const  XTensor &s);//通过x和s反映射为a
	XTensor Match(XTensor &x, XTensor &s, int flag);// 压缩s为乘法做准备 返回的一维向量
	void Match(XTensor &x1, XTensor &s1, XTensor &x2, XTensor &s2);//调整s1和s2为加法做准备
	void  Mymul_int8(const XTensor &a, const XTensor &b, XTensor &c,const XTensor &sa, const XTensor &sb, XTensor &sc, int corenum);//x与s的乘法
	void Myadd_int8(const XTensor &a, const XTensor &b, XTensor &c,const XTensor &sa, const XTensor &sb, XTensor &sc);//x与s的加法
	/*下面是带上了多线程的上面内容*/
	void Mapping(const XTensor &a, XTensor &x, XTensor &s, int corenum);
	void Match(XTensor &x1, XTensor &s1, XTensor &x2, XTensor &s2, int corenum);
	XTensor Match(XTensor &x, XTensor &s, int flag,int corenum);
	void UnMapping(XTensor &a, const  XTensor &x, const  XTensor &s,int corenum);
	void InitMaptensor(XTensor &a, XTensor &x, XTensor &s,int corenum=0);//通过a对x和s快速初始化
	void MYHardTanH(const XTensor &xa, const XTensor &sa, XTensor &xb, XTensor &sb, int corenum);//b=hardtanh(a)
	void Initinput(const XTensor &a,  XTensor &x,XTensor &s, int corenum);//对input初始化
 	void InitMap_Row(const XTensor &a, XTensor &x, XTensor &s, int corenum);//快速初始化并且对s按行压缩
	void InitMap_1(const XTensor &a, XTensor &x, XTensor &s, int corenum);//快速初始化并且对s按行列压缩
	void InitMap_Col(const XTensor &a, XTensor &x, XTensor &s, int corenum);//快速初始化并且对s按列压缩
	XTensor Myadd_int8(const XTensor &a, const XTensor &b, const XTensor &sa,int corenum);//x与s的加法
} // namespace nts(NiuTrans.Tensor)

#endif //  

 