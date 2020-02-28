#ifndef __MYSUM_H__
#define __MYSUM_H__
#include"./tensor/XTensor.h"
#include <thread>
#define MAXNDATANUM 10000000 ////�˷��������ʱ�����С
namespace nts { // namespace nts(NiuTrans.Tensor)
	
	void MySum(const XTensor &a, const XTensor &b, XTensor &c); //���ص�int�ӷ�
	void Mymul(const XTensor &a, const XTensor &b, XTensor &c,int corenum=1);// int�˷�
	void Mymul_c(const XTensor &a, const XTensor &b, XTensor &c, int corenum = 1);//int_8 �˷�
	void Mymul_f(const float *a, const float *b, const int &leni, const int &lenj, const int &lenk, float  *c, int corenum = 1);//float�˷�
    /*����ж�*/
	bool OverAddChar(const char &a, const char &b, const char &c);
	bool OverAddInt(const int &a, const int &b, const int &c);
	bool OverMulChar(const char &a, const char &b, const char &c);
	bool OverMulInt(const int &a, const int &b, const int &c);
	void MYMatrixMul(const XTensor &a, const XTensor &b, XTensor &c,int corenum=1);//��ӳ���int�˷�
	void Show(const XTensor &x, const XTensor &s, int flag = 0); //�����ӳ���Ľ��,sΪһά����
	///////////////
	void Mapping(const XTensor &a, XTensor &x, XTensor &s); //��aӳ��Ϊx��s a,sΪfloat�� xΪint_8�� 
	void UnMapping(XTensor &a, const  XTensor &x, const  XTensor &s);//ͨ��x��s��ӳ��Ϊa
	XTensor Match(XTensor &x, XTensor &s, int flag);// ѹ��sΪ�˷���׼�� ���ص�һά����
	void Match(XTensor &x1, XTensor &s1, XTensor &x2, XTensor &s2);//����s1��s2Ϊ�ӷ���׼��
	void  Mymul_int8(const XTensor &a, const XTensor &b, XTensor &c,const XTensor &sa, const XTensor &sb, XTensor &sc, int corenum);//x��s�ĳ˷�
	void Myadd_int8(const XTensor &a, const XTensor &b, XTensor &c,const XTensor &sa, const XTensor &sb, XTensor &sc);//x��s�ļӷ�
	/*�����Ǵ����˶��̵߳���������*/
	void Mapping(const XTensor &a, XTensor &x, XTensor &s, int corenum);
	void Match(XTensor &x1, XTensor &s1, XTensor &x2, XTensor &s2, int corenum);
	XTensor Match(XTensor &x, XTensor &s, int flag,int corenum);
	void UnMapping(XTensor &a, const  XTensor &x, const  XTensor &s,int corenum);
	void InitMaptensor(XTensor &a, XTensor &x, XTensor &s,int corenum=0);//ͨ��a��x��s���ٳ�ʼ��
	void MYHardTanH(const XTensor &xa, const XTensor &sa, XTensor &xb, XTensor &sb, int corenum);//b=hardtanh(a)
	void Initinput(const XTensor &a,  XTensor &x,XTensor &s, int corenum);//��input��ʼ��
 	void InitMap_Row(const XTensor &a, XTensor &x, XTensor &s, int corenum);//���ٳ�ʼ�����Ҷ�s����ѹ��
	void InitMap_1(const XTensor &a, XTensor &x, XTensor &s, int corenum);//���ٳ�ʼ�����Ҷ�s������ѹ��
	void InitMap_Col(const XTensor &a, XTensor &x, XTensor &s, int corenum);//���ٳ�ʼ�����Ҷ�s����ѹ��
	XTensor Myadd_int8(const XTensor &a, const XTensor &b, const XTensor &sa,int corenum);//x��s�ļӷ�
} // namespace nts(NiuTrans.Tensor)

#endif //  

 