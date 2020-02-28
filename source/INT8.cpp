#include"INT8.h"
#include"tensor/XGlobal.h"
#include<vector>
#include<cmath>
#include<iostream>
#include<algorithm>
namespace nts {
	 inline char round(float x)
	{
		return x>0? (char)(x + 0.5): (char)(x -0.5);
	} 
	float maxf(const float &a, const float &b) //比较大小
	{
		if (a > b) return a;
		else return b;
	}
	int maxi(const int &a, const int &b) {
		if (a > b) return a;
		else return b;
	}
	float minf(const float &a, const float &b) //比较大小
	{
		if (a > b) return b;
		else return a;
	}
	void MySum(const XTensor &a, const XTensor &b, XTensor &c)
	{
		CheckNTErrors(a.unitNum == b.unitNum && a.unitNum == c.unitNum,
			"Unmatched tensors in addition!");
		CheckNTErrors(a.order == b.order, "order don't match");
		for (int i = 0; i < a.order; i++)
		{
			CheckNTErrors(a.dimSize[i] == b.dimSize[i], "ordersize don't match");
		}
		register int i = 0, num = a.unitNum;  
		char* cp = (char*)c.data;
		char* ap = (char*)a.data;
		char* bp = (char*)b.data;
	 
		int x = num - 4;
		for (i; i < x; i += 4) //直接四个一起做 最后不足四个在另算
		{
			cp[i] = ap[i] + bp[i];
			cp[i + 1] = ap[i + 1] + bp[i + 1];
			cp[i + 2] = ap[i + 2] + bp[i + 2];
			cp[i + 3] = ap[i + 3] + bp[i + 3];
		}
		while (i < num) {
			cp[i] = ap[i] + bp[i];
			++i;
		}
	}
	auto fun = [&](const int &bg, const int &ed, const int &leni, const int &lenj, const int &lenk   //多线程所用匿名函数 完成一个线程的乘法
		, const int *a, const int *b, long long *c)
	{
		for (register int i = bg; i < ed; i++)
			for (register int k = 0; k < lenk; k++)	
				for (register int j = 0; j < lenj; j++)
				{
					*(c + i * lenj + j) += 1ll * (*(a + i * lenk + k))*(*(b + k * lenj + j));
				}
			 

	}; 
	auto fun_int = [&](const int &bg, const int &ed, const int &leni, const int &lenj, const int &lenk
		, const int *a, const int *b, int *c)
	{
		for (register int i = bg; i < ed; i++)
			for (register int k = 0; k < lenk; k++)
				for (register int j = 0; j < lenj; j++)
				{
					*(c + i * lenj + j) +=  (*(a + i * lenk + k))*(*(b + k * lenj + j));
				}


	};
	auto fun_f = [&](const int &bg, const int &ed, const int &leni, const int &lenj, const int &lenk
		, const float *a, const float *b, float  *c)
	{
		for (register int i = bg; i < ed; ++i)
			for (register int k = 0; k < lenk; ++k)
				for (register int j = 0; j < lenj; ++j)
				{
					*(c + i * lenj + j) +=   (*(a + i * lenk + k))*(*(b + k * lenj + j));
				}

	};
 
	auto fun_c = [&](const int &bg, const int &ed, const int &leni, const int &lenj, const int &lenk
		, const char *a, const char *b, int  *c)
	{
		for (register int i = bg; i < ed; ++i)
			for (register int k = 0; k < lenk; ++k)
			{
				register int j;
				for (  j = 0; j < lenj-4; j+=4)
				{
					*(c + i * lenj + j) += (int)(*(a + i * lenk + k))*(*(b + k * lenj + j));
					*(c + i * lenj + j+1) += (int)(*(a + i * lenk + k))*(*(b + k * lenj + j+1));
					*(c + i * lenj + j+2) += (int)(*(a + i * lenk + k))*(*(b + k * lenj + j+2));
					*(c + i * lenj + j+3) += (int)(*(a + i * lenk + k))*(*(b + k * lenj + j+3));
				}
				while (j < lenj)
				{
					*(c + i * lenj + j) += (int)(*(a + i * lenk + k))*(*(b + k * lenj + j));
					++j;
				}
			}

	};
	void Mymul(const int *a, const int *b, const int &leni, const int &lenj, const int &lenk, long long *c,int corenum)
	{
		std::vector<std::thread> v; //管理线程的向量
		int bg, ed;
		bg = -leni / corenum;
		ed = 0;
		for (int i = 0; i < corenum; i++)
		{
			ed += leni / corenum;
			bg += leni / corenum;
			if (i == corenum - 1) ed += leni % corenum;
			v.push_back(std::thread(fun, bg, ed, leni, lenj, lenk, a, b, c));
		}
		for (int i = 0; i < corenum; i++)
			v[i].join();
	}
	void Mymul(const char *a, const char *b, const int &leni, const int &lenj, const int &lenk, int *c, int corenum)
	{
		std::vector<std::thread> v;
		int bg, ed;
		bg = -leni / corenum;
		ed = 0;
		for (register int i = 0; i < corenum; ++i)
		{
			ed += leni / corenum;
			bg += leni / corenum;
			if (i == corenum - 1) ed += leni % corenum;
			v.push_back(std::thread(fun_c, bg, ed, leni, lenj, lenk, a, b, c));
		}
		for (int i = 0; i < corenum; ++i)
			v[i].join();
	}
	void Mymul_f(const float *a, const float *b, const int &leni, const int &lenj, const int &lenk, float  *c, int corenum)
	{
		std::vector<std::thread> v;
		int bg, ed;
		bg = -leni / corenum;
		ed = 0;
		for (int i = 0; i < corenum; ++i)
		{
			ed += leni / corenum;
			bg += leni / corenum;
			if (i == corenum - 1) ed += leni % corenum;
			v.push_back(std::thread(fun_f, bg, ed, leni, lenj, lenk, a, b, c));
		}
		for (int i = 0; i < corenum; ++i)
			v[i].join();
	}
	void Mymul(const XTensor &a, const XTensor &b, XTensor &c, int corenum)
	{
		CheckNTErrors(a.dataType == b.dataType, "Input tensors should have the same data type!");
		CheckNTErrors(a.order >= 2 && b.order >= 2, "Input tensors must have a order >= 2!");
		CheckNTErrors(a.dimSize[0] == c.dimSize[0] && a.dimSize[1] == b.dimSize[0] && b.dimSize[1] == c.dimSize[1], "no match dimsize");
		std::vector<std::thread> v;
		int bg, ed, leni, lenj, lenk;
		 	leni = a.dimSize[0], lenk = a.dimSize[1], lenj = b.dimSize[1];
		bg = -leni / corenum;
		ed = 0;
		int* cp = (int*)c.data;
		int* ap = (int*)a.data;
		int* bp = (int*)b.data;
		for (int i = 0; i < corenum; i++)
		{
			ed += leni / corenum;
			bg += leni / corenum;
			if (i == corenum - 1) ed += leni % corenum;
			 v.push_back(std::thread(fun_int, bg, ed, leni, lenj, lenk, ap, bp, cp));
		}
		for (int i = 0; i < corenum; i++)
			v[i].join();
	}
	bool OverAddChar(const char &a, const char &b, const char &c)
	{
		return ((((a >= 0) && (b >= 0) && (c < 0)) || \
			((a < 0) && (b < 0) && (c >= 0))));
	}
	bool OverAddInt(const int &a, const int &b, const int &c)
	{
		return ((((a >= 0) && (b >= 0) && (c < 0)) || \
			((a < 0) && (b < 0) && (c >= 0))));
	}
	bool OverMulChar(const char &a, const char &b, const char &c)
	{
		return ((((c) / a) != b));
	}
	bool OverMulInt(const int &a, const int &b, const int &c)
	{
		return ((((c) / a) != b));
	}
	void Show(const XTensor &x, const XTensor &s,int flag)
	{
		if (flag==0)
		{
			for (int i = 0; i < x.dimSize[0]; i++)
			{
				for (int j = 0; j < x.dimSize[1]; j++)
					std::cout << *((int*)x.data + i * x.dimSize[1] + j)*1.0 / (*((float*)s.data + i))<<" ";
			}
			std::cout << std::endl;
		}
		else
		{
			for (int i = 0; i < x.dimSize[0]; i++)
			{
				for (int j = 0; j < x.dimSize[1]; j++)
					std::cout << *((int*)x.data + i * x.dimSize[1] + j)*1.0 / (*((float*)s.data + j)) << " ";
					std::cout << std::endl;
			}
		}
	}
	 
	int data[MAXNDATANUM];
	void MYMatrixMul(const XTensor &a, const XTensor &b, XTensor &c,int corenum)
	{
		XTensor xa, xb, sa, sb, xc, sc;
		float maxx;
		float *sdata = (float *)malloc(xa.dimSize[0] * xb.dimSize[1] * sizeof(float));
		long long *cdata = (long long *)malloc(xa.dimSize[0] * xb.dimSize[1] * sizeof(long long));
		InitTensor2D(&xa, a.dimSize[0], a.dimSize[1], X_INT); InitTensor2D(&sa, a.dimSize[0], 1);
		InitTensor2D(&xb, b.dimSize[0], b.dimSize[1], X_INT); InitTensor2D(&sb, 1, b.dimSize[1]);
		InitTensor2D(&xc, a.dimSize[0], b.dimSize[1], X_INT); InitTensor2D(&sc, a.dimSize[0], b.dimSize[1]);
	//	 std::cout << "初始化" << clock() - st << std::endl;
		for (register int i = 0; i < a.dimSize[0]; i++)
		{
			maxx = std::fabs(*((float*)a.data + i * a.dimSize[1]));
			for (register int j = 0; j < a.dimSize[1]; j++)
			{
				maxx =  maxf(maxx, std::fabs(*((float*)a.data + i*a.dimSize[1]+j)));
			}
			sdata[i] = 2e9 / maxx;
			for (register int j = 0; j < a.dimSize[1]; j++)
			{
				data[i*a.dimSize[1] + j] = (int)std::round(sdata[i] * (*((float*)a.data + i * a.dimSize[1] + j)));
			}
		}
		xa.SetData(data, xa.unitNum); sa.SetData(sdata, sa.unitNum);
	//	 std::cout << "映射1" << clock() - st << std::endl;
		for (register int j = 0; j < b.dimSize[1]; j++)
		{
			maxx = std::fabs(*((float*)b.data + j));
			for (register int i = 0; i < b.dimSize[0]; i++)
			{
				maxx =  maxf(maxx, std::fabs(*((float*)b.data + i * b.dimSize[1] + j)));
			}
			sdata[j] = 2e9 / maxx;
			for (register int i = 0; i < b.dimSize[0]; i++)
			{
				data[i*b.dimSize[1]+j]= (int)std::round(sdata[j]* (*((float*)b.data + i * b.dimSize[1] + j)));
			}
		}
		xb.SetData(data, xb.unitNum); sb.SetData(sdata, sb.unitNum);
		// std::cout << "映射2" << clock() - st << std::endl;
		memset(cdata, 0, xa.dimSize[0] * xb.dimSize[1] * sizeof(long long));
		memset(sdata, 0, xa.dimSize[0] * xb.dimSize[1] * sizeof(float));
		//  std::cout << "初始化2" << clock() - st << std::endl;
		Mymul((int*)xa.data, (int*)xb.data, xa.dimSize[0],xb.dimSize[1],xa.dimSize[1],cdata,corenum);
		//std::cout << "数乘" << clock() - st << std::endl;
		Mymul_f((float*)sa.data, (float*)sb.data,sa.dimSize[0],sb.dimSize[1],1,(float*)sdata,corenum);
		//std::cout << "比例乘" << clock() - st << std::endl;
		int ni = xa.dimSize[0], nj = xb.dimSize[1];
		//  std::cout << "xa xs" << std::endl;
		//  Show(xa, sa);
		//   xa.Dump(stderr); sa.Dump(stderr);
		// std::cout << "xb xs" << std::endl;
		//  Show(xb, sb,1);
		//   xb.Dump(stderr); sb.Dump(stderr);
		//  std::cout << "ans" << std::endl;
		for (int i = 0; i < ni; i++)
		{
			for (int j = 0; j < nj; j++)
			{
				*((float*)(c.data) + i * nj + j) = (*(cdata + i * nj + j))*1.0 / (*(sdata + i * nj + j));
				//std::cout << *((float*)(c.data) + i * nj + j) << " " << (*(cdata + i * nj + j)) << " " << (*(sdata + i * nj + j)) << std::endl;;
			}
			// std::cout << std::endl;
		}		
		//std::cout << "反映射" << clock() - st << std::endl;
	}
 
	void Mymul_c(const XTensor &a, const XTensor &b, XTensor &c, int corenum )
	{
		char da[MAXNDATANUM], db[MAXNDATANUM];
		XTensor xa, xb, sa, sb, xc, sc;
		float maxx;
		float *sdata = (float *)malloc(xa.dimSize[0] * xb.dimSize[1] * sizeof(float));
		long long *cdata = (long long *)malloc(xa.dimSize[0] * xb.dimSize[1] * sizeof(long long));
		float sfa[MAXNDATANUM], sfb[MAXNDATANUM];
		 
	 //std::cout <<"b size"<<b.dimSize[0] << " "<<b.dimSize[1]<<" "<<clock() - st << std::endl;
		double st = clock();
	//	std::cout << "a size" <<a.dimSize[0] << " " <<a.dimSize[1] << " " << clock() - st << std::endl;
		for (register int i = 0; i < a.dimSize[0]; i++)
		{
			maxx = std::fabs(*((float*)a.data + i * a.dimSize[1]));
			 for (register int j = 0; j < a.dimSize[1]; j++)
			{
				maxx = maxf(maxx, std::fabs(*((float*)a.data + i * a.dimSize[1] + j)));
			} 
			sfa[i] = 1.0*127/ maxx;
			 
		//	sfa[i] = 100.0 / maxx;
			for (register int j = 0; j < a.dimSize[1]; j++)
			{
				da[i*a.dimSize[1] + j] = round(sfa[i] * (*((float*)a.data + i * a.dimSize[1] + j)));
			}
		}
	//	std::cout << "映射1:" << clock() - st << std::endl;
		/*for (register int j = 0; j < b.dimSize[1]; j++)
		{
			maxx = std::fabs(*((float*)b.data + j));
			for (register int i = 0; i < b.dimSize[0]; i++)
			{
				maxx = maxf(maxx, std::fabs(*((float*)b.data + i * b.dimSize[1] + j)));
			}
			sfb[j] = 127.0 / maxx;
			for (register int i = 0; i < b.dimSize[0]; i++)
			{
				db[i*b.dimSize[1] + j] = (char)std::round(sfb[j] * (*((float*)b.data + i * b.dimSize[1] + j)));
			}
		}*/
		register int tmp;
	//	std::cout <<"b size"<<b.dimSize[0] << " "<<b.dimSize[1]<<" "<<clock() - st << std::endl;
	 for (register int i = 0; i < b.dimSize[0]; ++i)
		{
			tmp = i * b.dimSize[1];
			for (register int j = 0; j < b.dimSize[1]; ++j)
			{
				if (i == 0)
					sfb[j] = std::fabs(*((float*)b.data + j));
				else sfb[j] = maxf(sfb[j], std::fabs(*((float*)b.data + tmp + j)));
			}
	 	} 
	//	std::cout << "映射2.1 :" << clock() - st << std::endl;
		for (register int i = 0; i < b.dimSize[0]; ++i)
		{
			tmp = i * b.dimSize[1];
			for (register int j = 0; j < b.dimSize[1]; ++j)
			{
				 if(i==0) sfb[j] = 127.0 / sfb[j];
				//if(i==0) sfb[j] = 100.0 / std::fabs(*((float*)b.data + j));;
				 db[tmp+ j] =  round(sfb[j] * (*((float*)b.data + tmp + j)));
			}
		}
	//	std::cout << "映射2.2:" << clock() - st << std::endl;
	 	 memset(data, 0, a.dimSize[0] * b.dimSize[1] * sizeof(int));
		 memset(sdata, 0, a.dimSize[0] * b.dimSize[1] * sizeof(float));
	//	 std::cout << "初始化" << clock() - st << std::endl;
		Mymul(da, db, a.dimSize[0], b.dimSize[1], a.dimSize[1], data, corenum);
	//	std::cout << "数乘" << clock() - st << std::endl;
		Mymul_f(sfa, sfb, a.dimSize[0], b.dimSize[1],1,sdata, corenum);
	//	std::cout << "比例乘" << clock() - st << std::endl;
		int ni = a.dimSize[0], nj = b.dimSize[1];
		for (int i = 0; i < c.unitNum; i++)
		{
			 
				*((float*)(c.data) + i ) = (*(data + i))*1.0 / (*(sdata + i));
				//std::cout << *((float*)(c.data) + i * nj + j) << " " << (*(data + i * nj + j)) << " " << (*(sdata + i * nj + j)) << std::endl;;
			 
			// std::cout << std::endl;
		}
	//	std::cout << "反映射" << clock() - st << std::endl;
	}
	
	////////////////////////////////////////////////
	void Mapping(const XTensor &a,XTensor &x ,XTensor &s)
	{
		CheckNTErrors(a.dataType == s.dataType &&x.dataType==X_INT8&&s.dataType == X_FLOAT, "Tensors should have the right data type!");
		CheckNTErrors(a.order == s.order &&s.order ==  x.order&&s.order == 2, "Tensors should have the right order!");
		CheckNTErrors(a.dimSize[0] == s.dimSize[0]&& s.dimSize[0] == x.dimSize[0]&&
					  a.dimSize[1] == s.dimSize[1]&& s.dimSize[1] == x.dimSize[1], "Tensors should have the right dimsize!");
		register int tmp;
		float *ap = (float *)a.data,*sp= (float *)s.data;
		char *xp = (char*)x.data;
		for (register int i = 0; i < a.dimSize[0]; i++)
		{
			tmp =  i * a.dimSize[1];
			 for (register int j = 0; j < a.dimSize[1]; j++)
			{
				 *(sp + tmp + j) = 127.0 / fabs(*(ap + tmp + j));
				 *(xp + tmp + j) = (char)round(*(sp + tmp + j)*(*(ap + tmp + j)));
				 
			 } 
		}
	}
	void UnMapping( XTensor &a, const  XTensor &x, const  XTensor &s)
	{
		CheckNTErrors(a.dataType == s.dataType &&x.dataType == X_INT8 && s.dataType == X_FLOAT, "Tensors should have the right data type!");
		CheckNTErrors(a.order == s.order &&s.order == x.order&&s.order == 2, "Tensors should have the right order!");
		CheckNTErrors(a.dimSize[0] == s.dimSize[0] && s.dimSize[0] == x.dimSize[0] &&
			a.dimSize[1] == s.dimSize[1] && s.dimSize[1] == x.dimSize[1], "Tensors should have the right dimsize!");
		register int tmp;
		float *ap = (float *)a.data, *sp = (float *)s.data;
		char *xp = (char*)x.data;
		for (register int i = 0; i < a.dimSize[0]; i++)
		{
			tmp = i * a.dimSize[1];
			for (register int j = 0; j < a.dimSize[1]; j++)
			{
			 
				*(ap + tmp + j) = *(xp + tmp + j)*1.0 / (*(sp + tmp + j));
			}
		}
	}
	XTensor Match(XTensor &x, XTensor &s,int flag)
	{
		CheckNTErrors( x.dataType == X_INT8 && s.dataType == X_FLOAT, "Tensors should have the right data type!");
		CheckNTErrors( s.order == x.order&&s.order == 2, "Tensors should have the right order!");
		CheckNTErrors( s.dimSize[0] == x.dimSize[0] &&
			           s.dimSize[1] == x.dimSize[1], "Tensors should have the right dimsize!");
		XTensor tmp;
		char *dx = (char*)x.data;
		float *ds = (float*)s.data;
		if (flag == 0)
			InitTensor2D(&tmp, x.dimSize[0], 1);
		else InitTensor2D(&tmp, 1, x.dimSize[1]);
		float *dt = (float*)tmp.data;
		for(register int i =0;i<x.dimSize[0];i++)
			for (register int j = 0; j < x.dimSize[1]; j++)
			{
				if (flag == 0)
				{
					if (j == 0)
						*(dt+i) = *(ds+i*s.dimSize[1]+j);
					else *(dt+i) = minf(*(dt + i), *(ds + i * s.dimSize[1] + j));
				}
				else
				{ 
					if(i==0)
						*(dt+ j) = *(ds+ i * s.dimSize[1] + j);
					else *(dt + j) = minf(*(dt + j), *(ds + i * s.dimSize[1] + j));
				}
			}
		for (register int i = 0; i < x.dimSize[0]; i++)
			for (register int j = 0; j < x.dimSize[1]; j++)
			{
				if (flag == 0)
				{
					*(dx + i * x.dimSize[1] + j) = (char) round(*(dx + i * x.dimSize[1] + j)*1.0 / (std::ceil(*(ds + i * x.dimSize[1] + j)/ *(dt + i))));
					*(ds + i * x.dimSize[1] + j) = *(dt + i);
				}
				else
				{ 
					*(dx + i * x.dimSize[1] + j) = (char) round(*(dx + i * x.dimSize[1] + j)*1.0 / (std::ceil(*(ds + i * x.dimSize[1] + j) / *(dt + j))));
					*(ds + i * x.dimSize[1] + j) = *(dt + j);
				}
			}
		return tmp;
	}
	void Match(XTensor &x1, XTensor &s1, XTensor &x2, XTensor &s2)
	{
		CheckNTErrors(s1.dataType == s2.dataType &&x1.dataType == X_INT8 && s1.dataType == X_FLOAT&x1.dataType==x2.dataType, "Tensors should have the right data type!");
		CheckNTErrors(s1.order == s2.order &&s1.order == x1.order&&s1.order == x2.order&&s1.order==2, "Tensors should have the right order!");
		CheckNTErrors(s1.dimSize[0] == s2.dimSize[0] && s1.dimSize[0] == x1.dimSize[0] && s1.dimSize[0] == x2.dimSize[0] &&
			s1.dimSize[1] == s1.dimSize[1] && s1.dimSize[1] == x1.dimSize[1] && s1.dimSize[1] == x2.dimSize[1], "Tensors should have the right dimsize!");
		float *sd1 = (float*)s1.data, *sd2 = (float*)s2.data;
		char *xd1 = (char*)x1.data, *xd2 = (char*)x2.data;
		for(int i =0;i<x1.dimSize[0];i++)
			for (int j = 0; j < x1.dimSize[1]; j++)
			{
				if (*(sd1 + i * x1.dimSize[1] + j) <= *(sd2 + i * x1.dimSize[1] + j))
				{
					*(xd2 + i * x1.dimSize[1] + j) = (char) round(*(xd2 + i * x1.dimSize[1] + j)*1.0 / (std::ceil(*(sd2 + i * x1.dimSize[1] + j) / *(sd1 + i * x1.dimSize[1] + j))));
					*(sd2 + i * x1.dimSize[1] + j) = *(sd1 + i * x1.dimSize[1] + j);
				}
				else
				{
					*(xd1 + i * x1.dimSize[1] + j) = (char) round(*(xd1 + i * x1.dimSize[1] + j)*1.0 / (std::ceil(*(sd1 + i * x1.dimSize[1] + j) / *(sd2 + i * x1.dimSize[1] + j))));
					*(sd1 + i * x1.dimSize[1] + j) = *(sd2 + i * x1.dimSize[1] + j);
				}
			}
	}
	char judge(const int &tmp, float &s) //判断结果是否超过127 若超过 则重新映射
	{
		if (std::abs(tmp) > 127)
		{
			float t = 127.0 / std::abs(tmp);
			s = t * s;
			return (char) round(tmp*t);
		}
		else return (char)tmp;
	}
	auto fun_mul_fi = [&](const int &bg, const int &ed, const int &lenj,int *tmp,const int *data)
	{
		for (register int i = bg; i < ed; ++i)
		{
			for (register int j = 0; j <lenj; j++)
			{
				if (i == 0) tmp[j] = abs(data[i*lenj + j]);
				else
					tmp[j] = maxi(tmp[j], abs(data[i*lenj + j]));
			}
		}
	};
	auto fun_mul_unmap = [&](const int &bg, const int &ed, const int &lenj, char *dc, const int *data,float *ds)
	{
		for (register int i = bg; i < ed; ++i)
		{
			register int j;
			for (j = 0; j < lenj - 4; j += 4)
			{
				dc[i*lenj+j] = judge(data[i*lenj + j], ds[i*lenj + j]);
				dc[i*lenj + j + 1] = judge(data[i*lenj + j + 1], ds[i*lenj + j + 1]);
				dc[i*lenj + j + 2] = judge(data[i*lenj + j + 2], ds[i*lenj + j + 2]);
				dc[i*lenj + j + 3] = judge(data[i*lenj + j + 3], ds[i*lenj + j + 3]);
			}
			 
			while (j < lenj) {
				dc[i*lenj + j] = judge(data[i*lenj + j], ds[i*lenj + j]);
				++j;
			}
		}
	};
	auto fun_mul_unmap2 = [&](const int &bg, const int &ed, const int &lenj, char *dc, const int *data, float *ds)
	{
		for (register int i = bg; i < ed; ++i)
		{
			int tmp = abs(data[i*lenj]);
			register int j = 0;
			for ( j = 0; j < lenj-4; j+=4)
			{

				tmp = maxi(tmp, abs(data[i*lenj + j]));
				tmp = maxi(tmp, abs(data[i*lenj + j+1]));
				tmp = maxi(tmp, abs(data[i*lenj + j+2]));
				tmp = maxi(tmp, abs(data[i*lenj + j+3]));
			}
			while (j < lenj) { tmp = maxi(tmp, abs(data[i*lenj + j])); ++j; }
			float ts = 127.0 / tmp; // ts<1
			//std::cout << tmp << " " << ts << " " << ds[i] << " " << ds[i] * ts << "\n";
			ds[i] = ds[i] * ts;
			for ( j = 0; j < lenj - 4; j+=4)
			{
				if (ts < 1)
				{
					dc[i*lenj + j] = (char)round(data[i*lenj + j] * ts);
					dc[i*lenj + j+1] = (char)round(data[i*lenj + j+1] * ts);
					dc[i*lenj + j+2] = (char)round(data[i*lenj + j+2] * ts);
					dc[i*lenj + j+3] = (char)round(data[i*lenj + j+3] * ts);

				}
				else
				{
					dc[i*lenj + j] = (char)round(data[i*lenj + j]);
					dc[i*lenj + j+1] = (char)round(data[i*lenj + j+1]);
					dc[i*lenj + j+2] = (char)round(data[i*lenj + j+2]);
					dc[i*lenj + j+3] = (char)round(data[i*lenj + j+3]);
				}
				//std::cout << dc[i*c.dimSize[1] + j] << " " << ds[i] << std::endl;
			}
			while (j < lenj) {  
				if (ts < 1)
				{
					dc[i*lenj + j] = (char)round(data[i*lenj + j] * ts);				 
				}
				else
				{
					dc[i*lenj + j] = (char)round(data[i*lenj + j]);
				}
				++j; }
		}
	};
	void  Mymul_int8(const XTensor &a, const XTensor &b,XTensor &c,
		             const XTensor &sa, const XTensor &sb, XTensor &sc, int corenum)
	{

		CheckNTErrors(a.dataType == b.dataType&&a.dataType==X_INT8, "Input tensors should have the same data type!");
		CheckNTErrors(a.order == 2 && b.order == 2, "Input tensors must have a order = 2!");
		CheckNTErrors(a.dimSize[0] == c.dimSize[0] && a.dimSize[1] == b.dimSize[0] && b.dimSize[1] == c.dimSize[1], "no match dimsize");
 
		char *da = (char*)a.data;
		char *db = (char*)b.data;
		char *dc = (char*)c.data;
		float *ds = (float*)sc.data;
		memset(data, 0, a.dimSize[0] * b.dimSize[1] * sizeof(int));
		Mymul(da, db, a.dimSize[0], b.dimSize[1], a.dimSize[1],data, corenum);
		memset(sc.data, 0, sa.dimSize[0] * sb.dimSize[1] * sizeof(float));
		Mymul_f((float*)sa.data, (float*)sb.data, sa.dimSize[0], sb.dimSize[1], sa.dimSize[1], (float*)sc.data, corenum);
		//  double st = clock();
		std::vector<std::thread> v;
		int bg, ed;
		int leni = a.dimSize[0];
		if (sc.dimSize[0] == c.dimSize[0])
		{
		 	//std::cout << "h1\n";
			if (sc.dimSize[1] == c.dimSize[1]) {
		 	// std::cout << "h2 n";
				 
				bg = -leni / corenum;
				ed = 0;
				register int i;
				for (i = 0; i < corenum; ++i)
				{
					ed += leni / corenum;
					bg += leni / corenum;
					if (i == corenum - 1) ed += leni % corenum;
					v.push_back(std::thread(fun_mul_unmap, bg, ed, c.dimSize[1], dc, data,ds));
				}
				for (i = 0; i < corenum; ++i)
					v[i].join();
				 
			}
			else
			{
			 	//std::cout << "h3 ";
				 
				bg = -leni / corenum;
				ed = 0;
				register int i;
				for (i = 0; i < corenum; ++i)
				{
					ed += leni / corenum;
					bg += leni / corenum;
					if (i == corenum - 1) ed += leni % corenum;
					v.push_back(std::thread(fun_mul_unmap2, bg, ed, c.dimSize[1], dc, data, ds));
				}
				for (i = 0; i < corenum; ++i)
					v[i].join();
 
			}

		}
		else
		{
			//std::cout << "h4 ";
			int *tmp = (int*)malloc(sizeof(int)*c.dimSize[1]);
			float *ts= (float*)malloc(sizeof(float)*c.dimSize[1]);
			bg = -leni / corenum;
			ed = 0;
			register int i;
			for ( i = 0; i < corenum; ++i)
			{
				ed += leni / corenum;
				bg += leni / corenum;
				if (i == corenum - 1) ed += leni % corenum;
				v.push_back(std::thread(fun_mul_fi, bg, ed, c.dimSize[1], tmp, data));
			}
			for (i = 0; i < corenum; ++i)
				v[i].join();
		 
		 
			 
			for (int i = 0; i < c.dimSize[1]; i++) ts[i] = 127.0 / tmp[i];
			for (register int i = 0; i < c.dimSize[0]; ++i)
				for (register int j = 0; j < c.dimSize[1]; ++j)
				{
					if (ts[j] < 1)
					{
						dc[i*c.dimSize[1] + j] = (char)round(data[i*c.dimSize[1] + j] * ts[j]);

					}
					else dc[i*c.dimSize[1] + j] = (char)round(data[i*c.dimSize[1] + j]);
					//std::cout << dc[i*c.dimSize[1] + j] << " " << ds[i] << std::endl;
				}
			}
		 
		//  std::cout << clock() - st << std::endl;

	} 
 
	void Myadd_int8(const XTensor &a, const XTensor &b, XTensor &c,
		const XTensor &sa, const XTensor &sb, XTensor &sc)
	{
		//MySum(a, b, c);
		char* cp = (char*)c.data;
		char* ap = (char*)a.data;
		char* bp = (char*)b.data;
		float *da = (float*)sa.data, *db = (float*)sb.data, *dc = (float*)sc.data;
		memcpy(sc.data, sa.data, sa.unitNum*sizeof(float));
	//	for (int i = 0; i < sa.unitNum; i++)
			//std::cout << (int)ap[i] << " " << (int)bp[i]  << " " << da[i] << " " << dc[i] << std::endl;;
		int x = a.unitNum - 4,i=0;
		for (i; i < x; i += 4)
		{
			
			cp[i] = judge((int)ap[i] + (int)bp[i],dc[i]);
			cp[i + 1] = judge((int)ap[i+1] + (int)bp[i+1], dc[i+1]);
			cp[i + 2] = judge((int)ap[i+2] + (int)bp[i+2], dc[i+2]);
			cp[i + 3] = judge((int)ap[i+3] + (int)bp[i+3], dc[i+3]);
		}
		while (i < a.unitNum) {
			cp[i] = judge((int)ap[i] + bp[i], dc[i]);
			++i;
		}
		//for (int i = 0; i < sa.unitNum; i++)
		//	std::cout << (int)cp[i]  << " "<<dc[i]<<std::endl;;

	}
	 //////尝试并行map/////
	auto fun_mapping = [&](const int &bg, const int &ed, const int &lenj,const float *a, char *x,float *s)
	{
		for (register int i = bg; i < ed; i++)
				for (register int j = 0; j < lenj; j++)
				{
					*(s + i*lenj + j) = 127.0 / fabs(*(a + i * lenj + j));
					*(x + i * lenj + j) = (char)round(*(s + i * lenj + j)*(*(a + i * lenj + j)));
				}

	};
	void Mapping(const XTensor &a, XTensor &x, XTensor &s,int corenum)
	{
		CheckNTErrors(a.dataType == s.dataType &&x.dataType == X_INT8 && s.dataType == X_FLOAT, "Tensors should have the right data type!");
		CheckNTErrors(a.order == s.order &&s.order == x.order&&s.order == 2, "Tensors should have the right order!");
		CheckNTErrors(a.dimSize[0] == s.dimSize[0] && s.dimSize[0] == x.dimSize[0] &&
			a.dimSize[1] == s.dimSize[1] && s.dimSize[1] == x.dimSize[1], "Tensors should have the right dimsize!");
		float *ap = (float *)a.data, *sp = (float *)s.data;
		char *xp = (char*)x.data;
		std::vector<std::thread> v;
		int bg, ed;
		//memset(c, 0, leni*lenk * sizeof(long long));
		bg = -a.dimSize[0] / corenum;
		ed = 0;

		for (int i = 0; i < corenum; i++)
		{
			ed += a.dimSize[0] / corenum;
			bg += a.dimSize[0] / corenum;
			if (i == corenum - 1) ed += a.dimSize[0] % corenum;
			v.push_back(std::thread(fun_mapping, bg, ed, a.dimSize[1], ap, xp, sp));
		}
		for (int i = 0; i < corenum; i++)
			v[i].join();
	}
	auto fun_matchadd = [&](const int &bg, const int &ed, const int &lenj, char *x1, char*x2,float *s1,float *s2)
	{
		for (register int i =bg; i < ed; i++)
			for (register int j = 0; j < lenj; j++)
			{
				if (*(s1 + i * lenj + j) <= *(s2 + i * lenj + j))
				{
					*(x2 + i * lenj + j) = (char) round(*(x2 + i * lenj + j)*1.0 / (std::ceil(*(s2 + i * lenj + j) / *(s1 + i * lenj + j))));
					*(s2 + i * lenj + j) = *(s1 + i * lenj + j);
				}
				else
				{
					*(x1 + i * lenj + j) = (char) round(*(x1 + i * lenj + j)*1.0 / (std::ceil(*(s1 + i * lenj + j) / *(s2 + i * lenj + j))));
					*(s1 + i * lenj + j) = *(s2 + i * lenj + j);
				}
			}

	};
	void Match(XTensor &x1, XTensor &s1, XTensor &x2, XTensor &s2,int corenum)
	{
		CheckNTErrors(s1.dataType == s2.dataType &&x1.dataType == X_INT8 && s1.dataType == X_FLOAT & x1.dataType == x2.dataType, "Tensors should have the right data type!");
		CheckNTErrors(s1.order == s2.order &&s1.order == x1.order&&s1.order == x2.order&&s1.order == 2, "Tensors should have the right order!");
		 
		float *sd1 = (float*)s1.data, *sd2 = (float*)s2.data;
		char *xd1 = (char*)x1.data, *xd2 = (char*)x2.data;
		std::vector<std::thread> v;
		int bg, ed;
		//memset(c, 0, leni*lenk * sizeof(long long));
		bg = -x1.dimSize[0] / corenum;
		ed = 0;

		for (int i = 0; i < corenum; i++)
		{
			ed += x1.dimSize[0] / corenum;
			bg += x1.dimSize[0] / corenum;
			if (i == corenum - 1) ed += x1.dimSize[0] % corenum;
			v.push_back(std::thread(fun_matchadd, bg, ed, x1.dimSize[1], xd1,xd2,sd1,sd2));
		}
		for (int i = 0; i < corenum; i++)
			v[i].join();
	}
	auto fun_match1 = [&](const int &bg, const int &ed, const int &lenj,const float *ds, float *dt,const int &flag)
	{
		for (register int i = bg; i < ed; i++)
			for (register int j = 0; j <lenj; j++)
			{
				if (flag == 0)
				{
					if (j == 0)
						*(dt + i) = *(ds + i * lenj + j);
					else *(dt + i) = minf(*(dt + i), *(ds + i  *lenj + j));
				}
				else
				{
					if (i == 0)
						*(dt + j) = *(ds + i *lenj + j);
					else *(dt + j) = minf(*(dt + j), *(ds + i  *lenj + j));
				}
			}

	};
	auto fun_match2 = [&](const int &bg, const int &ed, const int &lenj, char* dx , float *ds,const float *dt, const int &flag)
	{
		for (register int i = bg; i < ed; i++)
			for (register int j = 0; j < lenj; j++)
			{
				if (flag == 0)
				{
					*(dx + i * lenj + j) = (char) round(*(dx + i * lenj + j)*1.0 / (std::ceil(*(ds + i * lenj + j) / *(dt + i))));
					*(ds + i * lenj + j) = *(dt + i);
				}
				else
				{
					*(dx + i * lenj + j) = (char) round(*(dx + i * lenj + j)*1.0 / (std::ceil(*(ds + i * lenj + j) / *(dt + j))));
					*(ds + i * lenj + j) = *(dt + j);
				}
			}

	};
	XTensor Match(XTensor &x, XTensor &s, int flag, int corenum)
	{
		CheckNTErrors( x.dataType == X_INT8 && s.dataType == X_FLOAT, "Tensors should have the right data type!");
		CheckNTErrors( s.order == x.order&&s.order == 2, "Tensors should have the right order!");
		CheckNTErrors(  s.dimSize[0] == x.dimSize[0] &&
			           s.dimSize[1] == x.dimSize[1], "Tensors should have the right dimsize!");
		//std::cout << x.dimSize[0] << " " << flag << std::endl;;
		XTensor tmp;
		char *dx = (char*)x.data;
		float *ds = (float*)s.data;
		if (flag == 0)
			InitTensor2D(&tmp, x.dimSize[0], 1);
		else InitTensor2D(&tmp, 1, x.dimSize[1]);
		float *dt = (float*)tmp.data;
		 
		std::vector<std::thread> v;
		int bg, ed;
		//memset(c, 0, leni*lenk * sizeof(long long));
		bg = -x.dimSize[0] / corenum;
		ed = 0;

		for (int i = 0; i < corenum; i++)
		{
			//std::cout << i << " cor \n";
			ed += x.dimSize[0] / corenum;
			bg += x.dimSize[0] / corenum;
			if (i == corenum - 1) ed += x.dimSize[0] % corenum;
			v.push_back(std::thread(fun_match1, bg, ed, x.dimSize[1], ds, dt, flag));
		}
		for (int i = 0; i < corenum; i++)
			v[i].join();
		//std::cout << "ok1" << std::endl;
		v.clear();
		bg = -x.dimSize[0] / corenum;
		ed = 0;
		for (int i = 0; i < corenum; i++)
		{
			ed += x.dimSize[0] / corenum;
			bg += x.dimSize[0] / corenum;
			if (i == corenum - 1) ed += x.dimSize[0] % corenum;
			v.push_back(std::thread(fun_match2, bg, ed, x.dimSize[1], dx,ds, dt, flag));
		}
		for (int i = 0; i < corenum; i++)
			v[i].join();
		//std::cout << "ok\n";
		return tmp;
	}
	auto fun_unmap = [&](const int &bg, const int &ed, const int &lenj, float *ap, const char *xp, const float *sp,int flag)
	{
		for (register int i = bg; i < ed; ++i)
		{
			register int tmp = i * lenj;
			for (register int j = 0; j < lenj; ++j)
			{
				if(flag==1)
				*(ap + tmp + j) = *(xp + tmp + j)*1.0 / (*(sp + tmp + j));
				else if(flag==2) *(ap + tmp + j) = *(xp + tmp + j)*1.0 / (*(sp+i));
				else *(ap + tmp + j) = *(xp + tmp + j)*1.0 / (*(sp + j));
			}
		}

	};
	 
	void UnMapping(XTensor &a, const  XTensor &x, const  XTensor &s, int corenum)
	{
		CheckNTErrors(a.dataType == s.dataType &&x.dataType == X_INT8 && s.dataType == X_FLOAT, "Tensors should have the right data type!");
		CheckNTErrors(a.order == s.order &&s.order == x.order&&s.order == 2, "Tensors should have the right order!");
	 
		float *ap = (float *)a.data, *sp = (float *)s.data;
		char *xp = (char*)x.data;
		std::vector<std::thread> v;
		int bg, ed,flag;
		bg = -x.dimSize[0] / corenum;
		ed = 0;
		if (s.dimSize[0] == a.dimSize[0])
		{
			if (s.dimSize[1] == a.dimSize[1]) flag = 1;
			else flag = 2;
		}
		else flag = 3;
		for (int i = 0; i < corenum; ++i)
		{
			ed += x.dimSize[0] / corenum;
			bg += x.dimSize[0] / corenum;
			if (i == corenum - 1) ed += x.dimSize[0] % corenum;
			v.push_back(std::thread(fun_unmap, bg, ed, x.dimSize[1], ap, xp, sp,flag));
		}
		for (int i = 0; i < corenum; i++)
			v[i].join();
		 
	}

	void InitMaptensor(XTensor &a, XTensor &x, XTensor &s,int corenum)
	{
		InitTensor2D(&s, a.dimSize[0], a.dimSize[1]);
		InitTensor2D(&x, a.dimSize[0], a.dimSize[1], X_INT8);
		if (corenum)
			Mapping(a, x, s,corenum);
		else Mapping(a, x, s);
	}
	////////////////////////////////
	auto fun_hardtanh = [&](const int &bg, const int &ed, const int &lenj, const char *xa, const float *sa,char *xb,float *sb)
	{
		for (register int i = bg; i < ed; i++)
		{
			register int tmp = i * lenj;
			for (register int j = 0; j < lenj; j++)
			{
				float t = xa[tmp + j] * 1.0 / sa[tmp + j];
				if (t > 1) xb[tmp + j] = sb[tmp + j] = 1;
				else if (t < -1) xb[tmp + j] = -1, sb[tmp + j] = 1;
				else xb[tmp + j] = xa[tmp + j], sb[tmp + j] = sa[tmp + j];
			}
		}

	};
	void MYHardTanH(const XTensor &xa, const XTensor &sa, XTensor &xb, XTensor &sb, int corenum)
	{
		std::vector<std::thread> v;
		int bg, ed;
		//memset(c, 0, leni*lenk * sizeof(long long));
		bg = -xa.dimSize[0] / corenum;
		ed = 0;
		for (int i = 0; i < corenum; i++)
		{
			ed += xa.dimSize[0] / corenum;
			bg += xa.dimSize[0] / corenum;
			if (i == corenum - 1) ed += xa.dimSize[0] % corenum;
			v.push_back(std::thread(fun_hardtanh, bg, ed, xa.dimSize[1], (char*)xa.data,(float*)sa.data, (char*)xb.data, (float*)sb.data));
		}
		for (int i = 0; i < corenum; i++)
			v[i].join();
	}
	auto fun_copy = [&](const int &bg, const int &ed, const int &lenj, const float *a,  char *x)
	{
		for (register int i = bg; i < ed; ++i)
		{
			register int tmp = i * lenj;
			register int j = 0;
			for ( j = 0; j < lenj-4; j++)
			{
				 
				x[tmp + j] =(char) a[tmp + j];
				x[tmp + j+1] = (char)a[tmp + j+1];
				x[tmp + j+2] = (char)a[tmp + j+2];
				x[tmp + j+3] = (char)a[tmp + j+3];
			}
			while (j < lenj)
			{
				x[tmp + j] = (char)a[tmp + j];
				++j;
			}
		}

	};
	void Initinput(const XTensor &a,  XTensor &x, XTensor &s,int corenum)
	{
		InitTensor2D(&x, a.dimSize[0], a.dimSize[1],X_INT8);
		InitTensor2D(&s, a.dimSize[0], a.dimSize[1]);
		memcpy(s.data, a.data, a.unitNum * sizeof(float));
		//	for (int i = 0; i < s.unitNum; i++) *((float*)s.data + i) = 1;
		std::vector<std::thread> v;
		int bg, ed;
		bg = -x.dimSize[0] / corenum;
		ed = 0;
		for (int i = 0; i < corenum; ++i)
		{
			ed += x.dimSize[0] / corenum;
			bg += x.dimSize[0] / corenum;
			if (i == corenum - 1) ed += x.dimSize[0] % corenum;
			v.push_back(std::thread(fun_copy, bg, ed, a.dimSize[1], (float*)a.data, (char*)x.data ));
		}
		for (int i = 0; i < corenum; ++i)
			v[i].join();
	}
 
	auto fun_Map_Row = [&](const int &bg, const int &ed, const int &lenj, const float *a, char *x,float *s)
	{
		//std::cout << bg << " " << ed << std::endl;
		for (register int i = bg; i < ed; ++i)
		{
			register int tmp = i * lenj;
			register float maxx = fabs(a[tmp]);
			register int j = 0;
			for (j=0; j < lenj-4;j+=4 )
			{
				maxx = maxf(maxx, fabs(a[tmp + j]));
				maxx = maxf(maxx, fabs(a[tmp + j+1]));
				maxx = maxf(maxx, fabs(a[tmp + j+2]));
				maxx = maxf(maxx, fabs(a[tmp + j+3]));
			}
			while (j < lenj) { maxx = maxf(maxx, fabs(a[tmp + j])); ++j; }
			s[i] = 127.0 / maxx;
			for ( j = 0; j < lenj-4; j+=4)
			{
				x[tmp + j] = (char)round(s[i] * a[tmp + j]);
				x[tmp + j+1] = (char)round(s[i] * a[tmp + j+1]);
				x[tmp + j+2] = (char)round(s[i] * a[tmp + j+2]);
				x[tmp+j+3] = (char)round(s[i]*a[tmp+j+3]);
			}
			while (j < lenj) { x[tmp + j] = (char)round(s[i] * a[tmp + j]); ++j; }
		}

	};
	void InitMap_Row(const XTensor &a, XTensor &x, XTensor &s, int corenum)
	{
		InitTensor2D(&x, a.dimSize[0], a.dimSize[1], X_INT8);
		InitTensor2D(&s, a.dimSize[0], 1);
		std::vector<std::thread> v;
		int bg, ed;
		int leni = a.dimSize[0];
		bg = -leni / corenum;
		ed = 0;
		for (register int i = 0; i < corenum; ++i)
		{
			ed += leni / corenum;
			bg += leni / corenum;
			if (i == corenum - 1) ed += leni % corenum;
			v.push_back(std::thread(fun_Map_Row, bg, ed, a.dimSize[1],  (float*)a.data, (char*)x.data, (float*)s.data));
		}
		for (int i = 0; i < corenum; ++i)
			v[i].join();
	}
	auto fun_Map_1 = [&](const int &bg, const int &ed, const int &lenj, const float *a, char *x,const float &s)
	{
		//std::cout << bg << " " << ed << std::endl;
		for (register int i = bg; i < ed; i++)
		{
			register int tmp = i * lenj;
			for (register int j = 0; j < lenj; j++)
			{
				x[tmp + j] = (char)round(s * a[tmp + j]);
			}
		}

	};
	auto fun_Map_max = [&](const int &bg, const int &ed, const float *a, float  *maxx)
	{
		//std::cout << bg << " " << ed << std::endl;
		register int i = bg;
		for ( ; i < ed-4;++i)
		{
			*maxx = maxf(*maxx, fabs(a[i]));
			*maxx = maxf(*maxx, fabs(a[i+1]));
			*maxx = maxf(*maxx, fabs(a[i+2]));
			*maxx = maxf(*maxx, fabs(a[i+3]));
		}
		while (i < ed)
		{
			*maxx = maxf(*maxx, fabs(a[i])); ++i;
		}

	};
	float fmax[20];
	void InitMap_1(const XTensor &a, XTensor &x, XTensor &s, int corenum)
	{
		InitTensor2D(&x, a.dimSize[0], a.dimSize[1], X_INT8);
		InitTensor2D(&s, 1, 1);
		//float maxx = fabs(*(float*)a.data);
		std::vector<std::thread> v;
		int bg, ed;
		//float* maxx = (float*)malloc(sizeof(float)*corenum);
		 
			int leni = a.unitNum;
		bg = -leni / corenum;
		ed = 0;
		for (int i = 0; i < corenum; ++i)
		{
			ed += leni / corenum;
			bg += leni / corenum;
			if (i == corenum - 1) ed += leni % corenum;
			fmax[i] = *((float*)a.data + i);
			v.push_back(std::thread(fun_Map_max, bg, ed, (float*)a.data,&fmax[i]));
		}
		for (int i = 0; i < corenum; ++i)
			v[i].join();
		for (int i = 1; i < corenum; ++i)
		{
			fmax[0] = maxf(fmax[0], fmax[i]);//fabs(*((float*)a.data+i)));
		}
		*(float*)s.data = 127.0 / fmax[0];
		 
		v.clear();
		 leni = a.dimSize[0];
		bg = -leni / corenum;
		ed = 0;
		for (int i = 0; i < corenum; ++i)
		{
			ed += leni / corenum;
			bg += leni / corenum;
			if (i == corenum - 1) ed += leni % corenum;
			v.push_back(std::thread(fun_Map_1, bg, ed, a.dimSize[1], (float*)a.data, (char*)x.data, *(float*)s.data));
		}
		for (int i = 0; i < corenum; ++i)
			v[i].join();
	}
	auto fun_Map_col = [&](const int &bg, const int &ed, const int &lenj, const float *a, char *x, float *s)
	{
		//std::cout << bg << " " << ed << std::endl;
		for (register int i = bg; i < ed; i++)
		{
			register int tmp = i * lenj;
			for (register int j = 0; j < lenj; j++)
			{
				if(i==0)
			    	s[j] = 127.0 / s[j];
				x[tmp + j] = (char)round(s[j] * a[tmp + j]);
			}
		}

	};
	void InitMap_Col(const XTensor &a, XTensor &x, XTensor &s, int corenum)
	{
		InitTensor2D(&x, a.dimSize[0], a.dimSize[1], X_INT8);
		InitTensor2D(&s, 1, a.dimSize[1]);
		std::vector<std::thread> v;
		for(int i =0;i<a.dimSize[0];i++)
			for (int j = 0; j < a.dimSize[1]; j++)
			{
				if (i == 0)
					*((float*)s.data + j) = fabs(*((float*)a.data + i * a.dimSize[1] + j));
				else
					*((float*)s.data + j) = maxf(*((float*)s.data + j), fabs(*((float*)a.data + i * a.dimSize[1] + j)));

			}
		int bg, ed;
		int leni = a.dimSize[0];
		bg = -leni / corenum;
		ed = 0;
		for (int i = 0; i < corenum; i++)
		{
			ed += leni / corenum;
			bg += leni / corenum;
			if (i == corenum - 1) ed += leni % corenum;
			v.push_back(std::thread(fun_Map_col, bg, ed, a.dimSize[1], (float*)a.data, (char*)x.data, (float*)s.data));
		}
		for (int i = 0; i < corenum; i++)
			v[i].join();
	}
	auto fun_new_add = [&](const int &bg, const int &ed, const int &lenj, const char *a, const float *b, float *c,const float *s,const int flag)
	{
		//std::cout << bg << " " << ed << std::endl;
		for (register int i = bg; i < ed; i++)
		{
			register int tmp = i * lenj;
			for (register int j = 0; j < lenj; j++)
			{
				if (flag == 0)
				{
					c[tmp + j] = a[tmp + j] / s[tmp + j] + b[tmp + j];
				 }
				else c[tmp + j] = a[tmp + j] / s[i] + b[tmp + j];
			}
		}

	};
	XTensor Myadd_int8(const XTensor &a, const XTensor &b, const XTensor &sa, int corenum)
	{
		XTensor c;
		InitTensor2D(&c,a.dimSize[0], b.dimSize[1]);
		float* cp = (float*)c.data;
		char* ap = (char*)a.data;
		float* bp = (float*)b.data;
		float *da = (float*)sa.data;
		int flag = 0;
		if (a.dimSize[0] == sa.dimSize[0] && a.dimSize[1] == sa.dimSize[1]) flag = 0;
		else flag = 1;
		std::vector<std::thread> v;
		int bg, ed;
		int leni = a.dimSize[0];
		bg = -leni / corenum;
		ed = 0;
		for (int i = 0; i < corenum; i++)
		{
			ed += leni / corenum;
			bg += leni / corenum;
			if (i == corenum - 1) ed += leni % corenum;
			v.push_back(std::thread(fun_new_add, bg, ed, a.dimSize[1], ap, bp, cp,da,flag));
		}
		for (int i = 0; i < corenum; i++)
			v[i].join();
		return c;
	}
}
 