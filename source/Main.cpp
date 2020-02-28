/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northestern University.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 /*
  * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-10
  */

#include <stdio.h>
#include "./network/XNet.h"
#include "./tensor/XUtility.h"
#include "./tensor/function/FHeader.h"
#include "./tensor/core/CHeader.h"
#include "./tensor/test/Test.h"
#include "./sample/fnnlm/FNNLM.h"
#include "./sample/transformer/Transformer.h"
#include "INT8.h"
#include<iostream>
//#include<time.h>
  //#define CRTDBG_MAP_ALLOC
  //#include <stdlib.h>
  //#include <crtdbg.h>

void BackwardTest();
void TransposeTest();
void SumDimTest();

using namespace nts;
using namespace fnnlm;
using namespace transformer;
using namespace std;
float  dda[200 * 30000 + 5];
float ddb[200 * 30000 + 5];
float ddc[200 * 30000 + 5];
float ddd[200 * 30000 + 5];
void rd(float da[], int x, int flag)
{
	if (flag) {
		for (int i = 0; i < x; i++)
		{
			da[i] = rand() % 2;
		}
		return;
	}
	for (int i = 0; i < x; i++)
	{
		float tmp = rand();
		if (tmp == 0) tmp = 1.0;
		 da[i] = (rand()*2.0 - RAND_MAX) / tmp;
		 da[i] = rand() % 10+1 ;
		  
	}

}
int main(int argc, const char ** argv)
{
/*	  srand(time(0));
	 XTensor a, b, c, d,sa,sb,sc,sd,xa,xb,xc,xd;
	InitTensor2D(&a, 2, 3); InitTensor2D(&sa, 2, 2); InitTensor2D(&xa, 2, 2,X_INT8);
	InitTensor2D(&b, 3, 1); InitTensor2D(&sb, 3, 3); InitTensor2D(&xb, 3, 3,X_INT8);
	InitTensor2D(&c, 3, 3); InitTensor2D(&sc, 3, 3); InitTensor2D(&xc, 2, 3,X_INT8);
	InitTensor2D(&d, 3, 3); InitTensor2D(&sd, 3, 3); InitTensor2D(&xd,2, 2,X_INT8);
	rd(dda, a.unitNum, 0);
	for (int i = 0; i < 6; i++) cin >>dda[i];
	a.SetData(dda, a.unitNum);
	rd(ddb, b.unitNum, 0); b.SetData(ddb, b.unitNum);
	InitTensor2D(&xc, a.dimSize[0], b.dimSize[1], X_INT8);
	 
	a.Dump(stderr); b.Dump(stderr);
	InitMap_Row(a, xa,sa, min(a.dimSize[0], 8));
	cout << "xa ";
	for (int i = 0; i < xa.unitNum; i++) cout << (int)*((char*)xa.data + i) << " ";
	cout << "\n";
	cout << "sa "; sa.Dump(stderr);
	InitMap_Col(a, xa, sa, min(a.dimSize[0], 8));
	cout << "xa ";
	for (int i = 0; i < xa.unitNum; i++) cout << (int)*((char*)xa.data + i) << " ";
	cout << "\n";
	cout << "sa "; sa.Dump(stderr);
	InitMap_1(a, xa, sa, min(a.dimSize[0], 8));
	cout << "xa ";
	for (int i = 0; i < xa.unitNum; i++) cout << (int)*((char*)xa.data + i) << " ";
	cout << "\n";
	cout << "sa "; sa.Dump(stderr);
	InitMaptensor(a, xa, sa, min(a.dimSize[0], 8));
	cout << "xa ";
	for (int i = 0; i < xa.unitNum; i++) cout << (int)*((char*)xa.data + i) << " ";
	cout << "\n";
	cout << "sa "; sa.Dump(stderr);
	InitMap_Row(b, xb, sb, 8); //xw 16400*100 sw16400*1
	cout << "xb ";
	for (int i = 0; i < xb.unitNum; i++) cout << (int)*((char*)xb.data + i) << " ";
	cout << "\n";
	cout << "sb ";
	for (int i = 0; i < sb.unitNum; i++) cout << *((float*)sb.data + i) << " ";
	cout << "\n";
	InitTensor2D(&sc, sa.dimSize[0],sb.dimSize[1]);
	//cout << "ok\n";
	Mymul_int8(xa, xb, xc, sa, sb, sc, 1);
	cout << "xc ";
	for (int i = 0; i < xc.unitNum; i++) cout << (int)*((char*)xc.data + i) << " ";
	cout << "\n";
	cout << "sc "; sc.Dump(stderr);
	UnMapping(c, xc, sc, 8);
	cout << "c "; c.Dump(stderr);
	MatrixMul(a, X_NOTRANS, b, X_NOTRANS, c);
	cout << "c "; c.Dump(stderr);
	getchar(); getchar();
	/*
	Mapping(c, xc, sc, 8); Mapping(d, xd, sd, 8);
	TensorList eList(2);
	XTensor ha;
	eList.Add(&xa); eList.Add(&xb);
	InitTensor2D(&ha, 8, 2 *1, X_INT8);
	Concatenate(eList, ha, 1);
	for (int i = 0; i < xc.unitNum; i++) cout << (int) *((char*)xc.data + i) << " ";
	cout << endl;
	for (int i = 0; i < xd.unitNum; i++) cout << (int) *((char*)xd.data + i) << " ";
	cout << endl;
	for (int i = 0; i < ha.unitNum; i++) cout << (int) *((char*)ha.data + i) << " ";
	cout << endl;
	 double st = clock();
	MatrixMul(a, X_NOTRANS, b, X_NOTRANS, d);
	cout << clock() - st << endl;
	st = clock();
	Mapping(a, xa, sa,8);
	//cout << clock() - st << endl;
	Mapping(b, xb, sb,8);
	//cout << clock() - st << endl;
	Mymul_int8(xa, xb, xc, Match(xa, sa, 0,8), Match(xb, sb, 1,8), sc, 8);
	//cout << clock() - st << endl;
	UnMapping(c, xc, sc,8);
	cout << clock() - st << endl;
	for (int i = 0; i < d.unitNum; i++)
	{
		cout << *((float*)d.data + i) << " "<<*((float*)c.data + i) << "\n";
		if (i % 10 == 0) getchar();
	}
 
		/*  映射   */
		/* 
		for (int i = 0; i < 100; i++) {
			rd(dda, a.unitNum, 0); a.SetData(dda, a.unitNum);
		 
			Mapping(a, xa, sa);
			float *da = (float*)a.data;
			float *db = (float*)b.data;
			//a.Dump(stderr);
		//	sa.Dump(stderr);
		//	xa.Dump(stderr);
			UnMapping(b,xa, sa);
		//	b.Dump(stderr);
			int j = 0;
			for ( j = 0; j < a.unitNum; j++)
			{
				if (fabs(da[j] - db[j])>1e-2)
				{
					cout << i << " err" << endl;
				//	a.Dump(stderr);
				//	b.Dump(stderr);
					cout << da[j] << " " << db[j] << " "<<fabs(da[j] - db[j]) << endl;
				//	getchar();
				 	break;
				}
			}
			//if(j==a.unitNum)
		//	cout << i<<" ok" << endl;
			 
		}*/
		//cout << "13" << endl;

		/*  乘法 */
		/*for (int i = 0; i < 100; i++) {
			rd(dda, a.unitNum, 0); a.SetData(dda, a.unitNum);
			rd(ddb, b.unitNum, 0); b.SetData(ddb, b.unitNum);
			Mapping(a, xa, sa);
			Mapping(b, xb, sb);
			float *dc = (float*)c.data;
			float *dd = (float*)d.data;
			double st = clock();
		 	Mymul_int8(xa, xb, xc, Match(xa, sa, 0), Match(xb, sb, 1), sc, 8);
		//	cout << clock() - st << endl;
		//st = clock();
			MatrixMul(a, X_NOTRANS, b, X_NOTRANS, d);
		//	cout << clock() - st << endl;
			UnMapping(c,xc, sc);
			int j = 0;
			for (j = 0; j < c.unitNum; j++)
			{
				if (fabs(dc[j] - dd[j]) > 1e-2)
				{
					cout << i << " err" << endl;
					 	//c.Dump(stderr);
					 	//d.Dump(stderr);
					 
				  cout <<"加速: "<< dc[j] << "  实际: " << dd[j] << "\n"  ;
					 //	Mymul_c(a, b, c, 8);
					 //	cout << dc[j] << endl;
					 	getchar();
					break;
				}
			}
			//getchar();
		}*/
		/*  加法 */
	/*for (int i = 0; i < 100; i++) {
		rd(dda, a.unitNum, 0); a.SetData(dda, a.unitNum);
		rd(ddb, b.unitNum, 0); b.SetData(ddb, b.unitNum);
		//a.Dump(stderr);
	//	b.Dump(stderr);
		Mapping(a, xa, sa);
		Mapping(b, xb, sb);
		float *dc = (float*)c.data;
		 
		double st = clock();
		Match(xa, sa, xb, sb);
	 
		Myadd_int8(xa, xb, xc, sa, sb, sc);
		d = a + b;
		float *dd = (float*)d.data;
		UnMapping(c, xc, sc);
		for (int j = 0; j < c.unitNum; j++)
		{
			if (fabs(dc[j] - dd[j]) > 1e-2)
			{
				cout << i << " err" << endl;
				cout << "加速: " << dc[j] << "  实际: " << dd[j] << "\n";
				getchar();
				break;
			}
		}
	}*/
	/*并行map*/
	/* 
	for (int i = 0; i < 100; i++) {
		rd(dda, a.unitNum, 0); a.SetData(dda, a.unitNum); b.SetData(dda, a.unitNum);
		Mapping(a, xa, sa);
		Mapping(b, xb, sb,8);
		float *da = (float*)a.data;
		float *db = (float*)b.data;
		//a.Dump(stderr);
	//	sa.Dump(stderr);
	//	xa.Dump(stderr);
		UnMapping(a, xa, sa);
		UnMapping(b, xb, sb);
		//	b.Dump(stderr);
		int j = 0;
		for (j = 0; j < a.unitNum; j++)
		{
			if (fabs(da[j] - db[j]) > 1e-2)
			{
				cout << i << " err" << endl;
				//	a.Dump(stderr);
				//	b.Dump(stderr);
				cout << da[j] << " " << db[j] << " " << fabs(da[j] - db[j]) << endl;
				//	getchar();
				break;
			}
		}
		//if(j==a.unitNum)
	 cout << i<<" ok" << endl;

	}
	*/
	/* 并行matchadd*/
/* 
for (int i = 0; i < 100; i++) {
	rd(dda, a.unitNum, 0); a.SetData(dda, a.unitNum); //c.SetData(dda, a.unitNum);
	rd(ddb, a.unitNum, 0); b.SetData(ddb, b.unitNum);//d.SetData(ddb, b.unitNum);
	Mapping(a, xa, sa); Mapping(a, xc, sc);
	Mapping(b, xb, sb, 8); Mapping(b, xd, sd, 8);
	Match(xa, sa, xb, sb);
	Match(xc, sc, xd, sd,8);
	UnMapping(a, xa, sa);
	UnMapping(c, xc, sc);
	float *da = (float*)a.data;
	float *db = (float*)b.data;
	float *dc = (float*)c.data;
	float *dd= (float*)d.data;
	//a.Dump(stderr);
//	sa.Dump(stderr);
//	xa.Dump(stderr);
	 
	//	b.Dump(stderr);
	int j = 0;
	for (j = 0; j < a.unitNum; j++)
	{
		if (fabs(da[j] - dc[j]) > 1e-2)
		{
			cout << i << " err" << endl;
			//	a.Dump(stderr);
			//	b.Dump(stderr);
			cout << da[j] << " " << dc[j] << " " << fabs(da[j] - dc[j]) << endl;
			//	getchar();
			break;
		}
	}
	//if(j==a.unitNum)
	cout << i << " ok" << endl;

}
*/
/* 并行match乘法*/
/*
		rd(dda, a.unitNum, 0); a.SetData(dda, a.unitNum);
		rd(ddb, b.unitNum, 0); b.SetData(ddb, b.unitNum);
		double st = clock();	 
		Mapping(a, xa, sa,8);	 
		Mapping(b, xb, sb, 8); 
for (int i = 0; i < 1; i++) {
 
	float *dc = (float*)c.data;
	float *dd = (float*)d.data;
	double st = clock();
	XTensor t1 = Match(xa, sa, 0);
	cout << "单线程 match a :" << clock() - st <<"ms"<<endl;
	st = clock();
	t1 = Match(xa, sa, 0,8);
	cout << "8线程 match a :" << clock() - st << "ms" << endl;
	st = clock();
	XTensor t2 = Match(xb, sb, 1);
	cout << "单线程 match b :" << clock() - st << "ms" << endl;
	st = clock();
	t2 = Match(xb, sb, 1,8);
	cout << "多线程 match b :" << clock() - st << "ms" << endl;
	st = clock();
	Mymul_int8(xa, xb, xc, t1, t2, sc, 8);
	 cout << "int8乘法: "<<clock() - st << "ms" << endl;
     st = clock();
	 MatrixMul(a, X_NOTRANS, b, X_NOTRANS, d);
	 cout << "自带乘法: "<< clock() - st << "ms" << endl;
	 st = clock();
	 UnMapping(c, xc, sc);
	cout << "反映射: "<<clock() - st << "ms" << endl;
}
	*/
//std::cout << "全部替换结果" << endl;
 
 if (argc > 1 && !strcmp(argv[1], "-test"))
		Test();
	else if (argc > 1 && !strcmp(argv[1], "-fnnlm"))
		FNNLMMain(argc - 1, argv + 1);
	else if (argc > 1 && !strcmp(argv[1], "-t2t"))
		TransformerMain(argc - 1, argv + 1);
	else     {
		 
		XTensor a, b, c, d;
		int tt = 1;
		std::freopen("out2.txt", "w", stdout);
		while (tt--) {
			int rrd = 7;
			InitTensor2D(&a, rrd, 16400);
			InitTensor2D(&b, 16400, 200);
			InitTensor2D(&c, rrd, 200);
			InitTensor2D(&d, 20, 20);
			rd(dda, a.unitNum, 0); a.SetData(dda, a.unitNum);
			rd(ddb, b.unitNum, 0); b.SetData(ddb, b.unitNum);
			rd(ddc, c.unitNum, 1); c.SetData(ddc, c.unitNum);
			std::cout << rrd << endl;
			std::cout << "自带矩阵乘法" << endl;
			double startT = GetClockSec();
			for (int i = 1; i <= 10; i++)
			{
				//cout << i << endl;
				rd(dda, a.unitNum, 0); a.SetData(dda, a.unitNum);
				rd(ddb, b.unitNum, 0); b.SetData(ddb, b.unitNum);
				rd(ddc, c.unitNum, 1); c.SetData(ddc, c.unitNum);
				MatrixMul(a, X_NOTRANS, b, X_NOTRANS, c);
			}
			double elapsed = GetClockSec();
			//startT = GetClockSec();- startT;
			std::cout << elapsed - startT << " ok" << endl;
			for (int x = 1; x <= 12; x++) {
				std::cout << endl << x << "线程整型加速乘法 " << endl;
				rd(dda, a.unitNum, 0); a.SetData(dda, a.unitNum);
				rd(ddb, b.unitNum, 0); b.SetData(ddb, b.unitNum);
				rd(ddc, c.unitNum, 1); c.SetData(ddc, c.unitNum);
				startT = GetClockSec();
				for (int ii = 1; ii <= 1; ii++)
				{
				 
					MYMatrixMul(a, b, c, x);
					
				}
				std::cout << GetClockSec() - startT << " ok" << endl;
			 
				std::cout << endl << x << "线程int8整型加速乘法" << endl;
				startT = GetClockSec();

				for (int ii = 1; ii <= 1; ii++)
				{
				 
					Mymul_c(a, b, c,x);
					
				}
				std::cout << GetClockSec() - startT << " ok" << endl;
			 
			}
		}
		 
	//_CrtDumpMemoryLeaks();
	 
	}  
	return 0;
}

void BackwardTest()
{
	XNet net;

	XTensor a;
	XTensor b;
	XTensor c;
	a.enableGrad = true;
	b.enableGrad = false;
	c.enableGrad = false;
	XTensor mean;
	XTensor origin;
	InitTensor2DV2(&a, 2, 3);
	InitTensor1DV2(&b, 2);

	a.SetZeroAll();
	b.SetZeroAll();
	a.Set2D(1.0F, 0, 0);
	a.Set2D(2.0F, 0, 1);
	a.Set2D(3.0F, 0, 2);
	a.Set2D(4.0F, 1, 0);
	a.Set2D(5.0F, 1, 1);
	a.Set2D(6.0F, 1, 2);

	b.Set1D(2.0F, 0);
	b.Set1D(1.0F, 1);

	DivDim(a, b, c, 0);
	c.Dump(stderr, "c:");
	auto loss = CrossEntropy(c, a);

	//XLink::ShowNetwork(stderr, &c);

	net.Backward(loss);

	a.grad->Dump(stderr);

}

void TransposeTest()
{
#ifdef USE_CUDA
	XMem mem0(0, UNI_FREE, MILLION * 64, 1024, MILLION * 64);
	//XMem mem1(1, UNI_FREE, MILLION * 64, 1024, MILLION * 64);
	XTensor x;
	XTensor y;
	XTensor z;

	int loops = 2000;

	int B = 3 * 2 * 4;
	int K = 8 * 1;
	int N = 50;
	int H = 512 * 4;

	int nnn = GDevs.nGPU;

	InitTensor3DV2(&x, B, N, H, X_FLOAT, 0);
	InitTensor4DV2(&y, K, B, N, H / K, X_FLOAT, 0);
	InitTensor3DV2(&z, B, N, H, X_FLOAT, 0);

	cudaEvent_t ctime0;
	cudaEvent_t ctime1;
	cudaEvent_t ctime2;
	cudaEvent_t ctime3;
	cudaEvent_t ctime4;
	cudaEvent_t ctime5;

	float elapsedSplit = 0.0;
	float elapsedMerge = 0.0;
	float elapsedSum = 0.0;

	cudaEventCreate(&ctime0);
	cudaEventCreate(&ctime1);
	cudaEventCreate(&ctime2);
	cudaEventCreate(&ctime3);
	cudaEventCreate(&ctime4);
	cudaEventCreate(&ctime5);

	cudaEventRecord(ctime0, 0);

	double time0 = GetClock();
	for (int i = 0; i < loops; i++)
		_Split(&x, &y, 2, K);
	double time1 = GetClock();

	cudaEventRecord(ctime1, 0);
	cudaEventSynchronize(ctime1);
	cudaEventElapsedTime(&elapsedSplit, ctime0, ctime1);

	cudaEventRecord(ctime2, 0);

	double time2 = GetClock();
	for (int i = 0; i < loops; i++)
		_Merge(&y, &x, 3);
	double time3 = GetClock();

	cudaEventRecord(ctime3, 0);
	cudaEventSynchronize(ctime3);
	cudaEventElapsedTime(&elapsedMerge, ctime2, ctime3);

	cudaEventRecord(ctime4, 0);

	double time4 = GetClock();
	for (int i = 0; i < loops; i++)
		_Sum(&x, &z, &x);
	double time5 = GetClock();

	cudaEventRecord(ctime5, 0);
	cudaEventSynchronize(ctime5);
	cudaEventElapsedTime(&elapsedSum, ctime4, ctime5);

	fprintf(stderr, "split:%f merge:%f sum:%f\n", time1 - time0, time3 - time2, time5 - time4);
	fprintf(stderr, "split:%f merge:%f sum:%f\n", elapsedSplit, elapsedMerge, elapsedSum);
#endif
}

void SumDimTest()
{
	XTensor x;
	XTensor y;
	XTensor z;

	int a = 5;
	int b = 7;
	int c = 3;

	InitTensor3DV2(&x, a, b, c, X_FLOAT, -1);
	InitTensor1DV2(&y, c, X_FLOAT, -1);
	InitTensor3DV2(&z, a, b, c, X_FLOAT, -1);

	x.SetZeroAll();
	y.SetZeroAll();
	z.SetZeroAll();

	DTYPE * data = new DTYPE[x.unitNum];

	for (int i = 0; i < x.unitNum; i++)
		data[i] = (DTYPE)i;
	x.SetData(data, x.unitNum);

	for (int i = 0; i < y.unitNum; i++)
		data[i] = -(DTYPE)i;
	y.SetData(data, y.unitNum);

	_SumDim(&x, &y, &z, 2);

	z.Dump(stderr, "z:");

	delete[] data;
}
