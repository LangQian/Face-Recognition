#include "Process.h"
#include "My_Matrix.h"

void calc_mean(double *T,double *m)
{
	int i,j;
	double temp;

	for (i=0;i<IMG_WIDTH*IMG_HEIGHT;i++)
	{
		temp=0;
		for (j=0;j<TRAIN_NUM;j++)
		{
			temp = temp + T[i*TRAIN_NUM+j];
		}
		m[i] = temp/TRAIN_NUM;
	}
}

void calc_covariance_matrix(double *T,double *T1, double *L,double *m)
{
	int i,j,k;
	//char *T1;
	//printf("1 Done");

	//T = T -m
	for (i=0;i<IMG_WIDTH*IMG_HEIGHT;i++)
	{
		for (j=0;j<TRAIN_NUM;j++)
		{
			//T[i*TRAIN_NUM+j] = T[i*TRAIN_NUM+j] - m[i];
		}
	}
	//printf("2 Done");


	//T1 = (char *)malloc(sizeof(char)*IMG_HEIGHT*IMG_WIDTH*TRAIN_NUM);
	//printf("3 Done");


	//L = T' * T
	matrix_reverse(T,T1,IMG_WIDTH*IMG_HEIGHT,TRAIN_NUM);
	//printf("4 Done");

	matrix_mutil(L,T1,T,TRAIN_NUM,IMG_HEIGHT*IMG_WIDTH,TRAIN_NUM);
	//printf("5 Done");


	//free(T1);
}


void pick_eignevalue(double *b,double *q,double *p_q,int num_q)
{
	int i,j,k;

	k=0;//p_q的列
	for (i=0;i<TRAIN_NUM;i++)//col
	{
		if (b[i]>1)
		{
			for (j=0;j<TRAIN_NUM;j++)//row
			{
				p_q[j*num_q+k] = q[j*TRAIN_NUM+i];//按列访问q,按列存储到p_q

			}
			k++;
		}
	}
}

void get_eigenface(double *p_q,double *T,int num_q,double *projected_train,double *eigenvector)
{
	//double	*temp;
	double tmp;
	int i,j,k;
	//IplImage *projected;
	//unsigned char res[20]={0};	//file name

	//projected = cvCreateImage(cvSize(IMG_WIDTH,IMG_HEIGHT),IPL_DEPTH_8U,1);
	//temp = (double *)malloc(sizeof(double)*IMG_HEIGHT*IMG_WIDTH*num_q);//按列存取

	memset(eigenvector,0,sizeof(double)*IMG_HEIGHT*IMG_WIDTH*num_q);
	memset(projected_train,0,sizeof(double)*TRAIN_NUM*num_q);
	
	//求特征脸
	//matrix_mutil(temp,T,p_q,IMG_WIDTH*IMG_HEIGHT,TRAIN_NUM,num_q);
	
	/*for (i=0;i<num_q;i++)
	{
		sprintf(res,"%d.jpg",i);
		for (j=0;j<IMG_HEIGHT;j++)
		{
			for (k=0;k<IMG_WIDTH;k++)
			{
				projected->imageData[j*IMG_WIDTH+k] = (unsigned char)abs(temp[(j*IMG_WIDTH+k)*num_q+i]);
			}
		}
		cvSaveImage(res,projected);
	}*/

	

	//求Q的特征向量X*e，矩阵相乘
	//matrix_mutil_3(eigenvector,T,p_q,IMG_HEIGHT*IMG_WIDTH,TRAIN_NUM,num_q);
	matrix_mutil(eigenvector,T,p_q,IMG_HEIGHT*IMG_WIDTH,TRAIN_NUM,num_q);


	//投影，计算特征空间变换,Eigenfaces'*A(:,i);
	//matrix_reverse_2(eigenvector,eigenvector,IMG_WIDTH*IMG_HEIGHT,num_q);
	//matrix_mutil_2(projected_train,eigenvector,T,num_q,IMG_WIDTH*IMG_HEIGHT,TRAIN_NUM);

	matrix_reverse(eigenvector,eigenvector,IMG_WIDTH*IMG_HEIGHT,num_q);
	matrix_mutil(projected_train,eigenvector,T,num_q,IMG_WIDTH*IMG_HEIGHT,TRAIN_NUM);

}


