#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <ctype.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>
#include <sys/time.h>

#include "Process.h"
#include "My_Matrix.h"
#include "Process.cpp"
#include "My_Matrix.cpp"



#include "kernels.cu"

#define IMG_DATA_OFFSET_POS 10
#define BITS_PER_PIXEL_POS 28


int main(int argc, char *argv[]) 
{
	int ii;
	int fd, fd1;
	char *fdata;
	char *alldata, *all;
	struct stat finfo;
	char * outputfname;
	int numPic = TRAIN_NUM;
	unsigned short *data_pos;
	int width, height, fileSize;
	char inputfname[8];


	double *T,*T1,*T_test,*m1;
	unsigned char *m;
	double *L, *b,*q,*c,*p_q,*projected_train,*projected_test,*eigenvector,*Euc_dist;  
    unsigned int *L1;
    double eps;
	double temp;  
    int i,j,flag,iteration,num_q;  

      		

	T = (double *)malloc(sizeof(double)*IMG_HEIGHT*IMG_WIDTH*TRAIN_NUM);
    T_test = (double*)malloc(sizeof(double)*IMG_HEIGHT*IMG_WIDTH*1);    
    m = (unsigned char *)malloc(sizeof(unsigned char)*IMG_HEIGHT*IMG_WIDTH);
	m1 = (double *)malloc(sizeof(double)*IMG_HEIGHT*IMG_WIDTH);
    L = (double *)malloc(sizeof(double)*TRAIN_NUM*TRAIN_NUM);   
	L1 = (unsigned int *)malloc(sizeof(unsigned int)*(numPic + 1)*numPic/2);    
    b = (double *)malloc(sizeof(double)*TRAIN_NUM);         
    q = (double *)malloc(sizeof(double)*TRAIN_NUM*TRAIN_NUM);
    c = (double *)malloc(sizeof(double)*TRAIN_NUM);             
	T1 = (double *)malloc(sizeof(double)*IMG_HEIGHT*IMG_WIDTH*TRAIN_NUM);




	if (argc < 2)
	{
		printf("USAGE: %s <picture filename>\n", argv[0]);
		exit(1);
	}

	outputfname = argv[1];

	printf("Match Process: Running...\n");

	for(int fla = 1; fla <= numPic*4; fla ++)
	{
		if(fla <= 9)
		{
			inputfname[0] = (char)(fla +'0');
			inputfname[1] = '.';
			inputfname[2] = 'b';
		    inputfname[3] = 'm';
			inputfname[4] = 'p';
			inputfname[5] = '\0';
		} else if(fla>9 && fla<=19){
			inputfname[0] = '1';
			inputfname[1] = (char)(fla - 10 +'0');
			inputfname[2] = '.';
			inputfname[3] = 'b';
		    inputfname[4] = 'm';
			inputfname[5] = 'p';
			inputfname[6] = '\0';
		} else if(fla>19 && fla<=29){
			inputfname[0] = '2';
			inputfname[1] = (char)(fla - 20 +'0');
		} else if(fla>29 && fla<=39){
			inputfname[0] = '3';
			inputfname[1] = (char)(fla - 30 +'0');
		} else if(fla>39 && fla<=49){
			inputfname[0] = '4';
			inputfname[1] = (char)(fla - 40 +'0');
		} else if(fla>49 && fla<=59){
			inputfname[0] = '5';
			inputfname[1] = (char)(fla - 50 +'0');
		} else if(fla>59 && fla<=69){
			inputfname[0] = '6';
			inputfname[1] = (char)(fla - 60 +'0');
		} else if(fla>69 && fla<=79){
			inputfname[0] = '7';
			inputfname[1] = (char)(fla - 70 +'0');
		} else if(fla>79 && fla<=89){
			inputfname[0] = '8';
			inputfname[1] = (char)(fla - 80 +'0');
		} else if(fla>89 && fla<=99){
			inputfname[0] = '9';
			inputfname[1] = (char)(fla - 90 +'0');
		} else {
			inputfname[0] = '1';
			inputfname[1] = '0';
			inputfname[2] = '0';
			inputfname[3] = '.';
			inputfname[4] = 'b';
			inputfname[5] = 'm';
			inputfname[6] = 'p';
			inputfname[7] = '\0';
		}
		fd = open(inputfname, O_RDONLY);
		fstat(fd, &finfo);


		if(fla == 1)
		{
			alldata = (char*) malloc(numPic * 4* finfo.st_size);
			all = (char*) malloc(numPic * finfo.st_size);
		}

		fdata = (char*) malloc(finfo.st_size);

		read (fd, fdata, finfo.st_size);
	
		data_pos = (unsigned short *)(&(fdata[IMG_DATA_OFFSET_POS]));
	

		width = *((int*)&fdata[18]);
		//printf("Width: %d\n", width);
		height = *((int*)&fdata[22]);
		//printf("Height: %d\n", height);

		fileSize = (int) finfo.st_size;	

		char* p = &(fdata[*data_pos]);

		memcpy(&alldata[0 + (fla - 1)* width * height * 3], p, width * height * 3);

	}
	
	dim3 grid_m(36, 1, 1);
	dim3 block_m(1000, 1, 1);

	dim3 grid(32, 1, 1);
	dim3 block(1024, 1, 1);
	
	unsigned char* d_input;
	cudaMalloc((void**) &d_input, numPic *4* width * height * 3);
    cudaMemcpy(d_input, alldata, numPic *4* width * height * 3, cudaMemcpyHostToDevice);
	
	
	unsigned char* d_output;
	cudaMalloc((void**) &d_output, numPic * width * height *3);
	cudaMemset(d_output, 0, numPic * width * height*3);
	
	struct timeval start_tv, end_tv;
	time_t sec;
	time_t ms;
	time_t diff;
	gettimeofday(&start_tv, NULL);
	
	mean<<<grid_m, block_m >>>((uchar3*) d_input, (uchar3*) d_output, numPic);
	
	cudaThreadSynchronize();
	
	gettimeofday(&end_tv, NULL);
	sec = end_tv.tv_sec - start_tv.tv_sec;
	ms = end_tv.tv_usec - start_tv.tv_usec;

	diff = sec * 1000000 + ms;

	//printf("%10s:\t\t%fms\n", "Time elapsed", (double)((double)diff/1000.0));
	
	cudaMemcpy(all, d_output, numPic * height * width*3, cudaMemcpyDeviceToHost);
	cudaFree(d_input);
	cudaFree(d_output);
	
	
	

	unsigned char* d_inputPixels;
	cudaMalloc((void**) &d_inputPixels, numPic * width * height * 3);
    cudaMemcpy(d_inputPixels, all, numPic * width * height * 3, cudaMemcpyHostToDevice);
	
	
	unsigned char* d_outputPixels;
	cudaMalloc((void**) &d_outputPixels, numPic * width * height );
	cudaMemset(d_outputPixels, 0, numPic * width * height);

	unsigned char* d_m;
	cudaMalloc((void**) &d_m, width * height );
	cudaMemset(d_m, 0, width * height);

	
	unsigned char* outputPixels = (unsigned char*) malloc(numPic * height * width );


	gettimeofday(&start_tv, NULL);


	rgb2gray_mean<<<grid, block ,numPic * 1024 * sizeof(unsigned char)>>>((uchar3*) d_inputPixels, (unsigned char*) d_outputPixels, numPic, (unsigned char*) d_m);
	
	cudaThreadSynchronize();

	gettimeofday(&end_tv, NULL);
	sec = end_tv.tv_sec - start_tv.tv_sec;
	ms = end_tv.tv_usec - start_tv.tv_usec;

	diff = sec * 1000000 + ms;

	//printf("%10s:\t\t%fms\n", "Time elapsed", (double)((double)diff/1000.0));

	cudaMemcpy(outputPixels, d_outputPixels, numPic * height * width, cudaMemcpyDeviceToHost);
	//cudaMemcpy(T, d_outputPixels, numPic * height * width, cudaMemcpyDeviceToHost);
	cudaMemcpy(m, d_m, 36000, cudaMemcpyDeviceToHost);


	memcpy(&(fdata[*data_pos]), &outputPixels[(numPic- 1)*height*width],  height * width);
	

	cudaFree(d_inputPixels);
	cudaFree(d_outputPixels);
	cudaFree(d_m);
	free(alldata);

	cudaDeviceReset();




	eps = 0.000001;  
    memset(L,0,sizeof(double)*TRAIN_NUM*TRAIN_NUM);  

    dim3 grid_L(36,1,1);
	dim3 block_L(1000,1,1);
	unsigned int* d_L;
	cudaMalloc((void**) &d_L, ((numPic + 1)*numPic)/2);
    cudaMemset(d_L, 0, ((numPic + 1)*numPic)/2);

	unsigned int* d_tem;
	cudaMalloc((void**) &d_tem, 36);
    cudaMemset(d_tem, 0, 36);

	//unsigned char* d_outputPixels;
	cudaMalloc((void**) &d_outputPixels, numPic * width * height );
	cudaMemcpy(d_outputPixels, outputPixels, numPic * width * height, cudaMemcpyHostToDevice);


	gettimeofday(&start_tv, NULL);

	calc_covariance_matrix_1<<<grid_L, block_L>>>((unsigned char*) d_outputPixels, numPic, (unsigned int*) d_L, (unsigned int*)d_tem);
	cudaThreadSynchronize();

	gettimeofday(&end_tv, NULL);
	sec = end_tv.tv_sec - start_tv.tv_sec;
	ms = end_tv.tv_usec - start_tv.tv_usec;

	diff = sec * 1000000 + ms;

	//printf("%10s:\t\t%fms\n", "Time elapsed", (double)((double)diff/1000.0));

	cudaMemcpy(L1, d_L, ((numPic + 1)*numPic)/2, cudaMemcpyDeviceToHost);
	cudaFree(d_outputPixels);
	cudaFree(d_L);
	cudaDeviceReset();

	//for(i=0; i<((numPic + 1)*numPic)/2; i++)
	//{
	//	printf("%d:  %d\n", i, (unsigned int)L1[i]);
	//}

	int num = 0;
	for (int i=0;i<numPic; i++)
	{
		for(int j=i; j<numPic; j++)
		{
			//L[i*numPic+j] = L1[num];
			//L[j*numPic+i] = L1[num];
			num++;
		}
	}



	//memcpy(T, outputPixels, numPic * height * width);
	for(int round=0; round < numPic * height * width; round++)
	{
		T[round]=outputPixels[round];
	}


	//memcpy(T_test, &outputPixels[(numPic-1) * height * width], height * width);
	//free(outputPixels);
	//free(fdata);

	//matrix_reverse(T11,T,IMG_HEIGHT*IMG_WIDTH,TRAIN_NUM);


		fd = open("1.bmp", O_RDONLY);
		fstat(fd, &finfo);
		fdata = (char*) malloc(finfo.st_size);
		read (fd, fdata, finfo.st_size);
		//unsigned short *bitsperpixel = (unsigned short *)(&(fdata[BITS_PER_PIXEL_POS]));

		// ensure its 3 bytes per pixel
	
		data_pos = (unsigned short *)(&(fdata[IMG_DATA_OFFSET_POS]));
	
		//int imgdata_bytes = (int)finfo.st_size - (int)(*(data_pos));
		//printf("This file has %d bytes of image data, %d pixels\n", imgdata_bytes, imgdata_bytes / 3);

		width = *((int*)&fdata[18]);
		//printf("Width: %d\n", width);
		height = *((int*)&fdata[22]);
		//printf("Height: %d\n", height);

		fileSize = (int) finfo.st_size;	


	FILE *writeFile; 
	writeFile = fopen("round1.bmp","w+");
	for(ii = 0; ii < fileSize; ii++)
		if(ii<*data_pos)
		{
			fprintf(writeFile,"%c", fdata[ii]);
		} else {
			fprintf(writeFile,"%c", (unsigned char)T[ii - *data_pos]);
		}
	fclose(writeFile);


	//printf("%d %d %d %d %d %d %d %d %d %d \n",(unsigned int)T[0],(unsigned int)T[1],(int)T[36000-2],(int)T[36000-3],(int)T[36000-4], (int)T[36000-5],(int)T[36000-6],(int)T[36000-7],(int)T[36000-8], (int)T[36000-9]);



	//求T矩阵行的平均值  
    //calc_mean(T,m1);  
	//for(int i=0; i<36000;i++){
	//printf("m: %d: %d  %f\n", i, (unsigned int)m[i], m1[i]);
	//}

	for(int round=0; round < height * width; round++)
	{
		m1[round]= m[round];
	}


    calc_covariance_matrix(T,T1,L,m1); 
	free(T1);
	//printf("%qwqwqwqwqw: %d  %d\n", (unsigned int)L[0],(unsigned int)L[1]);

    iteration = 60;  
    cstrq(L,TRAIN_NUM,q,b,c);  
    flag = csstq(TRAIN_NUM,b,c,q,eps,iteration); 

	if (flag<0)  
    {  
        //printf("fucking failed!\n");  
    }else  
    {  
        //printf("success to get eigen value and vector\n");  
    }  

    num_q=0;  
    for (i=0;i<TRAIN_NUM;i++)  
    {  
        if (b[i]>1)  
        {  
            num_q++;  
        }  
    }  

    p_q = (double *)malloc(sizeof(double)*TRAIN_NUM*TRAIN_NUM); 


    projected_train = (double *)malloc(sizeof(double)*TRAIN_NUM*num_q);


    eigenvector = (double *)malloc(sizeof(double)*IMG_HEIGHT*IMG_WIDTH*num_q);


    pick_eignevalue(b,q,p_q,num_q);  
	//for(int round = 1; round <=TRAIN_NUM; round ++)
	//{
	//	printf("Eigenvalue: %e\n", b[round]);
	//}


    get_eigenface(p_q,T,num_q,projected_train,eigenvector); 
	

	fd1 = open(outputfname, O_RDONLY);
	

	fstat(fd1, &finfo);

	char *fdata1;
	fdata1 = (char*) malloc(finfo.st_size);


	read (fd1, fdata1, finfo.st_size);

	data_pos = (unsigned short *)(&(fdata1[IMG_DATA_OFFSET_POS]));

	
	
		width = *((int*)&fdata1[18]);
		//printf("Width: %d\n", width);
		height = *((int*)&fdata1[22]);
		//printf("Height: %d\n", height);

	char* pointer = &(fdata1[*data_pos]);

	unsigned char* d_test;
	cudaMalloc((void**) &d_test,width * height * 3);
	cudaMemcpy(d_test, pointer, width * height * 3, cudaMemcpyHostToDevice);
	

	
	unsigned char* d_outtest;
	cudaMalloc((void**) &d_outtest, width * height );
	cudaMemset(d_outtest, 0, width * height);
	

	unsigned char* outtest = (unsigned char*) malloc(height * width );

	rgb2gray_test<<<grid, block>>>((uchar3*) d_test, (unsigned char*) d_outtest, 1);
	cudaThreadSynchronize();

	cudaMemcpy(outtest, d_outtest, height * width, cudaMemcpyDeviceToHost);


	memcpy(&(fdata1[*data_pos]), outtest,  height * width);
	FILE *writeFile1; 
	writeFile1 = fopen("round2.bmp","w+");
	for(ii = 0; ii < fileSize; ii++)
		fprintf(writeFile1,"%c", fdata1[ii]);
	fclose(writeFile1);

	for(int round=0; round < height * width; round++)
	{
		T_test[round]=outtest[round]-m[round];
	}

	projected_test = (double *)malloc(sizeof(double)*num_q*1);

    memset(projected_test,0,sizeof(double)*num_q);   
	matrix_mutil(projected_test,eigenvector,T_test,num_q,IMG_WIDTH*IMG_HEIGHT,1);  
 
    Euc_dist = (double *)malloc(sizeof(double)*TRAIN_NUM);  
    for (i=0;i<TRAIN_NUM;i++)  
    {  
        temp = 0;  
        for (j=0;j<num_q;j++)  
        {  
            temp = temp + (projected_test[j]-projected_train[j*TRAIN_NUM+i])*(projected_test[j]-projected_train[j*TRAIN_NUM+i]);  
        }  
        Euc_dist[i] = temp;  
		//printf("%d:  %e\n",i,  Euc_dist[i]);
    }  

    double min = Euc_dist[0];  
    int label;  
    for (i=0;i<TRAIN_NUM;i++)  
    {  
        if (min>=Euc_dist[i])  
        {  
            min = Euc_dist[i];  
            label = i;  
        }  
    }  
    printf("No.%d individual is mathcing!\n",label+1);  

	return 0;
} 




