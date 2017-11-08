#ifndef _IMAGEFILTER_KERNEL_H_
#define _IMAGEFILTER_KERNEL_H_

__global__ void rgb2gray_mean(uchar3* inputPixels, unsigned char* outputPixels, int numPic, unsigned char* m)
{

	int width = 180, height = 200;
	int numGrid = 1024 *32;
	int flag = (width * height*numPic - 1) / numGrid + 1;
	
	int flag2 = (width*height-1)/numGrid + 1;

	int location = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ uchar3 in[1024];
	extern __shared__ unsigned char row[];

	for(int k = 0; k < flag; k ++)
	{
		int cu_location = location + k * numGrid;
		if (cu_location < width * height*numPic){
			in[threadIdx.x] = inputPixels[cu_location];
			unsigned int sum = 0;
			//sum = ((int) in[threadIdx.x].x*299 + (int) in[threadIdx.x].y*587 + (int) in[threadIdx.x].z*114 + 500) / 1000;
			sum = ((unsigned int) in[threadIdx.x].x*2989 + (unsigned int) in[threadIdx.x].y*5870 + (unsigned int) in[threadIdx.x].z*1140) / 10000;

								
			//outputPixels[cu_location] = sum;
			outputPixels[(cu_location % 36000) * numPic + (cu_location / 36000)] = sum;

			//__syncthreads();
		}
		
	}
	__syncthreads();

	for(int k = 0; k < flag2; k ++)
	{
		//int cu_location = (k*32+blockIdx.x)*1024*numPic+threadIdx.x;
		//if ((k*32+blockIdx.x)*1024+threadIdx.x < width * height){
		//	double mean = 0.0;
		//	for (int i = 0; i < numPic; i++)
		//	{
		//		row[threadIdx.x + 1024*i] = outputPixels[cu_location + 1024*i];
		//	}
		//	__syncthreads();
	//
	//		for(int j=0; j<numPic; j++)
	//		{
	//			mean = mean + row[threadIdx.x * numPic + j];
	//		}
	//		__syncthreads();
	//		mean = mean/numPic;
	//		for(int j=0; j<numPic; j++)
	//		{
	//			row[threadIdx.x * numPic + j] = row[threadIdx.x * numPic + j] - mean;
	//		}
	//		__syncthreads();
//
//			for (int i = 0; i < numPic; i++)
//			{
//				outputPixels[cu_location + 1024*i] = row[threadIdx.x + 1024*i];
//			}
//			__syncthreads();
//
//			m[location + k * numGrid] = mean;
//		}
		
		int cu_location = (k*32+blockIdx.x)*1024+threadIdx.x;
		double mean = 0;
		if(cu_location < 36000)
		{
			for(int i = 0; i<numPic; i++)
			{
				mean = mean + outputPixels[numPic * cu_location +i];
			}
			mean = mean/numPic;
			for(int i = 0; i<numPic; i++)
			{
				outputPixels[numPic * cu_location +i] = outputPixels[numPic * cu_location +i] - mean;
			}
			m[cu_location] = mean;

		}		
	}

}

__global__ void rgb2gray_test(uchar3* inputPixels, unsigned char* outputPixels, int numPic)
{

	int width = 180, height = 200;
	int numGrid = 1024 *32;
	int flag = (width * height*numPic - 1) / numGrid + 1;
	int location = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ uchar3 in[1024];
	for(int k = 0; k < flag; k ++)
	{
		int cu_location = location + k * numGrid;
		if (cu_location < width * height*numPic){
			in[threadIdx.x] = inputPixels[cu_location];
		    unsigned int sum = 0;
			//sum = ((int) in[threadIdx.x].x*299 + (int) in[threadIdx.x].y*587 + (int) in[threadIdx.x].z*114 + 500) / 1000;
			sum = ((unsigned int) in[threadIdx.x].x*2989 + (unsigned int) in[threadIdx.x].y*5870 + (unsigned int) in[threadIdx.x].z*1140) / 10000;

								
			outputPixels[cu_location] = sum;
			//outputPixels[(cu_location % 36000) * numPic + (cu_location / 36000)] = sum;

			//__syncthreads();
		}
		
	}


}


__global__ void calc_covariance_matrix(unsigned char* T, int numPic, unsigned int* L, unsigned int* tem)
{
	__shared__ unsigned char a[3000];
	__shared__ unsigned int b[1000];
	int location = blockDim.x * blockIdx.x + threadIdx.x;
	int num = 0;

	for(int i=0; i<numPic; i++)
	{
		for(int m=0;m<3;m++)
		{
			a[threadIdx.x+m*1000] = T[location + i * 36000 + m*12000];
		}
		__syncthreads();
		
		for(int j=i; j<numPic; j++)
		{
			for(int m=0; m<3; m++)
			{
			if(j==i)
			{
				b[threadIdx.x] = a[threadIdx.x+m*1000] * a[threadIdx.x+m*1000];
			} else {
				b[threadIdx.x] = a[threadIdx.x+m*1000] * T[location + j * 36000+m*12000];
			}
			__syncthreads();
			for(int k=500; k>0; k=k/2)
			{
				if(threadIdx.x<k)
				{
					b[threadIdx.x] = b[threadIdx.x] + b[threadIdx.x + k];
				}
				if ((k%2==1) && (k!=1))
				{
					if (threadIdx.x == 0)
					{
						b[0] = b[0] + b[k-1];
					}
				}
				__syncthreads();
			}
			if(threadIdx.x == 0)
			{
				tem[blockIdx.x+m*12] = b[threadIdx.x];
			}
			}
			__syncthreads();
			
			for(int l = 0; l<36; l++)
			{
				if (location == 0)
				{
					L[num] = L[num] + tem[l];
				}
			}
			num++;
		}
	}
}

__global__ void calc_covariance_matrix_1(unsigned char* T, int numPic, unsigned int* L, unsigned int* tem)
{
	__shared__ unsigned int a[1000];
	int location = blockDim.x * blockIdx.x + threadIdx.x;
	int num = 0;

	for(int i=0; i<numPic; i++)
	{
		for(int j=i; j<numPic; j++)
		{
			a[threadIdx.x] = (unsigned int)T[location + i * 36000] * (unsigned int)T[location + j * 36000];
			__syncthreads();

			for(int k=500; k>0; k=k/2)
			{
				if(threadIdx.x<k)
				{
					a[threadIdx.x] = a[threadIdx.x] + a[threadIdx.x + k];
				}
				__syncthreads();
				if ((k%2==1) && (k!=1))
				{
					if (threadIdx.x == 0)
					{
						a[0] = a[0] + a[k-1];
					}
				}
				__syncthreads();
			}

			if(threadIdx.x == 0)
			{
				tem[blockIdx.x] = a[0];
			}
			__syncthreads();

			
			for(int l = 0; l<36; l++)
			{
				if (location == 0)
				{
					L[num] = L[num] + tem[l];
				}
			}
			num++;
		}
	}
	
}

__global__ void mean(uchar3* inputPixels, uchar3* outputPixels, int numPic)
{	
	int location = blockDim.x * blockIdx.x + threadIdx.x;
	//__shared__ uchar3 a[1000];
	for(int i=0; i<numPic; i++)
	{
		outputPixels[location + i*36000].x = (inputPixels[location + i*4*36000].x + inputPixels[location + i*4*36000 + 36000].x + inputPixels[location + i*4*36000 + 2*36000].x + inputPixels[location + i*4*36000 + 3*36000].x)/4;
		outputPixels[location + i*36000].y = (inputPixels[location + i*4*36000].y + inputPixels[location + i*4*36000 + 36000].y + inputPixels[location + i*4*36000 + 2*36000].y + inputPixels[location + i*4*36000 + 3*36000].y)/4;
		outputPixels[location + i*36000].z = (inputPixels[location + i*4*36000].z + inputPixels[location + i*4*36000 + 36000].z + inputPixels[location + i*4*36000 + 2*36000].z + inputPixels[location + i*4*36000 + 3*36000].z)/4;
	}
}



#endif // _IMAGEFILTER_KERNEL_H_
