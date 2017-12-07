#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>
#include <cuda.h>
#include <cstdio>
#include <ctime>
#define HANDLE_ERROR(x) if((x)!=cudaSuccess){std::cout<<cudaGetErrorString((x))<<std::endl;exit(-1);}

/****************CODE STARTS HERE*******************/

__device__ int satisfies(int i, int j, int *A, int *B)
{
	return (A[i] <= B[j]);
}

__global__ void MergePath(int *A, int *B, int* C, int *x, int *y, int n)
{
	int num_of_threads = blockDim.x;
	int idx = threadIdx.x;
	bool flag = false;
	if (idx == 0)
	{
		x[idx] = 0;
		y[idx] = 0;
		flag = true;
	}
	int A_start = idx*(2 * n) / num_of_threads; //only when len(A)==len(B)
	int B_start = max(0, A_start - (n - 1));
	A_start = min(n - 1, A_start);
	int length_of_array;

	if (B_start == 0)
	{

		length_of_array = A_start + 1;
	}
	else
		length_of_array = n - B_start;

	int left = 0, right = length_of_array - 1;
	// cout<<A_start<<" "<<B_start<<" "<<length_of_array<<endl<<"-------------------------------------------\n";
	
	while (left <= right && !flag)
	{
		// cout<<left<<" "<<right<<endl;
		int mid = left + (right - left) / 2;
		int I = A_start - mid;
		int J = B_start + mid;
		if (!satisfies(I, J, A, B))
		{
			left = mid + 1;
		}
		else
		{
			if (J == 0)
			{
				x[idx] = (I + 1);
				y[idx] = (J);
				flag = true;
			}
			else if (I == n - 1)
			{
				x[idx] = (I + 1);
				y[idx] = (J);
				flag = true;
			}
			else
			{
				if (!satisfies(I + 1, J - 1, A, B))
				{
					x[idx] = (I + 1);
					y[idx] = (J);
					flag = true;
				}
				else
				{
					right = mid;
				}
			}
		}
	}
	left--;
	if (!flag)
	{
		x[idx] = (A_start - left);
		y[idx] = (n);
	}
	__syncthreads();

	int end_x, end_y;
	if (idx == num_of_threads - 1)
	{
		end_x = n;
		end_y = n;
	}
	else
	{
		end_x = x[idx + 1];
		end_y = y[idx + 1];
	}
	int cur_x = x[idx];
	int cur_y = y[idx];
	int put_at = cur_x + cur_y;
	while (cur_x<end_x && cur_y<end_y)
	{
		if (A[cur_x] <= B[cur_y])
		{
			C[put_at++] = A[cur_x++];
		}
		else
		{
			C[put_at++] = B[cur_y++];
		}
	}
	while (cur_x<end_x)
		C[put_at++] = A[cur_x++];
	while (cur_y<end_y)
		C[put_at++] = B[cur_y++];
}
void printArr(int *C,int N)
{
	FILE *g = fopen("g.txt", "w+");
	for (int i = 0; i < N; i++)
	{
		//std::cout << C[i] << " ";
		fprintf(g, "%d ", C[i]);
	}
	//std::cout << std::endl;

}

int main()
{
	clock_t st, en;
	st = clock();
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int n;
	FILE *f = fopen("f.txt", "r");
	if (!f)
	{
		std::cout << "Error\n";
		exit(-1);
	}
	fscanf(f,"%d", &n);
	printf("%d\n",n);
	//std::cin >> n;
	int *C = (int*)malloc(sizeof(int)* 2 * n);
	int *A = (int*)malloc(sizeof(int)*n);
	int *B = (int*)malloc(sizeof(int)*n);
	
	for (int i = 0; i<n; i++)
	{
		//std::cin >> A[i];
		fscanf(f,"%d", A + i);
	}
	for (int i = 0; i<n; i++)
	{
		//std::cin >> B[i];
		fscanf(f,"%d", B + i);
	}
	std::sort(A, A + n);
	std::sort(B, B + n);

	/*
	for (int i = 0; i < n; i++)
		std::cout << A[i] << " ";
	std::cout << std::endl;
	for (int i = 0; i < n; i++)
		std::cout << B[i] << " ";
	*/
	int num_of_threads;
	//std::cin >> num_of_threads;
	fscanf(f, "%d", &num_of_threads);
	printf("%d\n", num_of_threads);
	int *d_x,*d_y,*d_A,*d_B,*d_C;
	HANDLE_ERROR(cudaMalloc((void**)&d_x,sizeof(int)*num_of_threads));
	HANDLE_ERROR(cudaMalloc((void**)&d_y, sizeof(int)*num_of_threads));
	HANDLE_ERROR(cudaMalloc((void**)&d_A, n*sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_B, n*sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_C, 2 * n*sizeof(int)));
	HANDLE_ERROR(cudaDeviceSynchronize());
	HANDLE_ERROR(cudaMemcpy(d_A, A, n*sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_B, B, n*sizeof(int), cudaMemcpyHostToDevice));
	cudaEventRecord(start, 0);

	MergePath << <1, num_of_threads >> >(d_A, d_B, d_C, d_x, d_y, n);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
	// Clean up:
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("%f\n", elapsedTime/1000);
	HANDLE_ERROR(cudaMemcpy(C, d_C, 2 * n*sizeof(int), cudaMemcpyDeviceToHost));
	printArr(C,2*n);
	HANDLE_ERROR(cudaFree(d_x));
	HANDLE_ERROR(cudaFree(d_A));
	HANDLE_ERROR(cudaFree(d_B));
	HANDLE_ERROR(cudaFree(d_C));
	HANDLE_ERROR(cudaFree(d_y));
	
	float elp = (float)(en - st) / CLOCKS_PER_SEC;
	printf("%.10f\n", elp);
	return 0;
}