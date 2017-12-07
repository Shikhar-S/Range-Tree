
#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>
#include <cuda.h>
#include <cstdio>

#define HANDLE_ERROR(x) if((x)!=cudaSuccess){std::cout<<cudaGetErrorString((x))<<std::endl;exit(-1);}
#define MAX_DEPTH 16

struct Point{
	int x; int y;
	Point(int _x, int _y) : x(_x), y(_y) {}
};
struct Query{
	int a, b, c, d;
	Query(int _a, int _b, int _c, int _d) : a(_a), b(_b), c(_c), d(_d) {}
};
void swap(int &a, int &b)
{
	int t = a;
	a = b;
	b = t;
}
int N;
int logN;
int q;
__device__ void selection_sort_x(int left, int right, Point* output)
{
	for (int i = left; i <= right; i++)
	{
		int min_idx = i;
		for (int j = i + 1; j <= right; j++)
		{
			if (output[j].x < output[min_idx].x)
			{
				min_idx = j;
			}
		}
		if (i != min_idx){
			Point t = output[i];
			output[i] = output[min_idx];
			output[min_idx] = t;
		}

	}
}

__global__ void quicksort_x(int left, int right, Point *output, int depth)
{
	if (depth >= MAX_DEPTH)
	{
		selection_sort_x(left, right, output);
		return;
	}
	Point *lptr = output + left;
	Point *rptr = output + right;
	Point pivot = output[left + (right - left) / 2];

	while (lptr <= rptr)
	{
		Point lval = *lptr;
		Point rval = *rptr;
		while (lval.x < pivot.x)
		{
			lptr++;
			lval = *lptr;
		}
		while (rval.x > pivot.x)
		{
			rptr--;
			rval = *rptr;
		}
		if (lptr <= rptr)
		{
			*lptr++ = rval;
			*rptr-- = lval;
		}
	}

	int new_right = rptr - output;
	int new_left = lptr - output;

	if (left < (rptr - output))
	{
		cudaStream_t s;
		cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
		quicksort_x <<<1, 1, 0, s >>>(left, new_right, output, depth + 1);
		cudaStreamDestroy(s);
	}

	if ((lptr - output)<right)
	{
		cudaStream_t s1;
		cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
		quicksort_x <<<1, 1, 0, s1 >>>(new_left, right, output, depth + 1);
		cudaStreamDestroy(s1);
	}
}
__global__ void merge_v3(Point *secondary)
{
	__shared__ Point elements[];
	int g_tid=blockIdx.x*blockDim.x+threadIdx.x;
	elements[threadIdx.x]=secondary[g_tid%blockDim.x];
	__syncthreads();
	
}
void build_secondary_tree(Point *d_secondary)
{
	
	//std::cout << "Calling with " << canonical_size << std::endl;
	/*include logic to find the correct number of threads and blocks to run*/
	int num_of_threads_per_block = N;
	if (num_of_threads_per_block > 1024)
		num_of_threads_per_block = 1024;
	int blocks = N / canonical_size;
	if(blocks>65535)
	{
		printf("Greater than max block limit\n");
		exit(-1);
	}
	int max_cap=1024;
	merge_v3<<<blocks,num_of_threads_per_block,max_cap*sizeof(Point)>>>merge_v3(d_secondary);

	HANDLE_ERROR(cudaDeviceSynchronize());
	build_secondary_tree(d_secondary, canonical_size * 2, call_num + 1);
}

void build_tree(Point *d_primary, Point *d_secondary, Point *points, Point *primary, Point *secondary)
{
	int left = 0;
	int right = N - 1;
	HANDLE_ERROR(cudaMemcpy(d_primary, points, N*sizeof(Point), cudaMemcpyHostToDevice));
	quicksort_x << <1, 1 >> >(left, right, d_primary, 0);
	HANDLE_ERROR(cudaDeviceSynchronize());
	HANDLE_ERROR(cudaMemcpy(d_secondary, d_primary, N*sizeof(Point), cudaMemcpyDeviceToDevice)); //building the first level
	build_secondary_tree(d_secondary);
	HANDLE_ERROR(cudaMemcpy(primary, d_primary, N*sizeof(Point), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(secondary, d_secondary, N*logN*sizeof(Point), cudaMemcpyDeviceToHost));
}
void print_tree(Point* primary, Point* secondary)
{
	std::cout << "-----------------------------------------------------Primary tree-------------------------------------------------------------------\n";
	for (int i = 0; i < N; i++)
	{
		std::cout << primary[i].x << " " << primary[i].y << std::endl;
	}
	std::cout << "-----------------------------------------------------Secondary tree-----------------------------------------------------------------\n";
	for (int i = 0; i < N*logN; i++)
	{
		if (i%N == 0)
		{
			std::cout << "____________________________________________________________________\n";
		}
		std::cout << secondary[i].x << " " << secondary[i].y << std::endl;
		
	}
}
void print_tree_to_file(Point* primary, Point* secondary)
{
	FILE *g = fopen("g.txt","w+");
	fprintf(g, "%d %d\n", N, logN);
	for(int i = 0; i < N; i++)
	{
		fprintf(g, "%d %d\n", primary[i].x, primary[i].y);
	}
	for (int i = 0; i < N*logN; i++)
	{
		fprintf(g, "%d %d\n", secondary[i].x, secondary[i].y);
	}
}
int main()
{
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	FILE *f = fopen("f.txt", "r");
	if (!f)
	{
		std::cout << "ERROR\n";
		exit(-1);
	}
	//std::cin >> N;
	for(int i=0;i<20;i++)
	{
		fscanf(f, "%d", &N);
		logN = 0;
		int cur = 1;
		while (cur < N)
		{
			cur = cur * 2;
			logN++;
		}
		logN++;
		//std::cout << logN << std::endl;
		Point *input = (Point*)malloc(N*sizeof(Point));
		Point *primary = (Point*)malloc(N*sizeof(Point));
		Point *secondary = (Point*)malloc(N*logN*sizeof(Point));

		for (int i = 0; i < N; i++)
		{
			int x, y;
			//std::cin >> x >> y;
			fscanf(f, "%d %d", &x, &y);
			Point t(x, y);
			input[i] = t;
		}

		Point *d_primary;
		Point *d_secondary;

		HANDLE_ERROR(cudaMalloc((void**)&d_primary, N*sizeof(Point)));
		HANDLE_ERROR(cudaMalloc((void**)&d_secondary, N*logN*sizeof(Point)));
		cudaEventRecord(start, 0);
		build_tree(d_primary, d_secondary, input, primary, secondary);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
		std::cout << N << " TOTAL TIME: " << elapsedTime << std::endl;
		//print_tree(primary, secondary);
		print_tree_to_file(primary,secondary);
		HANDLE_ERROR(cudaFree(d_primary));
		HANDLE_ERROR(cudaFree(d_secondary));
		free(primary);
		free(secondary);
	}
	
	
	return 0;
}
