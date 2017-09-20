
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
__global__ void merge_v2(Point* d_secondary,int canonical_size,int call_num,int N,int* x,int* y)
{
	int offset_first = (call_num)*N;
	int offset_block = canonical_size*blockIdx.x; //one block is responsible for one canonical sized node

	int num_of_threads = blockDim.x;
	int idx = threadIdx.x;
	bool flag = false;
	if (idx == 0)
	{
		x[idx + blockDim.x*blockIdx.x] = 0;
		y[idx + blockDim.x*blockIdx.x] = 0;
		flag = true;
	}
	
	int n = canonical_size / 2;
	int A_start = idx*(2 * n) / num_of_threads; //without offsets
	int B_start =  max(0, A_start - (n - 1));
	A_start =min(n - 1, A_start);
	int length_of_array;

	if (B_start == 0)
	{

		length_of_array = A_start + 1;
	}
	else
		length_of_array = n - B_start;

	int left = 0, right = length_of_array - 1;
	
	idx = blockDim.x*blockIdx.x + threadIdx.x;
	while (left <= right && !flag)
	{
		
		int mid = left + (right - left) / 2;
		int I =  A_start - mid;
		int i = offset_first + offset_block + I;
		int J =  B_start + mid;
		int j = offset_first + offset_block + n + J;
		if (d_secondary[i].y > d_secondary[j].y)
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
				if (d_secondary[i + 1].y > d_secondary[j - 1].y)
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
	if (idx-(blockDim.x*blockIdx.x) == num_of_threads - 1)
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
	end_x += offset_first + offset_block;
	end_y += offset_first + offset_block + n;
	cur_x += offset_first + offset_block;
	cur_y += offset_first + offset_block + n;
	put_at += offset_first + offset_block + N;
	while (cur_x<end_x && cur_y<end_y)
	{
		if (d_secondary[cur_x].y <= d_secondary[cur_y].y)
		{
			d_secondary[put_at++] = d_secondary[cur_x++];
		}
		else
		{
			d_secondary[put_at++] = d_secondary[cur_y++];
		}
	}
	while (cur_x<end_x)
		d_secondary[put_at++] = d_secondary[cur_x++];
	while (cur_y<end_y)
		d_secondary[put_at++] = d_secondary[cur_y++];
}
void build_secondary_tree(Point *d_secondary, int canonical_size, int call_num)
{
	if (canonical_size >= 2 * N)
	{
		return;
	}
	//std::cout << "Calling with " << canonical_size << std::endl;
	/*include logic to find the correct number of threads and blocks to run*/
	int num_of_threads_per_block = canonical_size;
	if (num_of_threads_per_block > 1024)
		num_of_threads_per_block = 1024;
	int blocks = N / canonical_size;


	int *d_index_x;
	int *d_index_y;

	HANDLE_ERROR(cudaMalloc((void**)&d_index_x,sizeof(int)*num_of_threads_per_block*blocks));
	HANDLE_ERROR(cudaMalloc((void**)&d_index_y, sizeof(int)*num_of_threads_per_block*blocks));
	HANDLE_ERROR(cudaDeviceSynchronize());

	merge_v2 <<< blocks, num_of_threads_per_block >> > (d_secondary, canonical_size, call_num, N, d_index_x, d_index_y);
	//merge << <1, (N / canonical_size) >> >(d_secondary, canonical_size, call_num, N);
	
	HANDLE_ERROR(cudaFree(d_index_x));
	HANDLE_ERROR(cudaFree(d_index_y));
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
	build_secondary_tree(d_secondary, 2, 0);
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

int main()
{
	FILE *f = fopen("f.txt", "r");
	if (!f)
	{
		std::cout << "ERROR\n";
		exit(-1);
	}
	//std::cin >> N;
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

	build_tree(d_primary, d_secondary, input, primary, secondary);
	print_tree(primary, secondary);

	HANDLE_ERROR(cudaFree(d_primary));
	HANDLE_ERROR(cudaFree(d_secondary));
	free(primary);
	free(secondary);
	
	return 0;
}
