
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

int N;
int logN;

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
		quicksort_x<<<1, 1, 0, s >>>( left, new_right, output, depth + 1);
		cudaStreamDestroy(s);
	}

	if ((lptr - output)<right)
	{
		cudaStream_t s1;
		cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
		quicksort_x<<<1, 1, 0, s1 >>>( new_left, right, output, depth + 1);
		cudaStreamDestroy(s1);
	}
}
__global__ void merge(Point *d_secondary,int sz,int call_num,int N)
{

	int offset_thread = sz*threadIdx.x;
	int offset = (call_num*N);
	int mid = offset + offset_thread + sz / 2;
	int leftptr = offset+offset_thread;
	int rightptr = offset+offset_thread+sz/2;
	int end = offset + offset_thread+sz;
	
	int current_idx = (call_num+1)*N + offset_thread;
	printf("for thread %d offset is %d mid is %d leftptr is %drightptr is %d end is %dcurrent_idx is  %d\n", threadIdx.x, offset, mid, leftptr, rightptr, end, current_idx);
	//std::cout << "for thread " << threadIdx.x << "offset is " << offset << " mid is " << mid << " leftptr is " << leftptr << "rightptr is " << rightptr << "end is " << end << "current_idx is " << current_idx << std::endl;
	while (leftptr < mid && rightptr<end)
	{
		
		if (d_secondary[leftptr].y < d_secondary[rightptr].y)
		{
			d_secondary[current_idx++] = d_secondary[leftptr++];
		}
		else
		{
			d_secondary[current_idx++] = d_secondary[rightptr++];
		}
	}
	while (leftptr<mid)
	{
		d_secondary[current_idx++] = d_secondary[leftptr++];
	}
	while (rightptr<end)
	{
		d_secondary[current_idx++] = d_secondary[rightptr++];
	}
}

void build_secondary_tree(Point *d_secondary,int canonical_size,int call_num)
{
	if (canonical_size >= 2*N)
	{
		return;
	}
	std::cout << "Calling with " << canonical_size<<std::endl;
	merge <<<1, (N / canonical_size) >>>(d_secondary,canonical_size,call_num,N);
	HANDLE_ERROR(cudaDeviceSynchronize());
	build_secondary_tree(d_secondary, canonical_size * 2,call_num+1);
}

void build_tree(Point *d_primary, Point *d_secondary, Point *points, Point *primary, Point *secondary)
{
	int left = 0;
	int right = N - 1;
	HANDLE_ERROR(cudaMemcpy(d_primary, points, N*sizeof(Point), cudaMemcpyHostToDevice));
	quicksort_x << <1, 1 >> >(left, right, d_primary, 0);
	HANDLE_ERROR(cudaDeviceSynchronize());
	HANDLE_ERROR(cudaMemcpy(d_secondary,d_primary,N*sizeof(Point),cudaMemcpyDeviceToDevice)); //building the first level
	build_secondary_tree(d_secondary,2,0);
	HANDLE_ERROR(cudaMemcpy(primary, d_primary, N*sizeof(Point), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(secondary, d_secondary, N*logN*sizeof(Point), cudaMemcpyDeviceToHost));
}
void print_tree(Point* primary, Point* secondary)
{
	std::cout << "---------------------------------------------------------Primary tree----------------------------------------------------------\n";
	for (int i = 0; i < N; i++)
	{
		std::cout << primary[i].x << " " << primary[i].y << std::endl;
	}
	std::cout << "----------------------------------------------------------Secondary tree--------------------------------------------------------\n";
	for (int i = 0; i < N*logN; i++)
	{
		std::cout << secondary[i].x << " " << secondary[i].y << std::endl;
	}
}

int main()
{
	std::cin >> N;
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
		std::cin >> x >> y;
		Point t(x, y);
		input[i] = t;
	}

	Point *d_primary;
	Point *d_secondary;

	HANDLE_ERROR(cudaMalloc((void**)&d_primary, N*sizeof(Point)));
	HANDLE_ERROR(cudaMalloc((void**)&d_secondary, N*logN*sizeof(Point)));

	build_tree(d_primary, d_secondary, input, primary, secondary);
	print_tree(primary, secondary);
	
	getchar();


}


