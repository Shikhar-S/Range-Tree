
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
	/*include logic to find the correct number of threads and blocks to run*/
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
	std::cout << "-----------------------------------------------------Primary tree-------------------------------------------------------------------\n";
	for (int i = 0; i < N; i++)
	{
		std::cout << primary[i].x << " " << primary[i].y << std::endl;
	}
	std::cout << "-----------------------------------------------------Secondary tree-----------------------------------------------------------------\n";
	for (int i = 0; i < N*logN; i++)
	{
		std::cout << secondary[i].x << " " << secondary[i].y << std::endl;
	}
}
/*
__device__ void solve_y(int d,int p,Query *queries,Point *secondary,int N,int logN)
{
	//code for searching on secondary tree
	int start, end;
	int idx = threadIdx.x;
	int l = queries[idx].b;
	int r = queries[idx].d;

	
	d = logN - d;

	start = d*N + (p - 1)*(1 << d);
	end = start + (1 << d) - 1;



	int flag = 1;
	while(flag)
	{
		flag = 0;
		int checkAt = start + (end - start) / 2;
		if (r <= secondary[checkAt].y)
		{
			end = checkAt;
			flag = 1;
		}
		else if (l>secondary[checkAt].y)
		{
			start = checkAt + 1;
			flag = 1;

		}

		if (start >= end)break;
	}
	if (start == end)
	{
		if (l <= secondary[start].y && secondary[start].y <= r)
			printf("%d %d", secondary[start] .x, secondary[start].y);
		return;
	}
	int s = start;
	int e = end;
	end = s + (e - s) / 2;

	while (start <= end)
	{
		
		int checkAt = start + (end - start) / 2;
		if (start == end)
		{
			if (l <= secondary[checkAt].y && secondary[checkAt].y <= r)
			{
				printf("%d %d", secondary[checkAt].x, secondary[checkAt].y);
			}
			break;
		}
		if (l <= secondary[checkAt].y)
		{
			for (j = start + (end - start) / 2 + 1; j <= end; j++)
				printf("%d %d", secondary[j].x, secondary[j].y);
			end = checkAt;
		}
		else
		{
			start = checkAt + 1;
		}
	}

	start = s + (e - s) / 2 + 1;
	end = e;
	while (start <= end)
	{
		
		int checkAt = start + (end - start) / 2;
		if (start == end)
		{
			if (l <= secondary[checkAt].y && secondary[checkAt].y <= r)
				printf("%d %d", secondary[checkAt].x, secondary[checkAt].y);
			break;
		}
		if (r <= secondary[checkAt].y)
		{
			end = checkAt;
		}
		else
		{
			for (int j = start; j <= checkAt;j++)
				printf("%d %d", secondary[j].x, secondary[j].y);
			start = checkAt + 1;
		}
	}
	return;
}
__device__ int right_child(int start, int end) //correct
{
	int s = start + (end - start) / 2;
	s++;
	return s + (end - s) / 2;
}
__device__ int left_child(int start, int end) //correct
{
	int e = start + (end - start) / 2;
	return start + (e - start) / 2;
}
__device__ int get_pos(int sz, int x) //correct
{
	int c = 1;
	int i = -1;
	while (1)
	{
		i += sz;
		if (i >= x)
		{
			return c;
		}
		c++;

	}
}
__global__ void solve_x(Query *queries, int *d_point_indices,Point* primary,Point* secondary,int N,int logN)
{
	int idx = threadIdx.x;
	int l = queries[idx].a;
	int r = queries[idx].c;
	
	
	int start = 0;
	int end = N - 1;
	int depth = 0;
	int flag = 1;
	while (flag)
	{
		flag = 0;
		int checkAt = start + (end - start) / 2;
		if (r <= primary[checkAt].x)
		{
			end = checkAt;
			flag = 1;
			depth++;
		}
		else if (l>primary[checkAt].x){
			start = checkAt + 1;
			flag = 1;
			depth++;
		}
		if (start >= end)break;
	}
	if (start == end)
	{
		if (l <= primary[start].x && primary[start].x <= r)
		{
			solve_y(depth, start + 1, queries,secondary,N,logN);
		}
		return;
	}
	depth++;
	int s = start, e = end;
	end = s + (e - s) / 2;
	int d = depth;

	
	while (start <= end)
	{
		int checkAt = start + (end - start) / 2;
		if (start == end)
		{
			if (l <= primary[checkAt].x && primary[checkAt].x <= r)
			{
				solve_y(depth, checkAt + 1, queries, secondary,N,logN);//array indexing is 0 based but calculations are done with 1 based.
			}
			break;
		}
		if (l <= primary[checkAt].x)
		{
			int rc = right_child(start, end);
			solve_y(depth + 1, get_pos((end - start + 1) / 2, rc), queries, secondary,N,logN);
			end = checkAt;
		}
		else
		{
			start = checkAt + 1;
		}
		depth++;
	}





	start = s + (e - s) / 2 + 1;
	end = e;
	depth = d;
	while (start <= end)
	{
	
		int checkAt = start + (end - start) / 2;
		if (start == end)
		{
			if (primary[checkAt].x >= l && primary[checkAt].x <= r)
			{
				 solve_y(depth, checkAt + 1, queries, secondary,N,logN);
			}
			break;
		}
		if (r <= primary[checkAt].x)
		{
			end = checkAt;
		}
		else
		{
			int lc = left_child(start, end);
			solve_y(depth + 1, get_pos((end - start + 1) / 2, lc), queries, secondary,N,logN);
			start = checkAt + 1;
		}
		depth++;
	}
}
void solve(Query *d_queries, int *d_point_indices,,Point* d_primary,Point* d_secondary)
{
	//include logic to find the correct number of threads to run
	solve_x<<<1, q >>>(d_queries,d_point_indices,d_primary,d_secondary,N,logN);

}
*/
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
	
	
	/*std::cin >> q;
	Query *queries = (Query*)malloc(q*sizeof(Query));
	for(int i = 0; i < q; i++)
	{
		int x, y, a, b;
		std::cin >> x >> y >> a >> b;
		if (x > a)
			swap(x, a);
		if (y > b)
			swap(y, b);
		Query t(x,y,a,b);
		queries[i] = t;
	}
	Query* d_queries;
	int *d_point_indices;
	HANDLE_ERROR(cudaMalloc((void**)&d_queries, q*sizeof(Query)));
	HANDLE_ERROR(cudaMemcpy(d_queries, queries, q*sizeof(Query), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMalloc((void**)&d_point_indices, N*sizeof(int)));
	solve(d_queries,d_point_indices,d_primary,d_secondary);
*/
	return 0;
}