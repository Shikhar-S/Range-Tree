
#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>
#include <cuda.h>
#include <cstdio>
#define HANDLE_ERROR(x) if((x)!=cudaSuccess){std::cout<<cudaGetErrorString((x))<<std::endl;exit(-1);}

/****************CODE STARTS HERE*******************/

__device__ int satisfies(int i,int j,int *A,int *B)
{
	return (A[i]<=B[j]);
}

__global__ void MergePath(int *A,int *B,int* C,int *x,int *y,int n)
{
    int num_of_threads=blockDim.x;
    int idx=threadIdx.x;
	if (idx==0)
	{
		x[idx]=0;
		y[idx]=0;
		return;
	}
	int A_start=idx*(2*n)/num_of_threads; //only when len(A)==len(B)
	int B_start=max(0,A_start-(n-1));
	A_start=min(n-1,A_start);
	int length_of_array;
	
	if(B_start==0)
	{
        
	   length_of_array=A_start+1;
    }
	else
		length_of_array=n-B_start;

	int left=0,right=length_of_array-1;
    // cout<<A_start<<" "<<B_start<<" "<<length_of_array<<endl<<"-------------------------------------------\n";
	bool flag=false
    while(left<=right && !flag)
	{
        // cout<<left<<" "<<right<<endl;
		int mid=left+(right-left)/2;
        int I=A_start-mid;
        int J=B_start+mid;
        if(!satisfies(I,J,A,B))
        {
            left=mid+1;
        }
        else
        {
            if(J==0)
            {
                x[idx]=(I+1);
                y[idx]=(J);
                flag=true;
            }
            else if(I==n-1)
            {
                x[idx]=(I+1);
                y[idx]=(J);
                flag=true;
            }
            else
            {
                if(!satisfies(I+1,J-1,A,B))
                {
                    x[idx]=(I+1);
                    y[idx]=(J);
                    flag=true;
                }
                else
                {
                    right=mid;
                }
            }
        }
	}
    left--;
    if(!flag)
    {
        x[idx]=(A_start-left);
        y[idx]=(n);
    }
    __syncthreads();

    int end_x,end_y;
    if(idx==num_of_threads-1)
    {
        end_x=n;
        end_y=n;
    }
    else
    {
        end_x=x[idx+1];
        end_y=y[idx+1];
    }
    int cur_x=x[idx];
    int cur_y=y[idx];
    int put_at=cur_x+cur_y;
    while(cur_x<end_x && cur_y<end_y)
    {
        if(A[cur_x]<=B[cur_y])
        {
            C[put_at++]=A[cur_x++];
        }
        else
        {
            C[put_at++]=B[cur_y++];
        }
    }
    while(cur_x<end_x)
        C[put_at++]=A[cur_x++];
    while(cur_y<end_y)
        C[put_at++]=B[cur_y++];
}
void printArr(int *C)
{
    for(int i=0;i<C.size();i++)
        cout<<C[i]<<" ";
    cout<<endl;
}

int main()
{
    int n;
    cin>>n;
    int *C=(int*)malloc(sizeof(int)*2*n);
    int *A=(int*)malloc(sizeof(int)*n);
    int *B=(int*)malloc(sizeof(int)*n);
    int i=0;
    for(int i=0;i<n;i++)
    {
        std::cin>>A[i];
    }
    for(int i=0;i<n;i++)
    {
    	std::cin>>B[i];
    }
    sort(A,A+n);
    sort(B,B+n);
    int num_of_threads;
    std::cin>>num_of_threads;
    int *x=(int*)malloc(sizeof(int)*num_of_threads);
    int *y=(int*)malloc(sizeof(int)*num_of_threads);
    HANDLE_ERROR(cudaMalloc(d_A,n*sizeof(int)));
    HANDLE_ERROR(cudaMalloc(d_B,n*sizeof(int)));
    HANDLE_ERROR(cudaMalloc(d_C,2*n*sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(d_A, A, N*sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_B,B,N*sizeof(int),cudaMemcpyHostToDevice));
    MergePath<<<1,num_of_threads>>>(d_A,d_B,d_C,x,y,n);
    HANDLE_ERROR(cudaMemcpy(C,d_C,2*N*sizeof(int),cudaMemcpyDeviceToHost));
    printArr(C);
    return 0;
}