#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <string>
#include <cctype>
#include <stack>
#include <queue>
#include <list>
#include <vector>
#include <map>
#include <cmath>
#include <bitset>
#include <utility>
#include <set>
#include <numeric>
#include <unordered_map>
#include <ctime>
#define MOD 1000000007
using namespace std;
typedef long long ll;
typedef vector < int > vi;
typedef vector < ll > vll;
typedef pair <int, int> pii;
typedef pair<ll, ll> pll;
#define rep(i,a,n) for(i=a;i<=n;i++)
#define per(i,n,a) for(i=n;i>=a;i--)
#define ff first
#define ss second
#define pb push_back
#define mp make_pair
#define all(vi) vi.begin(), vi.end()
#define tr(container, it) for(typeof(container.begin()) it = container.begin(); it != container.end(); it++)
#define iOS ios_base::sync_with_stdio(false); cin.tie(NULL)
#define imax numeric_limits<int>::max()
#define imin numeric_limits<int>::min()
#define llmax numeric_limits<ll>::max()
#define llmin numeric_limits<ll>::min()
ll powmod(ll a,ll b) {ll res=1;if(a>=MOD)a%=MOD;for(;b;b>>=1){if(b&1)res=res*a;if(res>=MOD)res%=MOD;a=a*a;if(a>=MOD)a%=MOD;}return res;}
ll gcd(ll a , ll b){return b==0?a:gcd(b,a%b);}
/****************CODE STARTS HERE*******************/
vi A;
vi B;
vi x;
vi y;
int n;
int num_of_threads;
int satisfies(int i,int j)
{
	return (A[i]<=B[j]);
}
void getLimit(int idx)
{
	if (idx==0)
	{
		x.pb(0);
		y.pb(0);
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
	while(left<=right)
	{
        // cout<<left<<" "<<right<<endl;
		int mid=left+(right-left)/2;
        int I=A_start-mid;
        int J=B_start+mid;
        if(!satisfies(I,J))
        {
            left=mid+1;
        }
        else
        {
            if(J==0)
            {
                x.pb(I+1);
                y.pb(J);
                return;
            }
            else if(I==n-1)
            {
                x.pb(I+1);
                y.pb(J);
                return;
            }
            else
            {
                if(!satisfies(I+1,J-1))
                {
                    x.pb(I+1);
                    y.pb(J);
                    return;
                }
                else
                {
                    right=mid;
                }
            }
        }
	}
    left--;

    x.pb(A_start-left);
    y.pb(n);
}
void merge_using(int idx,vi &C)
{
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
void printArr(vi &C)
{
    for(int i=0;i<C.size();i++)
        cout<<C[i]<<" ";
    cout<<endl;
}

int main()
{
    cin>>n;
    vi C(2*n);
    int i=0;
    for(int i=0;i<n;i++)
    {
    	int x;cin>>x;
    	A.pb(x);
    }
    for(int i=0;i<n;i++)
    {
    	int x;cin>>x;
    	B.pb(x);
    }
    sort(A.begin(),A.end());
    sort(B.begin(),B.end());
   
    cin>>num_of_threads;
    clock_t start=clock();
    for(int i=0;i<num_of_threads;i++)
    {
        // cout<<i<<endl;
    	getLimit(i);
        // cout<<num_of_threads<<endl;
    }
    for(int i=0;i<num_of_threads;i++)
    {
    	merge_using(i,C);
    }
    clock_t end=clock();
    float tm=(float)(end-start)/CLOCKS_PER_SEC;
    printf("%.10f\n",tm);
    /*for(int i=0;i<num_of_threads;i++)
        cout<<x[i]<<" "<<y[i]<<endl;*/
    // printArr(C);
    return 0;
}