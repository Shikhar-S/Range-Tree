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
int a[1000006];
int b[1000006];
int c[1000006];
int main()
{
    int n;
    cin>>n;

    for(int i=0;i<n;i++)
    	cin>>a[i];
    for(int i=0;i<n;i++)
    	cin>>b[i];
    clock_t start=clock();
    int left=0,right=0,put=0;
    while(left<n && right<n)
    {
    	if(a[left]<b[right])
    		c[put++]=a[left++];
    	else
    		c[put++]=b[right++];
    }
    while(left<n)
    {
    	c[put++]=a[left++];
    }
    while(right<n)
    {
    	c[put++]=b[right++];
    }
    clock_t end=clock();
    cout<<((end-start)*(1000.0)/CLOCKS_PER_SEC)<<endl;
    return 0;
}