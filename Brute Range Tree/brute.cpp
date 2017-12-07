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
vector<pii> P;
int main()
{
    int n;
    cin>>n;
    for(int i=0;i<n;i++)
    {
    	pii t;
    	cin>>t.ff>>t.ss;
    	P.pb(t);
    }
    int q;
    cin>>q;
    int ctr=0;
    while(q--)
    {
    	int a,b,c,d;
    	cin>>a>>b>>c>>d;
    	cout<<"Index of Query-->"<<ctr<<endl;
    	ctr++;
    	for(int i=0;i<n;i++)
    	{
    		
    		if(a<=P[i].ff){
    		if(P[i].ff<=c){
    		if(b<=P[i].ss){
    		if(P[i].ss<=d)
    		{
    			cout<<P[i].ff<<" "<<P[i].ss<<endl;
    		} }}}
    		
    	}
    }
    return 0;
}