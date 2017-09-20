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
vector<pii> points;
int N;
vector<pair<pii,pii> > Queries;
bool comp_x(pii &A,pii &B) //correct
{
    return A.ff<B.ff;
}
vector<pii> Primary;
vector<pii> Secondary;
void copy_own(vector<pii>::iterator start,vector<pii>::iterator end,vector<pii> &X) //correct
{
    for(vector<pii>::iterator t=start;t!=end;t++)
    {
        X.pb(*t);
    }
}

void build_tree(int canonical_size,int start_index)//correct
{
    // cout<<canonical_size<<endl;
    if(canonical_size>=2*Primary.size())
    {
        // cout<<"Terminating primary size is"<<Primary.size()<<endl;
        return;
    }
    int N=Secondary.size();
    for(int i=start_index;i<N;i+=canonical_size)
    {
        
        int first_ptr=i;
        int second_ptr=i+canonical_size/2;
        while(first_ptr<i+canonical_size/2 && second_ptr<i+canonical_size)
        {
            if(Secondary[first_ptr].ss<Secondary[second_ptr].ss)
            {
                Secondary.pb(Secondary[first_ptr]);
                first_ptr++;
            }
            else
            {
                Secondary.pb(Secondary[second_ptr]);
                second_ptr++;
            }
        }
        while(first_ptr<i+canonical_size/2)
        {
            Secondary.pb(Secondary[first_ptr]);
            first_ptr++;
        }
        while(second_ptr<i+canonical_size)
        {
            Secondary.pb(Secondary[second_ptr]);
            second_ptr++;
        }
    }
    build_tree(canonical_size*2,start_index+Primary.size());
}
void build_tree() //correct
{
    sort(points.begin(),points.end(),comp_x);
    copy_own(points.begin(),points.end(),Primary);
    copy_own(points.begin(),points.end(),Secondary);
    clock_t s=clock();
    build_tree(2,0);
    clock_t e=clock();
    cout<<(((e-s)*(1000.0))/CLOCKS_PER_SEC)<<endl;
    
}
///////////////////////////////////////// To Check//////////////////////////////////////





vector<pii> include_all(int s,int e)
{
    vector<pii> ans;
    // cout<<"do check\n";
    for(int i=s;i<=e;i++)
    {
        ans.pb(Secondary[i]);
    }
    return ans;
}

vector<pii> query_y(int d,int p,int idx)
{
    int start,end;
    int l=Queries[idx].ff.ss;
    int r=Queries[idx].ss.ss;
    
    if(l>r)swap(l,r);
    int total=log2(N);
    d=total-d;
    
    start=d*N +(p-1)*(1<<d);
    end=start+(1<<d)-1;
    
    vector<pii> ans_temp;
    vector<pii> temp;
    
    
    
    int flag=1;
    while(flag)
    {
        flag=0;
        int checkAt=start+(end-start)/2;
        if(r<=Secondary[checkAt].ss)
        {
            end=checkAt;
            flag=1;
        }
        else if(l>Secondary[checkAt].ss)
        {
            start=checkAt+1;
            flag=1;
            
        }
        
        if(start>=end)break;
    }
    if(start==end)
    {
        if(l<=Secondary[start].ss && Secondary[start].ss<=r)
            ans_temp.pb(Secondary[start]);
        return ans_temp;
    }
    int s=start;
    int e=end;
    end=s+(e-s)/2;
    
    while(start<=end)
    {
        temp.clear();
        int checkAt=start+(end-start)/2;
        if(start==end)
        {
            if(l<=Secondary[checkAt].ss && Secondary[checkAt].ss<=r)
            {
                ans_temp.pb(Secondary[checkAt]);
            }
            break;
        }
        if(l<=Secondary[checkAt].ss)
        {
            temp=include_all(start+(end-start)/2+1,end);
            end=checkAt;
        }
        else
        {
            start=checkAt+1;
        }
        for(int i=0;i<temp.size();i++)
            ans_temp.pb(temp[i]);
    }
    
    start=s+(e-s)/2+1;
    end=e;
    while(start<=end)
    {
        temp.clear();
        int checkAt=start+(end-start)/2;
        if(start==end)
        {
            if(l<=Secondary[checkAt].ss && Secondary[checkAt].ss<=r)
                ans_temp.pb(Secondary[checkAt]);
            break;
        }
        if(r<=Secondary[checkAt].ss)
        {
            end=checkAt;
        }
        else
        {
            temp=include_all(start,checkAt);
            start=checkAt+1;
        }
        for(int i=0;i<temp.size();i++)
            ans_temp.pb(temp[i]);
    }
    return ans_temp;
}



int right_child(int start,int end) //correct
{
    int s=start+(end-start)/2;
    s++;
    return s+(end-s)/2;
}
int left_child(int start,int end) //correct
{
    int e= start+(end-start)/2;
    return start+(e-start)/2;
}
int get_pos(int sz,int x) //incorrect
{
    int c=1;
    int i=-1;
    while(1)
    {
        i+=sz;
        if(i>=x)
        {
            return c;
        }
        c++;
        
    }
}




vector<pii> query_x(int idx)
{
    int l=Queries[idx].ff.ff;
    int r=Queries[idx].ss.ff;
    if(l>r)swap(l,r);
    vector<pii> ans;
    int start=0,end=N-1;
    int depth=0;
    int flag=1;
    while(flag)
    {
        flag=0;
        int checkAt=start+(end-start)/2;
        if(r<=Primary[checkAt].ff)
        {
            end=checkAt;
            flag=1;
            depth++;
        }
        else if(l>Primary[checkAt].ff){
            start=checkAt+1;
            flag=1;
            depth++;
        }
        if(start>=end)break;
    }
    if(start==end)
    {
        vector<pii> ans;
        if(l<=Primary[start].ff && Primary[start].ff<=r)
        {
            ans=query_y(depth, start+1, idx);
        }
        return ans;
    }
    depth++;
    int s=start,e=end;
    end=s+(e-s)/2;
    int d=depth;
    
    vector<pii> temp;
    while(start<=end)
    {
        int checkAt=start+(end-start)/2;
        if(start==end)
        {
            if(l<=Primary[checkAt].ff && Primary[checkAt].ff<=r)
            {
                temp=query_y(depth,checkAt+1,idx);//array indexing is 0 based but calculations are done with 1 based.
                if(temp.size()>0)
                    ans.pb(*temp.begin());
            }
            break;
            
        }
        if(l<=Primary[checkAt].ff)
        {
            int rc=right_child(start,end);
            temp=query_y(depth+1,get_pos((end-start+1)/2,rc),idx);
            end=checkAt;
        }
        else
        {
            start=checkAt+1;
        }
        for(int i=0;i<temp.size();i++)
        {
            ans.push_back(temp[i]);
        }
        temp.clear();
        depth++;
    }
    
    
    
    
    
    start=s+(e-s)/2+1;
    end=e;
    depth=d;
    while(start<=end)
    {
        temp.clear();
        int checkAt=start+(end-start)/2;
        if(start==end)
        {
            if(Primary[checkAt].ff>=l && Primary[checkAt].ff<=r)
            {
                temp=query_y(depth,checkAt+1,idx);
                if(temp.size()>0)
                    ans.pb(*temp.begin());
                
            }
            break;
        }
        if(r<=Primary[checkAt].ff)
        {
            end=checkAt;
        }
        else
        {
            int lc=left_child(start,end);
            temp=query_y(depth+1,get_pos((end-start+1)/2,lc),idx);
            start=checkAt+1;
        }
        for(int i=0;i<temp.size();i++)
            ans.pb(temp[i]);
        depth++;
    }
    return ans;
}

/////////////////////////////////////////////////////////////////////////////////////
void solve(int idx) //correct
{
    cout<<"Index of query --> "<<idx<<endl;
    
    vector<pii> Ans=query_x(idx);
    for(int i=0;i<Ans.size();i++)
        cout<<Ans[i].ff<<" "<<Ans[i].ss<<endl;
}
void print_tree() //correct
{
    cout<<"PRIMARY TREE-->"<<endl;
    int idx=0;
    for(vector<pii>::iterator it=Primary.begin();it!=Primary.end();it++)
    {
        cout<<it->ff<<" "<<it->ss<<"<---"<<idx<<endl;
        idx++;
    }
    cout<<"Secondary Tree-->"<<endl;
    int c=0;
    idx=0;
    for(vector<pii>::iterator it=Secondary.begin();it!=Secondary.end();it++)
    {
        cout<<it->ff<<" "<<it->ss<<"<---"<<idx<<endl;
        idx++;
        c++;
        if(c==N)
        {   c=0;
            cout<<"/////////////////////////////////////////////////////////////\n";
            
        }
    }
}
int main()
{
    
    cin>>N;
    for(int i=0;i<N;i++)
    {
        pii t;
        cin>>t.ff>>t.ss;
        points.pb(t);
    }
    clock_t start=clock();
    build_tree();
    clock_t end=clock();
    cout<<(((end-start)*(1000.0))/CLOCKS_PER_SEC)<<endl;
    // print_tree();
    /*int q;
    cin>>q;
    int Q=q;
    while(q--)
    {
        pii x,y;
        cin>>x.ff>>x.ss>>y.ff>>y.ss;
        Queries.pb(mp(x,y)); //assuming first point is less than second.
    }
    for(int i=0;i<Q;i++)
        solve(i);*/
    return 0;
}