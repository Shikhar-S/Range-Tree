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
ll powmod(ll a, ll b) { ll res = 1; if (a >= MOD)a %= MOD; for (; b; b >>= 1){ if (b & 1)res = res*a; if (res >= MOD)res %= MOD; a = a*a; if (a >= MOD)a %= MOD; }return res; }
ll gcd(ll a, ll b){ return b == 0 ? a : gcd(b, a%b); }
/****************CODE STARTS HERE*******************/
vector<pii> points;
int N;
vector<pair<pii, pii> > Queries;
bool comp_x(pii &A, pii &B) //correct
{
	return A.ff<B.ff;
}
vector<pii> Primary;
vector<pii> Secondary;
void copy_own(vector<pii>::iterator start, vector<pii>::iterator end, vector<pii> &X) //correct
{
	for (vector<pii>::iterator t = start; t != end; t++)
	{
		X.pb(*t);
	}
}

void build_tree(int canonical_size, int start_index)//correct
{
	// cout<<canonical_size<<endl;
	if (canonical_size >= 2 * Primary.size())
	{
		// cout<<"Terminating primary size is"<<Primary.size()<<endl;
		return;
	}
	int N = Secondary.size();
	for (int i = start_index; i<N; i += canonical_size)
	{

		int first_ptr = i;
		int second_ptr = i + canonical_size / 2;
		while (first_ptr<i + canonical_size / 2 && second_ptr<i + canonical_size)
		{
			if (Secondary[first_ptr].ss<Secondary[second_ptr].ss)
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
		while (first_ptr<i + canonical_size / 2)
		{
			Secondary.pb(Secondary[first_ptr]);
			first_ptr++;
		}
		while (second_ptr<i + canonical_size)
		{
			Secondary.pb(Secondary[second_ptr]);
			second_ptr++;
		}
	}
	build_tree(canonical_size * 2, start_index + Primary.size());
}
void build_tree() //correct
{
	clock_t start, end;
	//////////////////////////////
	start = clock();
	/////////////////////////////
	sort(points.begin(), points.end(), comp_x);
	/////////////////////////////
	end = clock();
	float milliseconds = ((end - start)*1000.0) / CLOCKS_PER_SEC;
	cout << N << " Primary sorting time ---->" << milliseconds << endl;
	/////////////////////////////
	copy_own(points.begin(), points.end(), Primary);
	copy_own(points.begin(), points.end(), Secondary);
	//////////////////////////////
	start = clock();
	/////////////////////////////
	build_tree(2, 0);
	/////////////////////////////
	end = clock();
	 milliseconds = ((end - start)*1000.0) / CLOCKS_PER_SEC;
	cout << N << " Secondary build time ---->" << milliseconds << endl;
	/////////////////////////////
}


/////////////////////////////////////////////////////////////////////////////////////

void print_tree() //correct
{
	cout << "PRIMARY TREE-->" << endl;
	int idx = 0;
	for (vector<pii>::iterator it = Primary.begin(); it != Primary.end(); it++)
	{
		cout << it->ff << " " << it->ss << "<---" << idx << endl;
		idx++;
	}
	cout << "Secondary Tree-->" << endl;
	int c = 0;
	idx = 0;
	for (vector<pii>::iterator it = Secondary.begin(); it != Secondary.end(); it++)
	{
		cout << it->ff << " " << it->ss << "<---" << idx << endl;
		idx++;
		c++;
		if (c == N)
		{
			c = 0;
			cout << "/////////////////////////////////////////////////////////////\n";

		}
	}
}
int main()
{
	FILE *f = fopen("f.txt", "r");
	//cin >> N;
	fscanf(f, "%d", &N);
	for (int i = 0; i<N; i++)
	{
		pii t;
		//cin >> t.ff >> t.ss;
		fscanf(f, "%d %d", &t.ff, &t.ss);
		points.pb(t);
	}
	

	build_tree();
	
	// print_tree();
	/*
	int q;
	cin >> q;
	int Q = q;
	while (q--)
	{
		pii x, y;
		cin >> x.ff >> x.ss >> y.ff >> y.ss;
		Queries.pb(mp(x, y)); //assuming first point is less than second.
	}
	for (int i = 0; i<Q; i++)
		solve(i);
	*/
	return 0;
}