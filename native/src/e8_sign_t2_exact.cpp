#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <cassert>

using namespace std;

static constexpr int N=5;
static constexpr int NB=2*N;
static constexpr int UMAX=1<<NB;
static constexpr int SIGNBIT=1<<NB;
static constexpr int SLOTBITS=NB+1; // signed Pauli point bits
static constexpr int DEN=16*(1<<N); // 512 fixed probability denominator

struct Val { int32_t a=0,b=0; }; // (a+b sqrt2)/DEN for Bellman values; coeff units /8 elsewhere

static inline Val addv(Val x, Val y){ return {x.a+y.a,x.b+y.b}; }
static inline Val negv(Val x){ return {-x.a,-x.b}; }
static inline bool eqv(Val x, Val y){ return x.a==y.a && x.b==y.b; }
static inline int signv(int64_t a, int64_t b){
    if(b==0) return (a>0)-(a<0);
    if(a==0) return (b>0)-(b<0);
    if((a>0 && b>0)) return 1;
    if((a<0 && b<0)) return -1;
    __int128 aa=(__int128)a*a, bb=(__int128)2*b*b;
    if(aa==bb) return 0;
    if(a>0 && b<0) return aa>bb?1:-1;
    return aa>bb?-1:1;
}
static inline bool lessv(Val x, Val y){ return signv((int64_t)x.a-y.a,(int64_t)x.b-y.b)<0; }
static inline double decv(Val x){ return ((double)x.a+(double)x.b*sqrt(2.0))/DEN; }

// local labels I=0,X=1,Z=2,Y=3. table: output label, i exponent
static constexpr int OUT[4][4]={{0,1,2,3},{1,0,3,2},{2,3,0,1},{3,2,1,0}};
static constexpr int PH[4][4] ={{0,0,0,0},{0,0,3,1},{0,1,0,3},{0,3,1,0}};

static inline int parity_u(unsigned x){ return std::popcount(x)&1; }
static inline int enc(int n,int x,int z,int sign=0){ return x | (z<<n) | ((sign&1)<<(2*n)); }
static inline void decp(int n,int p,int &x,int &z,int &s){
    s=p>>(2*n); int u=p&((1<<(2*n))-1); x=u&((1<<n)-1); z=u>>n;
}
static inline int mulp(int n,int p,int q){
    int x,z,s,a,b,t; decp(n,p,x,z,s); decp(n,q,a,b,t);
    int xr=0,zr=0,phase=0;
    for(int i=0;i<n;i++){
        int l=((x>>i)&1)|(((z>>i)&1)<<1);
        int r=((a>>i)&1)|(((b>>i)&1)<<1);
        int o=OUT[l][r]; phase=(phase+PH[l][r])&3;
        xr|=(o&1)<<i; zr|=((o>>1)&1)<<i;
    }
    if(phase!=0 && phase!=2){ cerr<<"anticommuting multiply\n"; abort(); }
    return enc(n,xr,zr,s^t^(phase>>1));
}
static inline bool commute_u(int n,int u,int v){
    int mask=(1<<n)-1; int x=u&mask,z=u>>n,a=v&mask,b=v>>n;
    return parity_u((x&b)^(z&a))==0;
}

vector<int> extend_code(const vector<int>& code,int q,int sign,int n){
    int sq=q | (sign<<(2*n));
    vector<int> out; out.reserve(code.size()*2);
    out.insert(out.end(),code.begin(),code.end());
    for(int g:code) out.push_back(mulp(n,g,sq));
    sort(out.begin(),out.end());
    return out;
}

vector<int> orth_basis3(int d){
    vector<int> basis; vector<int> span{0};
    for(int b=1;b<8;b++) if(parity_u(b&d)==0 && find(span.begin(),span.end(),b)==span.end()){
        basis.push_back(b); int sz=span.size(); for(int i=0;i<sz;i++) span.push_back(span[i]^b);
        if(basis.size()==2) return basis;
    }
    abort();
}
vector<int> state_code3(int u,int d,int sign){
    vector<int> code{enc(3,0,0,0)};
    code=extend_code(code,d,sign,3);
    for(int b:orth_basis3(d)) code=extend_code(code,b<<3,parity_u(b&u),3);
    return code;
}

array<array<Val,UMAX>,2> HYP;

void build_hyp(){
    // System sign mixtures. sys_num[s][u] means coefficient sys_num/4.
    array<array<int,64>,2> sys{};
    const int us[4]={0b000,0b010,0b001,0b100};
    const int ds[4]={0b111,0b100,0b010,0b001};
    for(int s=0;s<2;s++){
        for(int j=0;j<4;j++){
            auto code=state_code3(us[j],ds[j],s);
            for(int p:code){ int x,z,sg; decp(3,p,x,z,sg); int uu=x|(z<<3); sys[s][uu]+=sg?-1:1; }
        }
    }
    for(auto &arr:HYP) for(auto &v:arr) v={0,0};
    // Magic choices per ancilla: I, X, Y. k nonidentity gives factor (1/sqrt2)^k.
    for(int s=0;s<2;s++) for(int su=0;su<64;su++){
        int c=sys[s][su]; if(!c) continue;
        int sx=su&7, sz=su>>3;
        for(int m0=0;m0<3;m0++) for(int m1=0;m1<3;m1++){
            int x=sx,z=sz,k=0;
            int ms[2]={m0,m1};
            for(int j=0;j<2;j++){
                if(ms[j]){ k++; x|=1<<(3+j); if(ms[j]==2) z|=1<<(3+j); }
            }
            int uu=x|(z<<N);
            Val coeff;
            // combined coefficient in units of 1/8:
            // system c/4 times (1/sqrt2)^k.
            if(k==0) coeff={2*c,0};
            else if(k==1) coeff={0,c};
            else coeff={c,0};
            HYP[s][uu]=addv(HYP[s][uu],coeff);
        }
    }
}

uint64_t canonical_key(const vector<int>& code){
    array<int,UMAX> sign_of; sign_of.fill(-1);
    vector<int> rows;
    for(int p:code){ int u=p&(UMAX-1); int sg=(p>>NB)&1; sign_of[u]=sg; if(u) rows.push_back(u); }
    int r=std::countr_zero((unsigned)code.size());
    int row=0;
    for(int col=NB-1;col>=0 && row<r;col--){
        int piv=-1; for(int i=row;i<(int)rows.size();i++) if((rows[i]>>col)&1){piv=i;break;}
        if(piv<0) continue;
        swap(rows[row],rows[piv]);
        for(int i=0;i<(int)rows.size();i++) if(i!=row && ((rows[i]>>col)&1)) rows[i]^=rows[row];
        row++;
    }
    rows.resize(r);
    // order is pivot descending and canonical after full elimination
    uint64_t key=0;
    for(int i=0;i<r;i++){
        int u=rows[i]; int sg=sign_of[u]; if(sg<0){ cerr<<"missing sign lookup\n"; abort(); }
        int p=u|(sg<<NB); key|=(uint64_t)p<<(i*SLOTBITS);
    }
    return key;
}

vector<int> basis_from_key(uint64_t key){
    vector<int> b;
    uint64_t mask=(1ULL<<SLOTBITS)-1;
    for(int i=0;i<N;i++){
        int p=(key>>(i*SLOTBITS))&mask;
        if(!p) break;
        b.push_back(p);
    }
    return b;
}
vector<int> code_from_key(uint64_t key){
    auto basis=basis_from_key(key); vector<int> code{0};
    for(int p:basis){
        int q=p&(UMAX-1),sg=(p>>NB)&1;
        code=extend_code(code,q,sg,N);
    }
    return code;
}

// linear independence insertion on unsigned bit vectors
bool insert_lin(array<int,NB>& bas,int v){
    for(int bit=NB-1;bit>=0;bit--) if((v>>bit)&1){
        if(bas[bit]) v^=bas[bit];
        else { bas[bit]=v; return true; }
    }
    return false;
}

vector<int> nullspace_basis(const vector<int>& constraints){
    vector<int> rows=constraints; int m=rows.size(); int row=0;
    vector<int> pivcol;
    for(int col=NB-1;col>=0 && row<m;col--){
        int piv=-1; for(int i=row;i<m;i++) if((rows[i]>>col)&1){piv=i;break;}
        if(piv<0) continue;
        swap(rows[row],rows[piv]);
        for(int i=0;i<m;i++) if(i!=row && ((rows[i]>>col)&1)) rows[i]^=rows[row];
        pivcol.push_back(col); row++;
    }
    rows.resize(row);
    array<bool,NB> is_p{}; for(int c:pivcol) is_p[c]=true;
    vector<int> ns;
    for(int f=0;f<NB;f++) if(!is_p[f]){
        int v=1<<f;
        for(int i=0;i<row;i++){
            int p=pivcol[i];
            if(parity_u(rows[i]&v)) v|=1<<p;
        }
        ns.push_back(v);
    }
    return ns;
}

vector<int> logical_reps(uint64_t key){
    auto sb=basis_from_key(key);
    vector<int> su; su.reserve(sb.size());
    vector<int> constraints; constraints.reserve(sb.size());
    int mask=(1<<N)-1;
    for(int p:sb){
        int u=p&(UMAX-1); su.push_back(u);
        int x=u&mask,z=u>>N;
        constraints.push_back(z | (x<<N)); // dot(q, (z_s,x_s))=0
    }
    auto ns=nullspace_basis(constraints);
    array<int,NB> lin{};
    for(int u:su) insert_lin(lin,u);
    vector<int> comp;
    for(int v:ns) if(insert_lin(lin,v)) comp.push_back(v);
    int expected=2*(N-(int)su.size());
    if((int)comp.size()!=expected){ cerr<<"bad complement "<<comp.size()<<" expected "<<expected<<"\n"; abort(); }
    vector<int> reps; reps.reserve((1<<comp.size())-1);
    for(int bits=1;bits<(1<<(int)comp.size());bits++){
        int q=0; for(int i=0;i<(int)comp.size();i++) if((bits>>i)&1) q^=comp[i];
        reps.push_back(q);
    }
    return reps;
}

struct Hash64 { size_t operator()(uint64_t x) const noexcept { x^=x>>33; x*=0xff51afd7ed558ccdULL; x^=x>>33; x*=0xc4ceb9fe1a85ec53ULL; x^=x>>33; return (size_t)x; } };
unordered_map<uint64_t,Val,Hash64> memo;
uint64_t completed=0;
auto started=chrono::steady_clock::now();
array<uint64_t,N+1> byrank{};

Val solve(uint64_t key){
    auto it=memo.find(key); if(it!=memo.end()) return it->second;
    auto code=code_from_key(key); int r=std::countr_zero((unsigned)code.size());
    Val sums[2]={{0,0},{0,0}};
    for(int p:code){
        int u=p&(UMAX-1); int sg=(p>>NB)&1;
        for(int s=0;s<2;s++){
            Val c=HYP[s][u]; if(sg)c=negv(c); sums[s]=addv(sums[s],c);
        }
    }
    int factor=1<<(N-r);
    Val v0={sums[0].a*factor,sums[0].b*factor};
    Val v1={sums[1].a*factor,sums[1].b*factor};
    Val best=lessv(v0,v1)?v1:v0;
    Val upper=addv(v0,v1);
    if(r<N && lessv(best,upper)){
        auto reps=logical_reps(key);
        for(int q:reps){
            auto plus=extend_code(code,q,0,N); auto minus=extend_code(code,q,1,N);
            uint64_t kp=canonical_key(plus), km=canonical_key(minus);
            Val cand=addv(solve(kp),solve(km));
            if(lessv(best,cand)) best=cand;
            if(eqv(best,upper)) break;
        }
    }
    memo.emplace(key,best); completed++; byrank[r]++;
    if(completed%100000==0){
        double sec=chrono::duration<double>(chrono::steady_clock::now()-started).count();
        cerr<<"completed="<<completed<<" rank="<<r<<" memo="<<memo.size()<<" nodeV=("<<best.a<<","<<best.b<<")/"<<DEN<<" dec="<<decv(best)<<" elapsed="<<sec<<"s\n";
    }
    return best;
}

int main(){
    build_hyp();
    // sanity: root traces each 1
    vector<int> rootcode{0}; uint64_t root=canonical_key(rootcode);
    memo.reserve(25000000);
    memo.max_load_factor(0.8);
    Val ans=solve(root);
    double sec=chrono::duration<double>(chrono::steady_clock::now()-started).count();
    cout<<"RESULT_NUM "<<ans.a<<" "<<ans.b<<" DEN "<<DEN<<"\n";
    cout<<"p = ("<<ans.a<<" + "<<ans.b<<" sqrt(2))/"<<DEN<<" = "<<decv(ans)<<"\n";
    cout<<"completed="<<completed<<" memo="<<memo.size()<<" elapsed="<<sec<<"s\n";
    cout<<"byrank"; for(int r=0;r<=N;r++) cout<<" r"<<r<<"="<<byrank[r]; cout<<"\n";
}
