//matmul in C++
#include<algorithm>
#include<cmath>
#include<vector>
using namespace std; // yea controversial ik
#define BM 64
#define BN 64
#define BK 64
extern "C" void mul(const float* a,const float* b,float* c,int M,int N,int K){
    for(int i=0;i<M*N;i++)c[i]=0.0f;
    for(int ii=0;ii<M;ii+=BM){
        for(int kk=0;kk<K;kk+=BK){
            for(int jj=0;jj<N;jj+=BN){

                for(int i=ii;i<min(ii+BM,M);i++){
                    for(int k=kk;k<min(kk+BK,K);k++){
                        float a1=a[i*K+k];
                        for(int j=jj;j<min(jj+BN,N);j++){
                            c[i*N+j]+=a1*b[k*N+j];
                        }
                    }
                }
            }
        }
    }
}
extern "C" void softmax(const float* l,int n,float* res){
    float m=*max_element(l,l+n);
    float s=0.0f;
    for(int i=0;i<n;i++){
        res[i]=(exp(l[i]-m));
        s+=res[i];
    }
    s=1.0f/s;
    for(int i=0;i<n;i++){
        res[i]*=s;
    }
}