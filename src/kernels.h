// OpenCL kernels embedded in the executable.
#pragma once

// Copied unchanged from deflate.cl in the original "OpenCL Test" folder (fixed-Huffman DEFLATE, independent chunks).
inline const char* kDeflateKernel = R"CLC(
// Fixed-Huffman DEFLATE, independent byte-aligned chunks (RFC 1951).
__constant uint LB[29]={3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258};
__constant uint LE[29]={0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};
__constant uint DB[30]={1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577};
__constant uint DE[30]={0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};
uint rev(uint v,uint n){uint r=0;for(uint i=0;i<n;i++){r=(r<<1)|(v&1);v>>=1;}return r;}
void bits(__global uchar *out,uint *pos,uint value,uint n){
 for(uint i=0;i<n;i++){uint p=*pos;out[p>>3]|=(uchar)(((value>>i)&1)<<(p&7));*pos=p+1;}
}
void symbol(__global uchar *out,uint *pos,uint s){
 uint c,n;if(s<=143){c=48+s;n=8;}else if(s<=255){c=400+s-144;n=9;}
 else if(s<=279){c=s-256;n=7;}else{c=192+s-280;n=8;}
 bits(out,pos,rev(c,n),n);
}
__kernel void deflate_blocks(__global const uchar *src,__global uchar *dst,
 __global const uint *lengths,__global uint *sizes,__global int *tables,uint block){
 uint id=get_global_id(0),n=lengths[id],i=0,p=0;
 size_t s=(size_t)id*block,d=(size_t)id*block*2,t=(size_t)id*4096;
 __global uchar *out=dst+d;
 for(uint j=0;j<block*2;j++)out[j]=0;
 for(uint j=0;j<4096;j++)tables[t+j]=-1;
 bits(out,&p,2,3); // BFINAL=0, fixed Huffman
 while(i<n){
  uint len=0,distance=0;
  if(i+3<=n){
   uint h=(((uint)src[s+i]*251u+src[s+i+1])*251u+src[s+i+2])&4095u;
   int prev=tables[t+h];tables[t+h]=(int)i;
   if(prev>=0 && i-(uint)prev<=32768){
    uint limit=min(258u,n-i);
    while(len<limit && src[s+(uint)prev+len]==src[s+i+len])len++;
    distance=i-(uint)prev;
   }
  }
  if(len>=3){
   uint lc=0;while(lc<28 && LB[lc+1]<=len)lc++;
   symbol(out,&p,257+lc);bits(out,&p,len-LB[lc],LE[lc]);
   uint dc=0;while(dc<29 && DB[dc+1]<=distance)dc++;
   bits(out,&p,rev(dc,5),5);bits(out,&p,distance-DB[dc],DE[dc]);i+=len;
  }else{symbol(out,&p,src[s+i]);i++;}
 }
 symbol(out,&p,256);
 bits(out,&p,0,3); // non-final empty stored block aligns next chunk
 p=(p+7)&~7u;bits(out,&p,0,16);bits(out,&p,65535,16);
 uint size=(p+7)/8;
 if(size>=n+5){
  out[0]=0;out[1]=(uchar)n;out[2]=(uchar)(n>>8);
  uint inverse=(~n)&65535;out[3]=(uchar)inverse;out[4]=(uchar)(inverse>>8);
  for(uint j=0;j<n;j++)out[5+j]=src[s+j];size=n+5;
 }
 sizes[id]=size;
}
)CLC";

// Decoders for the earlier archive formats (from opencl_lz.py and opencl_compress.py).
inline const char* kLegacyKernels = R"CLC(
__kernel void lz_decode(__global const uchar *src, __global uchar *dst,
 __global const uint *sizes,__global const uint *lengths,uint block) {
 uint id=get_global_id(0),size=sizes[id],n=lengths[id],p=0,o=0;
 size_t s=(size_t)id*block*2,d=(size_t)id*block;
 if(size==n){for(uint i=0;i<n;i++)dst[d+i]=src[s+i];return;}
 while(p<size && o<n){
  uint flags=src[s+p++];
  for(uint bit=0;bit<8 && o<n;bit++){
   if(flags&(1u<<bit)){
    uint distance=(uint)src[s+p]|((uint)src[s+p+1]<<8),length=(uint)src[s+p+2]+3;p+=3;
    for(uint j=0;j<length;j++){dst[d+o]=dst[d+o-distance];o++;}
   }else{dst[d+o++]=src[s+p++];}
  }
 }
}
__kernel void rle_decode(__global const uchar *src, __global uchar *dst,
                     __global const uint *sizes, __global const uint *lengths, uint block) {
    uint id=get_global_id(0), size=sizes[id], n=lengths[id];
    size_t s=(size_t)id*block*2, d=(size_t)id*block;
    if(size==n) { for(uint i=0;i<n;i++) dst[d+i]=src[s+i]; }
    else { uint p=0; for(uint i=0;i<size;i+=2) {
        uint count=src[s+i]; uchar v=src[s+i+1];
        for(uint j=0;j<count;j++) dst[d+p++]=v;
    }}
}
)CLC";

// Optimised version of deflate_blocks. It produces exactly the same output bytes, but:
//  - bits are collected in a 64-bit register and stored 32 bits at a time
//    (the original did a global-memory read-modify-write for every single bit,
//    and first zeroed the whole 64 KiB output slot);
//  - Huffman codes, length codes and distance codes come from lookup tables, not loops;
//  - the hash table uses 16-bit entries (half the memory traffic), filled with vector stores;
//  - match lengths are compared 4 bytes at a time.
// The host prepends the tables LIT[288], LCODE[259], DCODE[512] and DREV[30] (see zip_create.cpp).
inline const char* kDeflateKernelFast = R"CLC(
__constant uint LB[29]={3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258};
__constant uint LE[29]={0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};
__constant uint DB[30]={1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577};
__constant uint DE[30]={0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};

#define PUT(v,n) do{ acc|=(ulong)((v)&((1u<<(n))-1u))<<cnt; cnt+=(n); \
  if(cnt>=32){ *(__global uint*)(out+pos)=(uint)acc; acc>>=32; cnt-=32; pos+=4; } }while(0)
#define SYM(s) do{ uint e_=LIT[s]; PUT(e_&0xFFFFu, e_>>16); }while(0)

__kernel void deflate_blocks(__global const uchar *src,__global uchar *dst,
 __global const uint *lengths,__global uint *sizes,__global ushort *tables,uint block){
 uint id=get_global_id(0),n=lengths[id],i=0;
 __global const uchar *in=src+(size_t)id*block;
 __global uchar *out=dst+(size_t)id*block*2;
 __global ushort *table=tables+(size_t)id*4096;
 __global uint4 *t4=(__global uint4*)table;
 for(uint j=0;j<512;j++)t4[j]=(uint4)(0xFFFFFFFFu);
 ulong acc=0; uint cnt=0,pos=0;
 PUT(2u,3u); // BFINAL=0, fixed Huffman
 while(i<n){
  uint len=0,distance=0;
  if(i+3<=n){
   uint h=(((uint)in[i]*251u+in[i+1])*251u+in[i+2])&4095u;
   uint prev=table[h];table[h]=(ushort)i;
   if(prev!=0xFFFFu && i-prev<=32768u){
    uint limit=min(258u,n-i);
    __global const uchar *a=in+prev,*b=in+i;
    while(len+4<=limit && all(vload4(0,a+len)==vload4(0,b+len)))len+=4;
    while(len<limit && a[len]==b[len])len++;
    distance=i-prev;
   }
  }
  if(len>=3){
   uint lc=LCODE[len];
   SYM(257+lc);PUT(len-LB[lc],LE[lc]);
   uint dc=distance<=256u?DCODE[distance-1]:DCODE[256+((distance-1)>>7)];
   PUT(DREV[dc],5u);PUT(distance-DB[dc],DE[dc]);i+=len;
  }else{SYM(in[i]);i++;}
 }
 SYM(256u);
 PUT(0u,3u); // non-final empty stored block aligns next chunk
 cnt=(cnt+7u)&~7u;
 if(cnt>=32){ *(__global uint*)(out+pos)=(uint)acc; acc>>=32; cnt-=32; pos+=4; }
 PUT(0u,16u);PUT(65535u,16u);
 for(uint k=0;k<cnt;k+=8)out[pos++]=(uchar)(acc>>k);
 uint size=pos;
 if(size>=n+5){
  out[0]=0;out[1]=(uchar)n;out[2]=(uchar)(n>>8);
  uint inverse=(~n)&65535;out[3]=(uchar)inverse;out[4]=(uchar)(inverse>>8);
  for(uint j=0;j<n;j++)out[5+j]=in[j];size=n+5;
 }
 sizes[id]=size;
}
)CLC";

// Packs the compressed chunks end to end so only the real bytes are downloaded.
// One work-group per chunk; its work-items copy the chunk's bytes together (coalesced).
inline const char* kCompactKernel = R"CLC(
__kernel void compact(__global const uchar *slots,__global const uint *sizes,
 __global const uint *offsets,__global uchar *packed,uint slot){
 uint c=get_group_id(0),n=sizes[c];
 __global const uchar *from=slots+(size_t)c*slot;
 __global uchar *to=packed+offsets[c];
 for(uint j=get_local_id(0);j<n;j+=get_local_size(0))to[j]=from[j];
}
)CLC";
