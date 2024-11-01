
function res = ESPRIT()

tgt_num=(h5read('signal.h5','/tgt_num'));
sig = (h5read('signal.h5','/signal'));
sig_real=sig(:,1).';
sig_imag=sig(:,2).';

sig=sig_real+1j*sig_imag;

RPP=sig;
siz=size(RPP);
M=siz(1,1);
N=siz(1,2);
n1=ceil(2*N/5);
dim=n1;
fblooknum=(N-n1+1);


x=0;
X1=zeros(dim,fblooknum);
for b=1:(N-n1+1)    
      B1=RPP(b:(b+n1-1));    
      x=x+1;
      z1=B1(:);                             
      X1(:,x)=z1;        %w2                    
end


x1=X1(1:n1-1,:);
x2=X1(2:end,:);

X=[x1;x2]; 
R=X*X'/fblooknum; 
[U,S,V]=svd(R); 
Us=U(:,1:tgt_num); 
Us1=Us(1:size(Us,1)/2,:); 
Us2=Us(size(Us,1)/2+1:end,:); 


M=pinv(Us1)*Us2; 
[Vm,Dm]=eig(M); 
res=(diag(angle(Dm)/2/pi)).';
estimated_damp=1./log((diag(abs(Dm))).');


