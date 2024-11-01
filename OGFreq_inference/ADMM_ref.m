function res = ADMM_ref()
sig = (h5read('signal.h5','/signal'));
sig_real=sig(:,1).';
sig_imag=sig(:,2).';
sig=sig_real+1j*sig_imag;

N = 128;
reslu = 1/N;
M=length(sig);

A = zeros(M,N);
Agrad = zeros(M,N);

grid = (0:reslu:1-reslu)';
for m = 1:M
    for n = 1:N
        temp = exp(1i * 2*pi *grid(n)*m);
        A(m,n) = temp;
        Agrad(m,n) = 1i *2*pi*m  * temp;
    end
end

Y=sig.'/max(abs(sig));
L=length(Y);
maxiter = 200;
tol = 1e-15;
Aori = A;

lambda1=(var(Y(:)))*0.1;
lambda2=(var(Y(:)))*0.001;
rou=1;
beta=zeros(N,1);
pre_beta=beta;
pre_x=zeros(N,1);
alpha=zeros(L,1);
total_cnt=0;
total_converged=0;

while ~total_converged
    Cx=Aori+Agrad*diag(beta*reslu);
    x=shrink((Cx'*(alpha+Y))/L,lambda1/rou);
    %figure;plot(fftshift(abs(x)))
    
    Cbeta=Agrad*diag(x*reslu);
    beta=shrink((Cbeta'*(alpha+Y-Aori*x))/L,lambda2/rou);
    ind_large=find(abs(real(beta))>1/2);
    beta(ind_large)=sign(beta(ind_large))*0.01;
    alpha=alpha+Y-(Aori+Agrad*diag(beta*reslu))*x;
    
    error_x=norm(x-pre_x)/norm(pre_x);
    error_beta=norm(beta-pre_beta)/norm(pre_beta);
    if ((error_x<tol && error_beta<tol) || total_cnt>maxiter ) 
        total_converged=1;
    end
    pre_x=x;
    pre_beta=beta;
    total_cnt=total_cnt+1;
end

res.u=abs(x)/max(abs(x));
res.beta=real(beta)*reslu;
%figure;plot(fftshift(abs(x)))
%figure;plot(fftshift(real(beta)))
%a=1;
end

function A = shrink(B,gamma)
        A = sign(B).*max(abs(B)-gamma,0);
end