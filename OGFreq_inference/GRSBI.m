function res = GRSBI()

format long


sig = (h5read('signal.h5','/signal'));
sig_real=sig(:,1).';
sig_imag=sig(:,2).';

sig=sig_real+1j*sig_imag;
L=128;
reslu = 1/(L-1);

N = L;
M=length(sig);

A = zeros(M,N);
Agrad = zeros(M,N);
grid=(-L/2:L/2-1)*reslu;
for m = 1:M
    for n = 1:N
        temp = exp(1i * 2*pi *grid(n)*m);
        A(m,n) = temp;
        Agrad(m,n) = 1i *2*pi*m  * temp;
    end
end

Y = sig.';
beta = zeros(N,1);

maxiter = 200;
tol = 1e-15;
eps = 1e-6;

alpha =zeros(N,1);            %paras.alpha;
alpha_bar=zeros(N,1);
I=[];

[~,pos]=max(abs(A'*Y));
I=[I,pos];
I=sort(I);
alpha(pos)=1;


converged=0;
iter = 0;
BHB = Agrad' * Agrad;
alpha0 = 1; %  1/ paras.sigma2;
lamda=1/alpha0;
B=lamda^(-1)*eye(M);

W_sq=B^(1/2);
B_w=W_sq*Agrad;
A_w=W_sq*A;
search_area=grid;



       

while ~converged || iter<=30
    iter = iter + 1;
    
    vari=pinv(A(:,I)'*B*A(:,I)+diag(1/(alpha(I)+eps)));
    u=vari*A(:,I)'*B*Y;

    for i =1:N
        Si=A(:,i)'*B*A(:,i)-A(:,i)'*B*A(:,I)*vari*A(:,I)'*B*A(:,i);
        Qi=A(:,i)'*B*Y-A(:,i)'*B*A(:,I)*vari*A(:,I)'*B*Y;
        si=Si/(1-alpha(i)*Si);
        qi=Qi/(1-alpha(i)*Si);
        alpha_hat(i)=(conj(qi)*qi-si)/si^2;
        if alpha_hat(i)>0 && ~ismember(i,I)
                mle(i)=Qi*conj(Qi)/Si+log(Si/(Qi*conj(Qi)))-1;
        elseif alpha_hat(i)>0 && ismember(i,I)
                mle(i)=Qi*conj(Qi)/(Si+1/(alpha_hat(i)-alpha(i)))-log(1+(alpha_hat(i)-alpha(i))*Si);
        elseif alpha_hat(i)<=0 
                mle(i)=Qi*conj(Qi)/(Si-1/alpha(i))-log(1-alpha(i)*Si);
        end
    end
    [~,pos]=max(abs(mle));
    if alpha_hat(pos)>0 && ~ismember(pos,I)
        I=[I,pos];
        I=sort(I);
        alpha(pos)=alpha_hat(pos);
    elseif alpha_hat(pos)>0 && ismember(pos,I)
        alpha(pos)=alpha_hat(pos);
    elseif alpha_hat(pos)<=0 && length(I)>1
        I=setdiff(I, pos);
        alpha(pos)=0;
    end
    K=sum(alpha>0);
    active_idx=find(alpha>0);
    vari=pinv(A(:,active_idx)'*B*A(:,active_idx)+diag(1/(alpha(active_idx)+eps)));
    u=vari*A(:,active_idx)'*B*Y;
    
    lamda=norm(Y-A(:,active_idx)*u)^2/(M-K);
    B=lamda^(-1)*eye(M);
    W_sq=B^(1/2);
    if norm(alpha - alpha_bar)/norm(alpha_bar) < tol || iter >= maxiter
        converged = true;
    end

    alpha_bar=alpha;
    

    %% Grid refinement
    idx =I;

    P = real(conj(BHB(I,I)) .* (u * u' + vari));
    v =  real(diag(conj(u)) * (B_w(:,idx)' * (Y - A_w(:,idx) * u)))...
        -  real(diag(B_w(:,idx)' * A_w(:,idx) * vari))  ;
    
    eigP=svd(P);
    if eigP(end)/eigP(1)>1e-5
        temp1 =  P \ v;
    else
        temp1=v./diag(P);
    end
    temp2=temp1';
    if iter<100
        ind_small=find(abs(temp2)<reslu/100 & abs(temp2)>reslu/10000);
        temp2(ind_small)=sign(temp2(ind_small))*reslu/100;
    end
    ind_large=find(abs(temp2)>reslu);
    temp2(ind_large)=sign(temp2(ind_large))*reslu/100;
    angle_cand=search_area(idx) + temp2;
    search_area(idx)=angle_cand;

    A_ect=exp(1i*2*pi*(1:size(A,1))'*search_area(idx));
    B_ect=(1i*2*pi*(1:size(A,1))')*ones(1,size(A_ect,2)).*A_ect;
    A_w(:,idx) =W_sq*A_ect;
    B_w(:,idx) =W_sq*B_ect;
    A(:,idx)= A_w(:,idx);
end
vari=inv(A(:,I)'*B*A(:,I)+diag(1/(alpha(I)+eps)));
u=vari*A(:,I)'*B*Y;
tmp=grid.';
tmp(idx)=search_area(idx);
res.grid=tmp;
res.mu = u;
res.I=I;
res.beta=beta;
end
