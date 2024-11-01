function res = OGSBI()


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
sigma2 = mean(var(Y))/100;

eps = 1e-16;

Aori = A;
alpha0 = 1/ sigma2; %  1/ paras.sigma2;
alpha =zeros(N,1);            %paras.alpha;
alpha_bar=zeros(N,1);
I=[];
r = reslu;

[~,pos]=max(abs(A'*Y));
I=[I,pos];
I=sort(I);
converged=0;
iter = 0;
iter_beta=1;
BHB = Agrad' * Agrad;
B=alpha0*eye(M);

while ~converged 
    iter = iter + 1;
    A=Aori;
    A(:,I) = A(:,I) + Agrad(:,I) * diag(beta(I));
    vari=inv(A(:,I)'*B*A(:,I)+diag(1/(alpha(I)+eps)));
    u=vari*A(:,I)'*B*Y;

    for i =1:N
        Si=A(:,i)'*B*A(:,i)-A(:,i)'*B*A(:,I)*vari*A(:,I)'*B*A(:,i);
        Qi=A(:,i)'*B*Y-A(:,i)'*B*A(:,I)*vari*A(:,I)'*B*Y;
        si=Si/(1-alpha(i)*Si);
        qi=Qi/(1-alpha(i)*Si);
        alpha_hat(i)=(conj(qi)*qi-si)/si.^2;
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
    elseif alpha_hat(pos)<=0 && ismember(pos,I) 
        if length(I)>1
            I=setdiff(I, pos);
            alpha(pos)=0;
        end
    end
    K=sum(alpha>0);
    active_idx=find(alpha>0);

    P=A(:,active_idx)*inv(A(:,active_idx)'*A(:,active_idx))*A(:,active_idx)';
    alpha0=(M-K)/trace((eye(M)-P)*(Y*Y'));

    if norm(alpha - alpha_bar)/norm(alpha_bar) < tol || iter >= maxiter
        converged = true;
        iter_beta=5;
    end
    iter=iter+1;
    alpha_bar=alpha;


    %% update beta
    B=alpha0*eye(M);
    A = Aori;
    A(:,I) = A(:,I) + Agrad(:,I) * diag(beta(I));
    vari=inv(A(:,I)'*B*A(:,I)+diag(1/(alpha(I)+eps)));
    u=vari*A(:,I)'*B*Y;


    P = real(conj(BHB(I,I)) .* (u * u' + vari));
    v = real(diag(u) * (Agrad(:,I)' * (Y - A(:,I) * u)));
    v = v - real(diag(Agrad(:,I)' * A(:,I) * vari));
    temp1 = P \ v;

    if any(abs(temp1) > r/2) || any(diag(P) == 0)
        for i = 1:iter_beta
            for n = 1:length(I)
                temp_beta = beta(I);
                temp_beta(n) = 0;
                beta(I(n)) = (v(n) - P(n,:) * temp_beta) / P(n,n);
                if beta(I(n)) > r/2
                    beta(I(n)) = r/2;
                end
                if beta(I(n)) < -r/2
                    beta(I(n)) = -r/2;
                end
                if P(n,n) == 0
                    beta(I(n)) = 0;
                end
            end
        end
    else
        beta(I) = temp1;
    end
    ind_large=find(abs(beta)>reslu/2);
    beta(ind_large)=sign(beta(ind_large))*reslu/100;
end
res.mu = u;
res.I=I;
res.beta=beta;
end
