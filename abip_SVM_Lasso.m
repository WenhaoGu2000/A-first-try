function [x,y,s,info] = abip_direct_SVM(SVM_data,params,lambda)

%% formulation
[m_X,n_X]=size(SVM_data.X);
Z=spdiags(SVM_data.y,0:0,m_X,m_X)*SVM_data.X;
% Z=sparse(diag(SVM_data.y))*SVM_data.X;
y=SVM_data.y;

fprintf('Data formulation finished!\n');

data.A=sparse([speye(m_X)        -speye(m_X)      Z               -Z     y -y]);
% data.AT=data.A';
% A=data.A;
data.b=ones(m_X,1);
data.c=[ones(m_X,1);zeros(m_X,1);lambda*ones(n_X,1);lambda*ones(n_X,1);1;1];



%% default settings
max_outiters = 100; 
max_iters    = 1000000;             
eps          = 1e-3;                   
alpha        = 1.8; 
mu           = 1.0; 
normalize    = 1;                
scale        = 1;  
rho_y        = 1e-3;
sigma        = 0.3; 
adaptive     = 1;                 
eps_cor      = 0.2;
eps_pen      = 0.1; 
%增加了一些（大写表示）
SPARSITY_RATIO=0.01;
ADAPTIVE_LOOKBACK=20;
if m_X>n_X
    METHOD=1; 
else
    METHOD=2;
end
fprintf('method=%3d\n',METHOD);

% conjugate gradient (CG) settings:
use_indirect    = false;   % use conjugate gradient rather than direct method
extra_verbose   = false;  % CG prints summary

%% constants
undet_tol    = 1e-18;             % tol for undetermined solution (tau = kappa = 0)

%% 输出参数


%% parameter setting 
if nargin==3
    if isfield(params,'max_ipm_iters');   max_outiters = params.max_ipm_iters;   end
    if isfield(params,'max_admm_iters');  max_iters    = params.max_admm_iters;      end
    if isfield(params,'eps');             eps          = params.eps;            end
    if isfield(params,'alpha');           alpha        = params.alpha;          end
    if isfield(params,'sigma');           sigma        = params.sigma;          end
    if isfield(params,'normalize');       normalize    = params.normalize;      end
    if isfield(params,'scale');           scale        = params.scale;          end
    if isfield(params,'rho_y');           rho_y        = params.rho_y;          end
    if isfield(params,'adaptive');        adaptive     = params.adaptive;       end
    if isfield(params,'eps_cor');         eps_cor      = params.eps_cor;        end  
    if isfield(params,'eps_pen');         eps_pen      = params.eps_pen;        end  
    %增加了一些
    if isfield(params,'SPARSITY_RATIO');    SPARSITY_RATIO=params.SPARSITY_RATIO;    end
    if isfield(params,'ADAPTIVE_LOOKBACK');    ADAPTIVE_LOOKBACK=params.ADAPTIVE_LOOKBACK;    end
    if isfield(params,'METHOD');     METHOD=params.METHOD;  end
        % METHOD=1: m much larger than n
        % METHOD=2: m much smaller than n
        % METHOD=3: use indirect method
end





%% data setting 

% dimension
n = length(data.c); 
m = length(data.b); 
l = n+m+1;
u = zeros(l, 1);
v = zeros(l, 1);

%% 计算矩阵稀疏度 sp=非零元素数/总数
sp=nnz(data.A)/(m*n);


%% 输出一些奇怪的东西
    if (min(sp,SPARSITY_RATIO)>0.5)
        
        fprintf('inner_stopper=mu^(-0.35), rho_y = %3.2e, alpha = %3.2e ,adaptive_lookback = %3d\n',rho_y,alpha,ADAPTIVE_LOOKBACK);
    elseif (min(sp,SPARSITY_RATIO)>0.2)
        
        fprintf('inner_stopper=mu^(-1), rho_y = %3.2e, alpha = %3.2e ,adaptive_lookback = %3d\n',rho_y,alpha,ADAPTIVE_LOOKBACK);
    else
        
        fprintf('inner_stopper=max_iters, rho_y = %3.2e, alpha = %3.2e ,adaptive_lookback = %3d\n',rho_y,alpha,ADAPTIVE_LOOKBACK);
    end


%初始化work
work=struct();



%测试版本
[Q,nm_b,nm_c,data,scale,work,D,E,sc_b,sc_c,sigma,gamma,final_check,double_check,mu,beta,h,g,gTh]=update_work_SVM_Lasso(data,...
    scale,work,sp,SPARSITY_RATIO,use_indirect,normalize,rho_y, extra_verbose, m_X, n_X, METHOD);



err_pri = 0; 
err_dual = 0;
gap = 0;

%% Initialization  (cold start)
u(m+1:l) = ones(n+1,1)*sqrt(mu/beta); 
v(m+1:l) = ones(n+1,1)*sqrt(mu/beta);

k        = 0; 



%初始化一个ratio
ratio=mu/eps;
    
tic;
for i=0:max_outiters-1   
    % 确定inner_stopper
    if (min(sp,SPARSITY_RATIO)>0.5)
        inner_stopper=mu^(-0.35);
%         fprintf('inner_stopper=mu^(-0.35), rho_y = %3.2e, alpha = %3.2e ,adaptive_lookback = %3d\n',rho_y,alpha,ADAPTIVE_LOOKBACK);
    elseif (min(sp,SPARSITY_RATIO)>0.2)
        inner_stopper=mu^(-1);
%         fprintf('inner_stopper=mu^(-1), rho_y = %3.2e, alpha = %3.2e ,adaptive_lookback = %3d\n',rho_y,alpha,ADAPTIVE_LOOKBACK);
    else
        inner_stopper=max_iters;
%         fprintf('inner_stopper=max_iters, rho_y = %3.2e, alpha = %3.2e ,adaptive_lookback = %3d\n',rho_y,alpha,ADAPTIVE_LOOKBACK);
    end
    
%     inner_stopper=mu^(-0.35); %% 减少了内循环数
        
    for j=0:inner_stopper
        % u_pre = u; 
        % v_pre = v;
        
        %% solve linear system
%         [ut, ~] = project_lin_sys(work, data, n, m, k, u, v, rho_y, use_indirect, extra_verbose, ...
%             h, g, gTh);
        ut = project_lin_sys_SVM_Lasso(data, work, m_X,n_X,n, m, u, v, h, g, gTh, rho_y, METHOD);
        rel_ut      = alpha*ut+(1-alpha)*u;
        rel_ut(1:m) = ut(1:m);                       
        u           = rel_ut - v;
        temp        = u(m+1:end)/2;
        u(m+1:end)  = temp+sqrt(temp.*temp+mu/beta);
    
        %% dual update:
        v = v + (u - rel_ut);
    
        %% convergence checking: (内部calc_residuals)
        % err_inner = norm([u-u_pre; v-v_pre])/(1+norm([u;v])+norm([u_pre;v_pre]));
        err_inner = norm(Q*u-v)/(1+norm([u;v]));
%         err_inner = norm(Q_times_x(data, m, n , u)-v)/(1+norm([u;v]));
        tol = gamma*mu;
        k = k+1;
        % fprintf('||Qu-v|| = %3.6f, gamma = %3.6f, mu = %3.6f\n', err_inner, gamma, mu);
        if err_inner < tol
            break;
        end
        if (final_check && mod(j+1,1)==0)
            tau = abs(u(end));
            kap = abs(v(end)) / (sc_b * sc_c * scale);
            y   = u(1:m) / tau;
            x   = u(m+1:m+n) / tau;
            s   = v(m+1:m+n) / tau;
    
            err_pri  = norm(D.*(data.A * x - data.b)) / (1 + nm_b) / (sc_b * scale); 
            err_dual = norm(E.*(data.A' * y + s - data.c)) / (1 + nm_c) / (sc_c * scale); 
            pobj     = data.c' * x / (sc_c * sc_b * scale);
            dobj     = data.b' * y / (sc_c * sc_b * scale);
            gap      = abs(pobj - dobj) / (1 + abs(pobj) + abs(dobj));
            
            error_ratio = max(gap,max(err_pri,err_dual))/eps;
            solved = error_ratio < 1;
        
            if solved; 
                break; 
            end
            
            if k+1>max_iters
                break; 
            end
        end
    end
    
    %增加de内容
    if (k>max_iters*0.8)
        final_check=1;
    end
    
    %% convergence checking:  (外部calc_residuals)
    tau = abs(u(end));
    kap = abs(v(end)) / (sc_b * sc_c * scale);
    y   = u(1:m) / tau;
    x   = u(m+1:m+n) / tau;
    s   = v(m+1:m+n) / tau;
    err_pri  = norm(D.*(data.A * x - data.b)) / (1 + nm_b) / (sc_b * scale); 
    err_dual = norm(E.*(data.A' * y + s - data.c)) / (1 + nm_c) / (sc_c * scale); 
    pobj     = data.c' * x / (sc_c * sc_b * scale);
    dobj     = data.b' * y / (sc_c * sc_b * scale);
    gap      = abs(pobj - dobj) / (1 + abs(pobj) + abs(dobj));
    
    if (data.c'*u(m+1:m+n) < 0)
        unb_res = norm(E.*data.c) * norm(D.*(data.A * u(m+1:m+n))) / (-data.c'*u(m+1:m+n)) / scale;
    else
        unb_res = inf;
    end
        
    if (-data.b'*u(1:m) < 0)
        inf_res = norm(D.*data.b) * norm(E.*(data.A' * u(1:m) + v(m+1:m+n))) / (data.b'*u(1:m)) / scale;
    else
        inf_res = inf;
    end
    
    ratio = mu/eps;
    error_ratio = max(gap,max(err_pri,err_dual))/eps;
    solved = error_ratio < 1;
    infeasible = inf_res < eps;
    unbounded = unb_res < eps;
        
    ttime = toc;
    
    %% 输出实时结果
    fprintf('i: %5d, mu: %3.2e, k: %5d presi: %3.7e dresi: %3.7e, dgap: %3.7e, time: %3.2e \n', ...
                i, mu, k, err_pri, err_dual, gap, ttime);
    
    if (solved || infeasible || unbounded)
        break;
    end
    
    if (k+1>max_iters)    %换了个位置
        break;
    end
    
    
    %% update_barrier    
    [sigma,gamma,mu,final_check,double_check]=update_barrier(sigma,gamma,mu,ratio,...
        error_ratio, final_check,double_check,sp,SPARSITY_RATIO);
  
    
    %% reinitialization
%     v(m+1:l) = (mu/beta)./u(m+1:l);
%     
%     %% reinitialize beta
%     if adaptive
%         beta = 1; 
%         v(m+1:l) = (mu/beta)./u(m+1:l);
%         u(m+1:l) = (mu/beta)./v(m+1:l);
%         beta = BBspectral(work, data, mu, n, m, l, k, u, v, rho_y, use_indirect, extra_verbose, ...
%             h, g, gTh, beta, alpha, eps_cor, eps_pen,ADAPTIVE_LOOKBACK);
%         v(m+1:l) = (mu/beta)./u(m+1:l);
%     end
    [u,v]=reinitialize_vars(sigma,m,l,u,v,0);
    
    if adaptive
        [u,v]=reinitialize_vars(sigma,m,l,u,v,1);
        beta=1;
%         beta = BBspectral(work, data, mu, n, m, l, k, u, v, rho_y, use_indirect, extra_verbose, ...
%             h, g, gTh, beta, alpha, eps_cor, eps_pen,ADAPTIVE_LOOKBACK);
        beta = BBspectral_SVM_Lasso(data, work, m_X, n_X, mu, n, m, l, u, v, h, g, gTh, beta,...
                  alpha, eps_cor, eps_pen, ADAPTIVE_LOOKBACK, rho_y, METHOD);
        [u,v]=reinitialize_vars(sigma,m,l,u,v,2);
    end
    
end

if (k+1 > max_iters); k=k+1; end
if (i+1 == max_outiters); i=i+1; end
ttime = toc;

%% Certificate of infeasibility
tau = abs(u(end));
kap = abs(v(end)) / (sc_b * sc_c * scale);

y = u(1:m) / tau;
x = u(m+1:m+n) / tau;
s = v(m+1:m+n) / tau;

if (tau > undet_tol)
    status = 'solved';
else
    y = nan(m,1);
    x = nan(n,1);
    s = nan(n,1);
    
    y_h = u(1:m);
    x_h = u(m+1:m+n);
    s_h = v(m+1:m+n);
    if norm((u+ut)/2)<=2*(undet_tol*sqrt(l))
        status = 'undetermined';
    elseif data.c'*x_h < data.b'*y_h
        status = 'infeasible';
        y = y_h * scale * sc_b * sc_c /(data.b'*y_h);
        s = s_h * scale * sc_b * sc_c /(data.b'*y_h);
        x = -x_h * scale * sc_b * sc_c /(data.c'*x_h);
    else
        status = 'unbounded';
        y = y_h * scale * sc_b * sc_c /(data.b'*y_h);
        s = s_h * scale * sc_b * sc_c /(data.b'*y_h);
    end
end

info.status    = status;
info.outiter  = i; 
info.iter = k;

info.resPri    = err_pri;
info.resDual   = err_dual;
info.relGap    = gap;
info.time      = ttime; 

if (normalize)
    x = x ./ (E * sc_b);
    y = y ./ (D * sc_c);   
    s = s .* (E / (sc_c * scale));
end

info.pobj    = data.c'*x; 


end