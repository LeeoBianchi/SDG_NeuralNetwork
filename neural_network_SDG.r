#Neural network
#h = ReLU
h = function(x)
{
  ifelse(x>0, x, 0)
}
dh = function(x)
{
  ifelse(x>0, 1, 0)
}
g = function(x)
{
  x
}
#Load data
M = 50 #network's width
set.seed(43253)
dat = read.table("functionestimationnn.ascii",header=F)
X = dat[,1]
X = cbind(1, X) #Has the effect of including the intercept into the alpha vector
y = dat[,2]
fGTs = dat[,3]
d.nn = data.frame(y=y,x=X)
p = ncol(X)
#initialize param with random values
beta = runif(M)
beta0 = runif(1) #I add the bias term
alpha = matrix(runif(M*p),ncol=p)

y<-d.nn$y
X<-as.matrix(d.nn[,-1])
N.epo = 20
lambda = 0
Qvec = NULL
Q.rep = rep(NA,N.epo)
TE.s = NULL #test error between f_est e GT after each epoch
for(epo in 1:N.epo)
{
  m = 50
  #Qvec = NULL
  itvec = NULL
  for(it in 0:10000)
  {
    #Subsampling
    ind = sample(1:n,m,replace=FALSE)
    X.sub = X[ind,,drop=FALSE]
    y.sub = y[ind]
    #Forwards
    zz = X.sub%*%t(alpha)
    z = h(zz)
    TT = z%*%matrix(beta,ncol=1)
    eta = g(TT + beta0)
    res = as.vector(y.sub-eta)
    
    #Backwards
    df.beta = z
    dQ.beta = rep(0,M)
    for(k in 1:M)
     dQ.beta[k] = -2*sum(res*df.beta[,k])+2*lambda*beta[k]
    dQ.alpha = matrix(nrow=M,ncol=p)
    for(k in 1:M)
    for(j in 1:p)
    {
      df.alpha = beta[k]*dh(zz[,k])*X.sub[,j]
      dQ.alpha[k,j] = -2*sum(res*df.alpha)
    }
    dQ.alpha
    dQ.alpha = dQ.alpha + 2*lambda*alpha

    #gamma = (1+0.5*rep/N.rep)/10000
    gamma = 1/(10000 + epo)
    beta0 = beta0 + 2* gamma*sum(res)
    beta = beta-gamma*dQ.beta
    alpha = alpha-gamma*dQ.alpha
    
    if(it%%1000==0)   #Compute full objective function
    {
      #Forwards
      zz = X%*%t(alpha)
      z = h(zz)
      TT = z%*%matrix(beta,ncol=1)
      eta = g(TT + beta0)
      res = as.vector(y-eta)
      Q = sum(res^2)+lambda*sum(alpha^2)+lambda*sum(beta^2)
      Qvec = c(Qvec,Q)
      itvec = c(itvec,it)
      show(c(it,Q,beta,beta0))
    }
  }
zz = X%*%t(alpha)
z = h(zz)
TT = z%*%matrix(beta,ncol=1)
eta = g(TT + beta0)
res = as.vector(fGTs-eta)
TE = sum(res^2)
TE.s = c(TE.s, TE)
Q.rep[epo] = Q
}
#Plot value of Q as function of iteration
plot(10:length(Qvec[-1]),Qvec[10:length(Qvec[-1])],type="l",xlab="Iteration (E+03)",ylab="Q")
plot(1:length(TE.s[-1]),TE.s[1:length(TE.s[-1])],type="l",xlab="Epoch",ylab="Test error")
dev.off()
