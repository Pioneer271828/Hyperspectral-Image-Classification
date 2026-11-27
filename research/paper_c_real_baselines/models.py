"""Nine-parameter logistic GD and a deterministic Euclidean-centroid reference."""
from time import perf_counter

import numpy as np

from c8_bridge import read,write,sha,canonical


def sigmoid(z):
    z=np.asarray(z,dtype=np.float64);result=np.empty_like(z);positive=z>=0
    result[positive]=1/(1+np.exp(-z[positive]));e=np.exp(z[~positive]);result[~positive]=e/(1+e)
    return result


def initial(seed):return np.r_[np.random.default_rng(seed).normal(0,.01,8),0.0]


def validate(X,y=None):
    X=np.asarray(X,dtype=np.float64)
    if X.ndim!=2 or X.shape[1]!=8 or not len(X) or not np.isfinite(X).all():raise ValueError('Finite N-by8 matrix required')
    if y is not None:
        y=np.asarray(y,dtype=np.float64)
        if y.shape!=(len(X),) or not np.isin(y,[0,1]).all():raise ValueError('Binary common targets required')
    return X,y


def objective_gradient(theta,X,y,l2=.01):
    X,y=validate(X,y);theta=np.asarray(theta,dtype=np.float64)
    if theta.shape!=(9,) or not np.isfinite(theta).all() or l2<0:raise ValueError('Finite9-parameter state and nonnegative penalty required')
    logits=X@theta[:8]+theta[8];residual=sigmoid(logits)-y
    loss=np.mean(np.logaddexp(0,logits)-y*logits)+.5*l2*(theta[:8]@theta[:8])
    gradient=np.r_[X.T@residual/len(X)+l2*theta[:8],residual.mean()]
    return float(loss),gradient


def fit(X,y,*,seed,steps,start=None,checkpoint_dir=None):
    X,y=validate(X,y)
    if type(steps)!=int or not 0<steps<=128:raise ValueError('Registered short-step limit')
    theta=initial(seed) if start is None else np.array(start,dtype=np.float64,copy=True);initial_sha=sha(theta.tobytes());trace=[];began=perf_counter();last=objective_gradient(theta,X,y)[0]
    trace.append({'step':0,'regularized_training_objective':last})
    for step in range(1,steps+1):
        _,gradient=objective_gradient(theta,X,y);theta-=.5*gradient
        if step%32==0 or step==steps:
            loss,_=objective_gradient(theta,X,y)
            if loss>last+1e-12 or not np.isfinite(theta).all():raise ValueError('Fixed-budget numerical descent failed; no target-driven extension')
            last=loss;trace.append({'step':step,'regularized_training_objective':loss})
            if checkpoint_dir is not None:
                path=checkpoint_dir/f'step-{step:03d}.json';write(path,{'step':step,'theta':theta.tolist(),'seed':seed,'learning_rate':.5,'l2':.01,'optimizer':'full_batch_GD_no_momentum'})
                restored=np.asarray(read(path)['theta'],dtype=np.float64)
                if not np.array_equal(restored,theta):raise ValueError('Checkpoint float64 reload differs')
                theta=restored
    return {'kind':'logistic','theta':theta.tolist(),'seed':seed,'steps':steps,'initial_theta_sha256':initial_sha,'terminal_theta_sha256':sha(theta.tobytes()),'training_trace':trace,'fit_seconds':perf_counter()-began,'optimizer_updates':steps,'parameter_count':9}


def centroid(X,y):
    X,y=validate(X,y)
    if set(y)!={0,1}:raise ValueError('Both common classes required')
    began=perf_counter();means=np.stack([X[y==c].mean(axis=0) for c in (0,1)]);theta=np.r_[2*(means[1]-means[0]),means[0]@means[0]-means[1]@means[1]]
    return {'kind':'centroid','theta':theta.tolist(),'class_means':means.tolist(),'seed':None,'steps':0,'fit_seconds':perf_counter()-began,'optimizer_updates':0,'prototype_entries':16,'derived_linear_parameters':9}


def predict(model,X):
    X,_=validate(X);theta=np.asarray(model['theta'],dtype=np.float64)
    if theta.shape!=(9,):raise ValueError('Registered linear endpoint required')
    logits=X@theta[:8]+theta[8];return logits,sigmoid(logits),(logits>0).astype(np.int64)
