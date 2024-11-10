import numpy as np
import wandb
import submitit
import argparse
import itertools
import random
import torch
import os

import torch
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from warnings import simplefilter


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.


# ignore all warnings
simplefilter(action='ignore')
def get_job_id():
    if "SLURM_ARRAY_JOB_ID" in os.environ:
        return os.environ["SLURM_ARRAY_JOB_ID"] + "_" + os.environ["SLURM_ARRAY_TASK_ID"]
    if "SLURM_JOB_ID" in os.environ:
        return os.environ["SLURM_JOB_ID"]
    return None


def ff(x,omega, alpha):
    if alpha==0:
        return omega
    else:
        return omega/(x**(-alpha)+1.e-6)


def calc_score(itau, dt, itfms, omega, q, sigma2, alpha):
    expt = np.exp(-dt*(itfms-itau))
    exp2t = expt**2
    gammat = (sigma2*exp2t+1-exp2t)
    tanh = np.tanh(q*expt/gammat)
    nonlinear = (expt/gammat)*(1+(1 - np.tanh(q*expt/gammat)) *
                               ff((expt/gammat)*(1-tanh), omega, alpha))  # K: fixed, divided by gammat
    # Norm of S_full-S_class # this is what is added in cfg
    normdiff = (expt/gammat)*(1-tanh)
    return nonlinear, normdiff

def init_backward():
    return np.random.normal(0,1)
    
def calc_traj_back_exact(qinit,nt,dt,itfms,omega,sigma2, alpha):
    qback = np.zeros(nt)
    time = np.zeros(nt)
    normd = np.zeros(nt)
    qback[0] = qinit
    q = qback[0]
    time[0] = 0
    for it in np.arange(nt-1):
        expt = np.exp(-dt*(itfms-it))
        exp2t = expt**2
        gammat = (sigma2*exp2t+1-exp2t)
        eta = np.random.normal(0, np.sqrt(2*dt))
        sc, normdiff = calc_score(it, dt, itfms, omega, q, sigma2, alpha)
        q = q-dt*q*(2/gammat-1)+2*dt*sc+eta
        qback[it+1] = q
        time[it+1] = it*dt
        normd[it] = normdiff
    return time, qback, normd


def kl_div(sample1, sample2):
    mean1, std1 = np.mean(sample1, axis=0), np.std(sample1, axis=0)
    mean2, std2 = np.mean(sample2, axis=0), np.std(sample2, axis=0)

    var1, var2 = np.power(std1, 2), np.power(std2, 2)
    kl = 0.5 * np.sum(np.power(mean1 - mean2, 2) / var2 +
                         var1 / var2 - 1.0 - np.log10(var1) + np.log10(var2))
    return np.sum(kl)


def classic_experiment(dim, seed=12345):
    nsample=100
    dt=.01
    nt=800 #final time is nt*dt
    sigma2=1.
    alpha = -0.5

    tspec=(1/2)*np.log(dim)
    ns=int(tspec/dt)
    print("Speciation time is: ", ns)
    #ns=200 #speciation time is t_s=ns*dt=(1/2) log (d)
    itfms=nt-ns
    nomega=5
    omegag=np.zeros(nomega)
    qtraj=np.zeros((nomega,nsample,nt))
    normstat=np.zeros((nomega,nsample,nt))
    for ijk in np.arange(nsample):
        qinit=init_backward()
        for iomega in np.arange(nomega):
            omega=4*iomega
            omegag[iomega]=omega
            time, qtraj[iomega,ijk,:],normstat[iomega,ijk,:]=calc_traj_back_exact(qinit,nt,dt,itfms,omega,sigma2, alpha)

    job_id = get_job_id()

    run = wandb.init(
        project="new_diff",
        config={
        "dim": dim,
        "job_id": job_id,
        "dt": dt,
        }
        )


    
    x_prev = all_samples
    
    # Define the mean vector
    mean_vector = m1-m2
    cov_matrix = torch.eye(2) * sigma
    distribution = torch.distributions.MultivariateNormal(mean_vector, cov_matrix)

    # Sample from the distribution
    target_sample = distribution.sample((num_samples,))

    timesteps = torch.tensor(np.arange(0, finish_time+dt, dt))
    for omega in torch.cat((torch.zeros(1), torch.logspace(-2, 2.5, 200, base=10, dtype=None, device=None, requires_grad=False))):
        for t in timesteps:
            t_end = torch.tensor(finish_time)
            x_cur = []
            
            for x in x_prev:
                Gamma_t = sigma**2*torch.exp(-2*(t_end-t)) + dt
                score = cfg(x, t_end-t, m1, m2, Gamma_t, omega=omega, label=0)
                eta = torch.randn_like(x) * torch.sqrt(torch.tensor(2*dt))
                x_new = x + 2 * dt * score + eta
                x_cur.append(x_new)

            x_cur = torch.stack(x_cur)
            x_prev = x_cur
        
        # reverse: KL(q,p) = KL(samples, true_samples) = KL(x_cur, target_sample) = kl_for
        # forward: KL(p,q) = KL(true_samples, samples) = kl_back
        kl_for = kl_div(x_cur, target_sample)
        kl_back = kl_div(target_sample, x_cur)
        js = (kl_for + kl_back) / 2
        
        
        run.log({"omega": omega, "kl_forward": kl_for, "kl_backward": kl_back, "js": js, "sigma": sigma})

    

        

def main():
    sigmas = torch.logspace(-1, 2, 30, base=10, dtype=None, device=None, requires_grad=False)
    
    executor = submitit.AutoExecutor(folder="experiments/logs/log_test_stand")
    executor.update_parameters(timeout_min=int(4*60), slurm_partition="scavenge")

    jobs = []
    with executor.batch():
        for sigma in sigmas:
            job = executor.submit(classic_experiment, sigma)
            jobs.append(job)


if __name__ == '__main__':
    main()
    