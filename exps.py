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



def calc_logsumexp(x, denom):
    return torch.exp(x - torch.logsumexp(denom, dim=0))

def uncond_score(x, t, m1, m2, Gamma_t, label=None, omega=None):
    exp_neg_t = torch.exp(-t)
    exp_neg_m1m2 = torch.exp(-m1 * m2 * exp_neg_t**2 / Gamma_t)
    exp_pos_m1m2 = torch.exp(m1 * m2 * exp_neg_t**2 / Gamma_t)
    x_dot_m1_plus_m2 = torch.dot(x, (m1 + m2))
    x_dot_m1_minus_m2 = torch.dot(x, (m1 - m2))

    arg1 = torch.log(exp_neg_m1m2) + x_dot_m1_plus_m2 * exp_neg_t / Gamma_t
    arg2 = torch.log(exp_neg_m1m2) - x_dot_m1_plus_m2 * exp_neg_t / Gamma_t
    arg3 = torch.log(exp_pos_m1m2) + x_dot_m1_minus_m2 * exp_neg_t / Gamma_t
    arg4 = torch.log(exp_pos_m1m2) - x_dot_m1_minus_m2 * exp_neg_t / Gamma_t

    denom = torch.stack([arg1, arg2, arg3, arg4], dim=0)
    l11 = (m1 + m2) * calc_logsumexp(arg1, denom)
    l12 = -(m1 + m2) * calc_logsumexp(arg2, denom)
    l21 = (m1 - m2) * calc_logsumexp(arg3, denom)
    l22 = - (m1 - m2) * calc_logsumexp(arg4, denom)

    S_i_t = (-x / Gamma_t) + (exp_neg_t / Gamma_t) * (l11 + l12 + l21 + l22)
    return S_i_t

def cond_score(x, t, m1, m2, Gamma_t, label=0, omega=None):
    score = (-x +((2*(label<2)-1) * m1 + (2*(label%2)-1) * m2) * torch.exp(-t)) / Gamma_t
    return score

def cfg(x, t, m1, m2, Gamma_t, omega, label=0):
    uncond_score_ = uncond_score(x, t, m1, m2, Gamma_t, label)
    cond_score_ = cond_score(x, t, m1, m2, Gamma_t, label)
    return (1+omega) * cond_score_ - omega * uncond_score_


def get_vf(m1, m2, score_func, t, dt, sigma, n=10, min=-5, max=5, label=0, plot=False, omega=None):
    t, sigma = torch.tensor(t), torch.tensor(sigma)
    Gamma_t = sigma**2*torch.exp(-2*t) + dt

    xx, yy = torch.meshgrid(
    [torch.linspace(min, max, steps=n)]*2, indexing='ij')
    inp = torch.stack((xx, yy), dim=-1).reshape(-1, 2)

    conc = []
    for inp_ in inp:
        conc.append(score_func(inp_, t, m1, m2, Gamma_t, label=label, omega=omega))

    conc = torch.stack(conc).reshape(n, n, 2).detach().numpy()
    inp = inp.reshape(n, n, 2).detach().numpy()
    
    if plot:
        plt.figure(figsize=(8, 8))
        plt.quiver(inp[:,:,0],inp[:,:,1],conc[:,:,0],conc[:,:,1])
        for plots in [m1+m2, m1-m2, -m1+m2, -m1-m2]:
            plt.scatter(plots[0], plots[1], color='r', s=100)
        plt.show()
    return inp, conc

def kl_div(sample1, sample2):
    mean1, std1 = sample1.mean(dim=0), sample1.std(dim=0)
    mean2, std2 = sample2.mean(dim=0), sample2.std(dim=0)

    var1, var2 = torch.pow(std1, 2), torch.pow(std2, 2)
    kl = 0.5 * torch.sum(torch.pow(mean1 - mean2, 2) / var2 +
                         var1 / var2 - 1.0 - torch.log10(var1) + torch.log10(var2))
    return kl.sum()

def classic_experiment(sigma, omega, seed=12345):
    m1 = torch.tensor([5.,0.])
    m2 = torch.tensor([0., 5])
    t = .1
    dt = 0.05
    min_val = -7
    max_val = 7
    finish_time = 4.

    num_samples = 200
    mean, cov = torch.tensor([0., 0.]), torch.tensor([[1., 0.], [0., 1.]])
    distribution = torch.distributions.MultivariateNormal(mean, cov)
    all_samples = distribution.sample((num_samples,))


    # run = wandb.init(
    #     project="diff_theory",
    #     config={
    #     "sigma": sigma,
    #     "omega": omega,
    #     }
    #     )


    
    images_new, x_prev1, x_prev2, x_prev3, kls, js = [], all_samples, all_samples, all_samples, [],[]
    job_id = get_job_id()
    
    # Define the mean vector
    mean_vector = torch.cat((m1-m2), axis=0)
    cov_matrix = torch.eye(2)
    distribution = torch.distributions.MultivariateNormal(mean_vector, cov_matrix)

    # Sample from the distribution
    target_sample = distribution.sample((num_samples,))

    timesteps = torch.tensor(np.arange(0, finish_time+dt, dt))
    for t in timesteps:
        t_end = torch.tensor(finish_time)
        x_cur1, x_cur2, x_cur3 = [], [], []
        
        for x1, x2, x3 in zip(x_prev1, x_prev2, x_prev3):
            Gamma_t = sigma**2*torch.exp(-2*(t_end-t)) + dt
            # score1 = cond_score(x1, t_end-t, m1, m2, Gamma_t, label=0)
            # score2 = uncond_score(x2, t_end-t, m1, m2, Gamma_t)
            score3 = cfg(x3, t_end-t, m1, m2, Gamma_t, omega=omega, label=0)
            eta = torch.randn_like(x1) * torch.sqrt(torch.tensor(2*dt))
            # x_new1 = x1 + 2 * dt * score1 + eta
            # x_new2 = x2 + 2 * dt * score2 + eta
            x_new3 = x3 + 2 * dt * score3 + eta
            # x_cur1.append(x_new1)
            # x_cur2.append(x_new2)
            x_cur3.append(x_new3)
        # x_cur1 = torch.stack(x_cur1)
        # x_prev1 = x_cur1
        # x_cur2 = torch.stack(x_cur2)
        # x_prev2 = x_cur2
        x_cur3 = torch.stack(x_cur3)
        x_prev3 = x_cur3    
        
    kls.append(kl_div(x_cur3, target_sample))
    js.append(1/2*(kl_div(x_cur3, target_sample) + kl_div(target_sample, x_cur3)))
        
        
    #     fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Create 3 subplots side by side
    #     for ax, c in zip(axes, [x_cur1, x_cur2, x_cur3]):
    #         ax.set_title(['Conditional vec. field, t= %.2f' % (t_end-t), 'Unconditional vec. field, t= %.2f' % (t_end-t), 'CFG vec. field, t= %.2f' % (t_end-t)][axes.tolist().index(ax)])
    #         ax.scatter(c[:,0].detach().numpy(), c[:,1].detach().numpy(), alpha=0.6)
    #         for plots in [m1+m2, m1-m2, -m1+m2, -m1-m2]:
    #             ax.scatter(plots[0], plots[1], color='black', s=25)
    #             ax.set_xlim(min_val, 10)
    #             ax.set_ylim(-10, max_val)
    #             # ax.set_xlim(min_val, max_val)
    #             # ax.set_ylim(min_val, max_val)
    #     plt.suptitle("$\omega$ = %.2f" % omega+", and $\sigma$ = %.2f" % sigma)
    #     plt_path = f'/checkpoint/krunolp/diffusion/ldm/temp/'+str(job_id)+'/temp_{(t_end-t).item():.2f}.png'

    #     plt.savefig(plt_path)
    #     plt.close()
    #     images_new.append(imageio.imread(plt_path))


    # imageio.mimsave('/private/home/krunolp/ldm/all_plots/'+str(job_id)+'/overshoot.gif', images_new, fps=20)
    
    
    # plt.figure()
    # plt.scatter(x_cur3[:,0].detach().numpy(), x_cur3[:,1].detach().numpy(), alpha=0.6)
    # for plots in [m1+m2, m1-m2, -m1+m2, -m1-m2]:
    #     plt.scatter(plots[0], plots[1], color='r', s=20)
    
    # plt.suptitle("Omega: %.2f" % omega + ", Sigma: %.2f" % sigma)
    # plt_path = f'temp_plots/temp_{t.item():.2f}.png'
    # plt.xlim(min_val, max_val)
    # plt.ylim(min_val, max_val)
    # plt.savefig('/private/home/krunolp/ldm/all_plots/'+str(job_id)+'.png')
    
    
    # run.log({"epoch": epoch})

    

        

def main():
    sigmas = [5.]
    omegas = [5., 2.5, 7.5, 10., 1., 15., 25., 0.5, 0.1, 0.01]
    
    executor = submitit.AutoExecutor(folder="experiments/logs/log_test_stand")
    executor.update_parameters(timeout_min=int(2*60), slurm_partition="scavenge")

    jobs = []
    with executor.batch():
        for sigma, omega in list(itertools.product(sigmas, omegas)):
            job = executor.submit(classic_experiment, sigma, omega)
            jobs.append(job)


if __name__ == '__main__':
    main()
    