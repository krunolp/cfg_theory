import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import itertools
from tqdm.auto import tqdm
import wandb
import submitit
import random
import os
import torch.nn.functional as F


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False -> one of these causes issues
    # torch.backends.cudnn.deterministic = True
from warnings import simplefilter

def get_job_id():
    if "SLURM_ARRAY_JOB_ID" in os.environ:
        return os.environ["SLURM_ARRAY_JOB_ID"] + "_" + os.environ["SLURM_ARRAY_TASK_ID"]
    if "SLURM_JOB_ID" in os.environ:
        return os.environ["SLURM_JOB_ID"]
    return None

# ignore all warnings
simplefilter(action='ignore')

device = 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
TIME_STEPS = 250
BETA = torch.tensor([0.02]).to(device)

def do_diffusion(data, steps=TIME_STEPS, beta=BETA, d=None):
    # starting from t=0 (i.e., the dataset)
    # returns a list of q(x(t)) and x(t)

    distributions, samples = [None], [data]
    xt = data
    for t in range(steps):
        q = torch.distributions.Normal(
            torch.ones(d)*torch.sqrt(1 - beta) * xt,
            torch.ones(d)*torch.sqrt(beta),
        )
        xt = q.sample()

        distributions.append(q)
        samples.append(xt)

    return distributions, samples


def one_hot_encode(labels, num_classes):
    return F.one_hot(labels, num_classes).float()

def compute_loss(forward_distributions, forward_samples, mean_model, var_model, labels, d, cfg, cfg_prob):
    # here we compute the loss in equation 3
    # forward = q , reverse = p

    # loss for x(T)
    p = torch.distributions.Normal(
    torch.zeros(forward_samples[0].shape),
    torch.ones(forward_samples[0].shape)
    )
    loss = -p.log_prob(forward_samples[-1]).mean()

    for t in range(1, len(forward_samples)):
        xt = forward_samples[t]         # x(t)
        xprev = forward_samples[t - 1]  # x(t-1)
        q = forward_distributions[t]    # q( x(t) | x(t-1) )

        # normalize t between 0 and 1 and add it as a new column
        # to the inputs of the mu and sigma networks
        xin = torch.cat(
            (xt, (t / len(forward_samples)) * torch.ones(xt.shape[0], d).to(device)),
            dim=1
        )
        # compute p( x(t-1) | x(t) ) as equation 1
        class_info = one_hot_encode(labels.long(), 2).to(device).squeeze()
        
        if torch.rand(1).item() < cfg_prob and cfg:
            class_info *= 0
            

        inputs = torch.cat((xin, class_info), dim=1).to(device)

        mu = mean_model(inputs)
        sigma = var_model(inputs)
        p = torch.distributions.Normal(mu, sigma)

        # add a term to the loss
        loss -= torch.mean(p.log_prob(xprev.to(device)))
        loss += torch.mean(q.log_prob(xt)).to(device)

    return loss / len(forward_samples)


def classic_experiment(lr, wd, d, n, model, cfg_prob, seed=1234):
    torch.manual_seed(seed)
    job_id = get_job_id()

    cfg = True

    mean1 = torch.tensor([4.0] * d)
    mean2 = torch.tensor([-4.0] * d)
    std = torch.tensor([1.0] * d)

    gaussian1 = torch.distributions.MultivariateNormal(mean1, torch.diag(std))
    gaussian2 = torch.distributions.MultivariateNormal(mean2, torch.diag(std))

    samples1 = gaussian1.sample((n,))
    samples2 = gaussian2.sample((n,))

    dataset = torch.cat((samples1, samples2), dim=0)
    labels = torch.cat((torch.zeros(n), torch.ones(n)))

    indices = torch.randperm(dataset.size(0))
    dataset = dataset[indices]
    labels = labels[indices]

    dataset = dataset.to(device)
    
    num_classes = 2

    if model == 0:
        mean_model = torch.nn.Sequential(
            torch.nn.Linear(2*d + num_classes, 4), 
            torch.nn.ReLU(),
            torch.nn.Linear(4, d)
        )
        var_model = torch.nn.Sequential(
            torch.nn.Linear(2*d + num_classes, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, d),
            torch.nn.Softplus()
        )
    elif model == 1:
        mean_model = torch.nn.Sequential(
            torch.nn.Linear(2 * d + num_classes, 16),  
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),               
            torch.nn.ReLU(),
            torch.nn.Linear(8, d)
        )
        var_model = torch.nn.Sequential(
            torch.nn.Linear(2 *d + num_classes, 16),  
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),               
            torch.nn.ReLU(),
            torch.nn.Linear(8, d),
            torch.nn.Softplus()
        )
    else:
        num_classes = 2
        mean_model = torch.nn.Sequential(
            torch.nn.Linear(2 * d + num_classes, 64),  
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),               
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, d)
        )
        var_model = torch.nn.Sequential(
            torch.nn.Linear(2 * d + num_classes, 64),  
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),               
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, d),
            torch.nn.Softplus()
        )
    optim = torch.optim.AdamW(
        itertools.chain(mean_model.parameters(), var_model.parameters()),
        lr=lr, weight_decay=wd,
    )


    run = wandb.init(
        project="diff_exp",
        config={
        "seed": seed,
        "lr": lr,
        "weight_decay": wd,
        "d": d,
        "n": n,
        "job_id": job_id,
        "cfg": cfg,
        "cfg_prob": cfg_prob,
        "model": model,
        }
        )

    loss_history = []
    bar = tqdm(range(1000))

    for e in bar:
        forward_distributions, forward_samples = do_diffusion(dataset, d=d)

        optim.zero_grad()
        loss = compute_loss(
            forward_distributions, forward_samples, mean_model, var_model, labels, d, cfg, cfg_prob
        )
        loss.backward()
        optim.step()

        bar.set_description(f'Loss: {loss.item():.4f}')
        loss_history.append(loss.item())
        run.log({"epoch": e, "loss": loss.item()})

    save_dir = '/checkpoint/krunolp/diffusion/edm/ldm/'
    
    torch.save(mean_model.state_dict(), save_dir+job_id+'_'+'mean_model_weights.pth')
    torch.save(var_model.state_dict(), save_dir+job_id+'_'+'var_model_weights.pth')

def main():
    lrs = [1e-2, 1e-1, 1e-3, 1e-4, 1e-5]
    wds = [1e-6, 1e-5, 1e-4, 1e-7, 0.]
    models = [2, 1]#, 0]
    cfg_probs = [0.1, 0.2, 0.5]
    
    ds = [10]#, 1, 5, 10]
    ns = [20000, 50000]
    
    executor = submitit.AutoExecutor(folder="experiments/logs/log_test_stand")
    executor.update_parameters(timeout_min=int(24*60), slurm_partition="learnlab")

    jobs = []
    with executor.batch():
        for lr, wd, d, n, model, cfg_prob in list(itertools.product(lrs, wds, ds, ns, models, cfg_probs)):
            job = executor.submit(classic_experiment, lr, wd, d, n, model, cfg_prob)
            jobs.append(job)


if __name__ == '__main__':
    main()
    