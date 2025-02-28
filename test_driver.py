import os
import csv
import time

os.environ["RAY_DEDUP_LOGS"] = "0" # Remove dedup log

import ray
import dill
import hydra
import torch
import numpy as np

from test_parameter import *
from test_worker import TestWorker
from diffusion_policy.workspace.base_workspace import BaseWorkspace


def run_test():
    device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    checkpoint = os.path.join(run_path, 'checkpoints', checkpoint_name)
    output_dir = os.path.join(run_path, 'inference')
    csv_path = os.path.join(run_path, f'test_results_{time.strftime("%d_%m_%Y_%H_%M_%S")}.csv')
    
    ## Load Diffusion Model
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    meta_agents = [Runner.remote(i) for i in range(NUM_META_AGENT)]

    curr_test = 0
    completed_tests = 0

    dist_history = []
    success_dist_history = []
    successes = 0

    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        job_list.append(meta_agent.job.remote(policy, curr_test))
        curr_test += 1

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode Number', 'Travel Distance', 'Success'])

        try:
            while completed_tests < curr_test:
                done_id, job_list = ray.wait(job_list)
                done_jobs = ray.get(done_id)

                for job in done_jobs:
                    completed_tests += 1
                    metrics, info = job

                    writer.writerow([info['episode_number'], metrics['travel_dist'], metrics['success_rate']])

                    if metrics['success_rate']:
                        successes += 1
                        success_dist_history.append(metrics['travel_dist'])
                    dist_history.append(metrics['travel_dist'])
                    
                if curr_test < NUM_TEST:
                    job_list.append(meta_agents[info['id']].job.remote(policy, curr_test))
                    curr_test += 1

            print('|#Total test:', NUM_TEST)
            print('|#Length of dist_history:', len(dist_history))
            print('|#Average length:', np.array(dist_history).mean())
            print('|#Length std:', np.array(dist_history).std())
            print('|#Successes:', successes)
            print('|#Success Rate:', successes / NUM_TEST)
            print('|#Length of success_dist_history:', len(success_dist_history))
            print('|#Average Success length:', np.array(success_dist_history).mean())
            print('|#Success Length std:', np.array(success_dist_history).std())

        except KeyboardInterrupt:
            print("CTRL_C pressed. Killing remote workers")
            for a in meta_agents:
                ray.kill(a)

@ray.remote(num_cpus=1, num_gpus=NUM_GPU/NUM_META_AGENT)
class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')

    def do_job(self, episode_number, policy):
        worker = TestWorker(self.meta_agent_id, policy, episode_number, device=self.device, save_image=SAVE_GIFS)
        worker.run_episode()

        perf_metrics = worker.perf_metrics
        del worker
        return perf_metrics

    def job(self, policy, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))

        metrics = self.do_job(episode_number, policy)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }

        return metrics, info

if __name__ == '__main__':
    ray.init()
    for i in range(NUM_RUN):
        run_test()