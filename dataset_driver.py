import os
import csv
import time

os.environ["RAY_DEDUP_LOGS"] = "0" # Remove dedup log

import ray
import zarr
import torch
import numpy as np

from dataset_parameter import *
from dataset_worker import DatasetWorker


def run_test():
    device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    csv_path = f'{dataset_path}/train_results_ {time.strftime("%d_%m_%Y_%H_%M_%S")}.csv'

    meta_agents = [Runner.remote(i) for i in range(NUM_META_AGENT)]

    curr_test = 0
    completed_tests = 0
    num_failed = 0

    dist_history = []

    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        job_list.append(meta_agent.job.remote(curr_test))
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
                    episode_data, metrics, info = job

                    writer.writerow([info["episode_number"], metrics['travel_dist'], metrics['success_rate']])
                    
                    if metrics['success_rate'] != True:
                        print(f'Episode {info["episode_number"]} failed')
                        num_failed += 1
                        continue  
                    dist_history.append(metrics['travel_dist'])

                    # save data using Zarr
                    zarr_path = f'{dataset_path}/data.zarr'
                    zarr_group = zarr.open_group(zarr_path, mode='a')

                    if DATA_TYPE == 'node':
                        if 'data' not in zarr_group:
                            data_group = zarr_group.create_group('data')
                            data_group.create_dataset('node_inputs', data=np.array(episode_data['node_inputs']), chunks=True)
                            data_group.create_dataset('node_padding_mask', data=np.array(episode_data['node_padding_mask']), chunks=True)
                            data_group.create_dataset('edge_mask', data=np.array(episode_data['edge_mask']), chunks=True)
                            data_group.create_dataset('current_index', data=np.array(episode_data['current_index']), chunks=True)
                            data_group.create_dataset('current_edge', data=np.array(episode_data['current_edge']), chunks=True)
                            data_group.create_dataset('edge_padding_mask', data=np.array(episode_data['edge_padding_mask']), chunks=True)
                            data_group.create_dataset('action', data=np.array(episode_data['action']), chunks=True)
                        else:
                            data_group = zarr_group['data']
                            data_group['node_inputs'].append(np.array(episode_data['node_inputs']))
                            data_group['node_padding_mask'].append(np.array(episode_data['node_padding_mask']))
                            data_group['edge_mask'].append(np.array(episode_data['edge_mask']))
                            data_group['current_index'].append(np.array(episode_data['current_index']))
                            data_group['current_edge'].append(np.array(episode_data['current_edge']))
                            data_group['edge_padding_mask'].append(np.array(episode_data['edge_padding_mask']))
                            data_group['action'].append(np.array(episode_data['action']))

                        if 'meta' not in zarr_group:
                            meta_group = zarr_group.create_group('meta')
                            meta_group.create_dataset('episode_ends', data=np.array(episode_data['episode_end']), chunks=True)
                        else:
                            meta_group = zarr_group['meta']
                            meta_group['episode_ends'].append(np.array(episode_data['episode_end'] + meta_group['episode_ends'][-1]))
                    elif DATA_TYPE == 'map':
                        if 'data' not in zarr_group:
                            data_group = zarr_group.create_group('data')
                            data_group.create_dataset('img', data=np.array(episode_data['img']), chunks=True)
                            data_group.create_dataset('state', data=np.array(episode_data['state']), chunks=True)
                            data_group.create_dataset('action', data=np.array(episode_data['action']), chunks=True)
                        else:
                            data_group = zarr_group['data']
                            data_group['img'].append(np.array(episode_data['img']))
                            data_group['state'].append(np.array(episode_data['state']))
                            data_group['action'].append(np.array(episode_data['action']))

                        if 'meta' not in zarr_group:
                            meta_group = zarr_group.create_group('meta')
                            meta_group.create_dataset('episode_ends', data=np.array(episode_data['episode_end']), chunks=True)
                        else:
                            meta_group = zarr_group['meta']
                            meta_group['episode_ends'].append(np.array(episode_data['episode_end'] + meta_group['episode_ends'][-1]))
                    else:
                        raise ValueError('Invalid dataset type, check dataset_parameter.py')
            
                if curr_test < NUM_EPISODES:
                    job_list.append(meta_agents[info['id']].job.remote(curr_test))
                    curr_test += 1

            print('|#Total test:', NUM_EPISODES)
            print('|#Average length:', np.array(dist_history).mean())
            print('|#Length std:', np.array(dist_history).std())
            print('|#Failed:', num_failed)

        except KeyboardInterrupt:
            print("CTRL_C pressed. Killing remote workers")
            for a in meta_agents:
                ray.kill(a)


@ray.remote(num_cpus=1, num_gpus=NUM_GPU/NUM_META_AGENT)
class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        
    def do_job(self, episode_number):
        worker = DatasetWorker(self.meta_agent_id, episode_number, device=self.device, save_image=SAVE_GIFS, greedy=True)
        worker.run_episode()

        episode_data = worker.episode_data
        perf_metrics = worker.perf_metrics
        return episode_data, perf_metrics

    def job(self, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))

        episode_data, metrics = self.do_job(episode_number)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }

        return episode_data,metrics, info


if __name__ == '__main__':
    ray.init()
    for i in range(NUM_RUN):
        run_test()