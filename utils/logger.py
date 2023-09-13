from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import json
import os
import shutil
import pickle
import moviepy.editor as mpy
import torch
import torchvision
import numpy as np
from termcolor import colored


class Logger:

    def __init__(self, log_dir, save_tb, n_logged_samples=10, summary_writer=None):
        self._log_dir = log_dir
        print('########################')
        print('logging outputs to ', log_dir)
        print('########################')
        self._n_logged_samples = n_logged_samples
        if save_tb:
            self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)
        else:
            self._summ_writer = None

    def log_scalar(self, name, scalar, step_):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step_)

    def log_scalars(self, scalar_dict, step):
        if self._summ_writer is not None:
            for key, value in scalar_dict.items():
                print('{} : {}'.format(key, value))
                self.log_scalar(key, value, step)
            self.dump_scalars_to_pickle(scalar_dict, step)

    def log_videos(self, videos, step, max_videos_to_save=1, fps=20, video_title='video'):
        if self._summ_writer is not None:
            # max rollout length
            max_videos_to_save = np.min([max_videos_to_save, videos.shape[0]])
            max_length = videos[0].shape[0]
            for i in range(max_videos_to_save):
                if videos[i].shape[0]>max_length:
                    max_length = videos[i].shape[0]

            # pad rollouts to all be same length
            for i in range(max_videos_to_save):
                if videos[i].shape[0]<max_length:
                    padding = np.tile([videos[i][-1]], (max_length-videos[i].shape[0],1,1,1))
                    videos[i] = np.concatenate([videos[i], padding], 0)

                clip = mpy.ImageSequenceClip(list(videos[i]), fps=fps)
                new_video_title = video_title+'{}_{}'.format(step, i) + '.gif'
                filename = os.path.join(self._log_dir, new_video_title)
                clip.write_gif(filename, fps=fps)

    def log_images(self, key, image, step):
        if self._summ_writer is not None:
            assert image.dim() == 3
            grid = torchvision.utils.make_grid(image.unsqueeze(1))
            self._summ_writer.add_image(key, grid, step)

    def dump_scalars_to_pickle(self, metrics, step, log_title=None):
        log_path = os.path.join(self._log_dir, "scalar_data.pkl" if log_title is None else log_title)
        with open(log_path, 'ab') as f:
            pickle.dump({'step': step, **dict(metrics)}, f)

    def flush(self):
        self._summ_writer.flush()
