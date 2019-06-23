import os
import math
import time
import numpy as np
import mxnet as mx
from tqdm import tqdm
from mxnet import gluon
from mxnet import autograd
import mxnet.ndarray as nd
from rank_utils.phrases import random_phrase
from sklearn.metrics import f1_score, accuracy_score

from typing import List
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import string
import random


def id_generator(size=8, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def get_params_count(gluon_mod):
    params = gluon_mod.collect_params()
    n_params = 0
    msg_list = []
    for k in params.keys():
        msg = '{} {}'.format(params[k].name, params[k].shape)
        msg_list.append(msg)
        print(msg)
        unit_count = 1
        for i in params[k].shape:
            unit_count *= i

        n_params += unit_count
    msg = '{} {}'.format('total params', n_params)
    print(msg)
    msg_list.append(msg)
    return n_params, msg_list


def set_wd_mult(block, wd_mult=1):
    params = block.collect_params()
    for k in params.keys():
        params[k].wd_mult = wd_mult


class ModelTrainer:
    def __init__(self,
                 gluon_model, gluon_trainer, model_ctx, train_iter, val_iter, evals_per_epoch=5,
                 external_eval_calls: List[dict] = [],
                 # to support eval metrics that can't use train_iter and val_iter
                 # each external eval call should be a dict, example
                 # {'name': 'ndcg@10',
                 #  'funcs': {
                 #      'train': ftrain, 'val': fval
                 #      }
                 # }
                 early_stopping_enabled=True,
                 early_stopping_metric_name='loss',
                 early_stopping_minimize=True,  # remember to change to False when needed
                 early_stopping_patience=10,
                 early_stopping_min_eval_counter=2,  # 3 evals
                 max_eval_count=None,
                 lr_decay_ratio=0.7,
                 lr_decay_count_max=5,
                 lr_min=1e-7,
                 job_id=None,
                 ):
        self.model_prefix = random_phrase()
        if not job_id:
            job_id = id_generator()
        self.model_prefix = job_id + '_' + self.model_prefix
        self.model_dir = os.path.join(os.path.expanduser('~'), 'model_results', self.model_prefix)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_ctx = model_ctx
        self.mod = gluon_model
        self.trainer = gluon_trainer
        self.train_iter = train_iter
        self.val_iter = val_iter

        self.evals_per_epoch = evals_per_epoch

        self.external_eval_calls = external_eval_calls

        self.metrics = dict()

        self.tv_str = ['train', 'val']
        for eval_name in ['loss']:
            self.metrics[eval_name] = dict()
            for tv in self.tv_str:
                self.metrics[eval_name][tv] = dict(value=[], epoch=[], nbatch=[])

        for ext_eval_entry in self.external_eval_calls:
            eval_name = ext_eval_entry['name']
            self.metrics[eval_name] = dict()
            for tag, func in ext_eval_entry['funcs'].items():
                self.metrics[eval_name][tag] = dict(value=[], epoch=[], nbatch=[])

        self.epoch_counter = 0
        self.current_epoch = 0
        self.current_batch = 0
        self.eval_counter = 0
        self.max_eval_count = max_eval_count

        self.early_stopping_enabled = early_stopping_enabled
        self.early_stopping_metric_name = early_stopping_metric_name
        self.early_stopping_minimize = early_stopping_minimize
        self.early_stopping_min_eval_counter = early_stopping_min_eval_counter
        self.early_stopping_patience = early_stopping_patience
        self.ES = EarlyStopping(mod=self.mod,
                                minimize=self.early_stopping_minimize,
                                patience=self.early_stopping_patience,
                                # restore_path=os.path.join(self.proj_folder, 'model_ES.params')
                                restore_path=os.path.join(self.model_dir,
                                                          '{}_model_ES.params'.format(self.model_prefix)),
                                )
        self.early_stopping_condition_met = False

        self.lr_decay_ratio = lr_decay_ratio
        self.lr_decay_count_max = lr_decay_count_max
        self.lr_decay_count = 0
        self.lr_min = lr_min

        self.lr_history = dict()

        self.model_params_printed = False

        self.log = []

        self.train_throughput = -1
        self.n_batches = len(train_iter)

    def set_checkpoint_nbatches(self):

        if self.evals_per_epoch > 0:
            batches_per_check = math.ceil(self.n_batches / self.evals_per_epoch)
            self.checkpoint_batch_num = [
                batches_per_check * i if batches_per_check * i <= self.n_batches else self.n_batches
                for i in range(1, self.evals_per_epoch + 1)]
        else:
            self.evals_per_epoch = 1
            self.checkpoint_batch_num = [self.n_batches]

    def external_evaluate(self):
        for eval_item in self.external_eval_calls:
            eval_name = eval_item['name']
            for tag, func in eval_item['funcs'].items():
                val = func(self)
                self.metrics[eval_name][tag]['value'].append(val)
                self.metrics[eval_name][tag]['epoch'].append(self.current_epoch)
                self.metrics[eval_name][tag]['nbatch'].append(self.current_batch)

    def print_latest_metrics(self):
        epoch_id = 'E{} B{} C{}:'.format(self.current_epoch, self.current_batch, self.eval_counter)

        msg = ''
        for metric_name in self.metrics:
            tags = list(self.metrics[metric_name].keys())
            tags_str = '/'.join(tags)
            values = ['{:.4f}'.format(self.metrics[metric_name][t]['value'][-1]) for t in tags]
            values_str = '/'.join(values)
            msg += '[{}] {} ({}) '.format(metric_name, values_str, tags_str)

        msg = '{} {}'.format(epoch_id, msg)
        self.log.append(msg)
        print(msg)

    def plot_metrics(self, metric_str):
        assert metric_str in self.metrics
        plt.figure()
        legend_str = []
        for tag in self.metrics[metric_str].keys():
            if metric_str in self.metrics and self.metrics[metric_str][tag]['value']:
                legend_str.append(tag)
                e = np.array(self.metrics[metric_str][tag]['epoch'])
                b = np.array(self.metrics[metric_str][tag]['nbatch'])
                nbatch = e * self.n_batches + b
                vals = np.array(self.metrics[metric_str][tag]['value'])
                plt.plot(nbatch, vals)
        plt.xlabel('nbatch')
        plt.ylabel(metric_str)
        plt.legend(legend_str)
        plt.title(metric_str)
        plt.savefig(os.path.join(self.model_dir, 'metric_plot_{}.png'.format(metric_str)))

    def plot_all_metrics(self):
        for m in self.metrics:
            self.plot_metrics(m)

    def reset_ES(self):
        self.early_stopping_condition_met = False
        self.ES.best_monitored_eval = self.eval_counter
        self.model_params_printed = False

    def train_one_epoch(self, epoch_id=None, progress_bar=False):
        start_time = time.time()
        last_eval_time = start_time
        last_eval_n_sample = 0

        self.set_checkpoint_nbatches()  # set proper checkpoint batch number based on train_iter

        if self.early_stopping_condition_met:
            return

        if epoch_id is None:
            self.current_epoch = self.epoch_counter
            self.epoch_counter += 1
        else:
            self.current_epoch = epoch_id
        n_sample = 0
        nbatch = 0

        if not isinstance(self.train_iter, gluon.data.DataLoader):
            self.train_iter.reset()

        if progress_bar:
            iterator = tqdm(self.train_iter, desc='train_iter')
        else:
            iterator = self.train_iter

        train_losses = []
        for batch in iterator:
            data = [x.as_in_context(self.model_ctx) for x in batch]
            batch_size = data[0].shape[0]
            with autograd.record():
                loss = self.mod(*data)
            loss.backward()
            self.trainer.step(batch_size)

            train_losses.append(np.mean(loss.asnumpy()))
            n_sample += batch_size
            nbatch += 1
            self.current_batch = nbatch

            if not self.model_params_printed:
                n_params, msg_list = get_params_count(self.mod)
                self.log += msg_list
                try:
                    n_params_emb = np.prod(self.mod.embedding.weight.shape)
                except:
                    n_params_emb = 0

                try:
                    n_params_emb += np.prod(self.mod.idf_emb.weight.shape)
                except:
                    n_params_emb += 0

                msg = '# params: {}\n'.format(n_params)
                msg += '# embedding params: {}\n'.format(n_params_emb)
                msg += '# params excluding emb: {}'.format(n_params - n_params_emb)
                print(msg)
                self.log.append(msg)

                try:
                    self.mod.summary(*data)
                except:
                    print('unable to print model summary')

                self.model_params_printed = True

            if nbatch in self.checkpoint_batch_num:
                mx.nd.waitall()
                tmp_start = time.time()
                train_time = tmp_start - last_eval_time

                val_losses = []
                for batch in self.val_iter:
                    data = [x.as_in_context(self.model_ctx) for x in batch]
                    batch_size = data[0].shape[0]
                    with autograd.predict_mode():
                        loss = self.mod(*data)
                    val_losses.append(np.mean(loss.asnumpy()))

                self.metrics['loss']['train']['value'].append(np.mean(train_losses))
                self.metrics['loss']['train']['epoch'].append(self.current_epoch)
                self.metrics['loss']['train']['nbatch'].append(self.current_batch)
                self.metrics['loss']['val']['value'].append(np.mean(val_losses))
                self.metrics['loss']['val']['epoch'].append(self.current_epoch)
                self.metrics['loss']['val']['nbatch'].append(self.current_batch)
                internal_eval_time = time.time() - tmp_start

                tmp_start = time.time()
                self.external_evaluate()
                external_eval_time = time.time() - tmp_start

                self.print_latest_metrics()

                time_so_far_in_epoch = time.time() - start_time
                time_so_far_in_eval = time.time() - last_eval_time

                training_samples_per_sec = (n_sample - last_eval_n_sample) / train_time
                self.train_throughput = training_samples_per_sec
                timing_msg = 'Time elapsed in epoch: {:.2f}s. Time elapsed in eval {:.2f}s.\n' \
                             '[Train: {:.2f}s, IntEval {:.2f}s, ExtEval {:.2f}s]' \
                             '[Train throughput: {:.2f}/s]'.format(time_so_far_in_epoch, time_so_far_in_eval,
                                                                   train_time, internal_eval_time,
                                                                   external_eval_time,
                                                                   training_samples_per_sec)
                print(timing_msg)
                self.log.append(timing_msg)

                lr_reduce = False
                if self.early_stopping_enabled and self.eval_counter >= self.early_stopping_min_eval_counter:
                    monitor_val = self.metrics[self.early_stopping_metric_name]['val']['value'][-1]
                    self.early_stopping_condition_met, lr_reduce, ES_message = self.ES(value=monitor_val,
                                                                                       eval_counter=self.eval_counter)
                    self.log.append(ES_message)
                    print(ES_message)

                self.lr_history[self.eval_counter] = self.trainer.learning_rate
                if lr_reduce:
                    old_lr = self.trainer.learning_rate
                    new_lr = old_lr * self.lr_decay_ratio
                    self.lr_decay_count += 1
                    LR_msg = 'LR updated: from {} to {}, lr_decay_count={} (max={})'.format(old_lr, new_lr,
                                                                                            self.lr_decay_count,
                                                                                            self.lr_decay_count_max)
                    self.log.append(LR_msg)
                    print(LR_msg)

                    self.trainer.set_learning_rate(new_lr)
                    if new_lr < self.lr_min:
                        msg = 'LR_min reached ({}), stopping'.format(self.lr_min)
                        print(msg)
                        self.log.append(msg)
                        self.early_stopping_condition_met = True
                    if self.lr_decay_count >= self.lr_decay_count_max:
                        msg = 'LR decay count max reached ({}), stopping'.format(self.lr_decay_count)
                        print(msg)
                        self.log.append(msg)
                        self.early_stopping_condition_met = True

                self.eval_counter += 1

                if self.early_stopping_condition_met:
                    # nm.print_header('Early stopping condition met.', symbol='$')
                    msg = 'Early stopping condition met.'
                    print(msg)
                    self.log.append(msg)
                    break

                if self.max_eval_count is not None:
                    if self.eval_counter >= self.max_eval_count:
                        msg = 'Reached max_eval_count {}'.format(self.max_eval_count)
                        print(msg)
                        self.log.append(msg)
                        self.early_stopping_condition_met = True  # to prevent further training
                        break

                last_eval_time = time.time()
                last_eval_n_sample = n_sample

    def train(self, n_epochs=20, progress_bar=False):
        for i in range(n_epochs):
            self.train_one_epoch(epoch_id=None, progress_bar=progress_bar)

        with open(os.path.join(self.model_dir, 'log.txt'), 'w') as f:
            f.write('\n'.join(self.log))


class EarlyStopping:
    def __init__(self, mod, patience=10, minimize=True, restore_path=None, reduce_lr=True, eps=1e-4):
        self.eps = eps
        self.minimize = minimize
        self.patience = patience
        self.best_monitored_value = np.inf if minimize else 0.
        self.best_monitored_eval = 0
        self.restore_path = restore_path
        self.mod = mod
        self.reduce_lr = reduce_lr
        if self.reduce_lr:
            self.reduce_lr_patience = self.patience // 3
        else:
            self.reduce_lr_patience = 2 * self.patience  # lr will never be reduced in this case

        setup_msg = 'ES patience {}, LR reduce patience {}, eps {}'.format(self.patience, self.reduce_lr_patience,
                                                                           self.eps)
        print(setup_msg)

    def load_model(self):
        if self.restore_path is not None:
            self.mod.load_parameters(self.restore_path)
        else:
            # nm.print_error("ERROR: Failed to restore model")
            print("ERROR: Failed to restore model")

    def __call__(self, value, eval_counter):

        es_condition_met = False
        lr_reduce_condition_met = False
        if (self.minimize and value < self.best_monitored_value - self.eps) or (
                not self.minimize and value > self.best_monitored_value + self.eps):
            self.best_monitored_value = value
            self.best_monitored_eval = eval_counter
            self.mod.save_parameters(self.restore_path)
            msg = 'EarlyStopping: best val loss {:.4f} @ current eval, params saved to {}'.format(
                value, self.restore_path)
        elif self.best_monitored_eval + self.patience < eval_counter:
            msg = 'EarlyStopping: patience ran out. Best loss {:.4f}@eval{}, loading best params from {}'.format(
                self.best_monitored_value, self.best_monitored_eval, self.restore_path)
            self.load_model()
            es_condition_met = True
        else:
            es_patience_remaining = self.best_monitored_eval + self.patience - eval_counter
            lr_patience_remaining = self.best_monitored_eval + self.reduce_lr_patience - eval_counter

            lr_msg = ''
            if self.reduce_lr:
                if lr_patience_remaining < 0:
                    lr_msg = 'LR patience ({}) ran out.'.format(self.reduce_lr_patience)
                    self.load_model()
                    # self.best_monitored_eval = eval_counter
                    lr_reduce_condition_met = True
                else:
                    lr_msg = 'LR patience {} remaining.'.format(lr_patience_remaining)

            msg = 'EarlyStopping: ES patience {} remaining. {} '.format(es_patience_remaining, lr_msg)
            msg += 'Current/best loss: {:.4f}@eval{} / {:.4f}@eval{}'.format(value, eval_counter,
                                                                             self.best_monitored_value,
                                                                             self.best_monitored_eval)

        return es_condition_met, lr_reduce_condition_met, msg
