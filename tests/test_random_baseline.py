import argparse
import contextlib
import io
import json
import logging
import math
import os
from pathlib import Path
import pickle
import random
import tempfile
import time
from types import SimpleNamespace
import unittest

from source_loader import load_class_methods, load_function
from util.batch_schedule import batches_per_epoch, candidate_batch_indices, select_batch_indices


ROOT = Path(__file__).resolve().parents[1]


class Permutation(list):
    def __getitem__(self, key):
        result = super().__getitem__(key)
        return Permutation(result) if isinstance(key, slice) else result

    def tolist(self):
        return list(self)


class ModelDouble:
    def __init__(self):
        self.training = True
        self._AEBuilder__encoder = {'test_encoder': True}

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode


class RandomBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Configuration = load_class_methods(
            ROOT / 'util/conf.py', 'Configuration',
            ['__init__', 'getHP', 'setHP', '__validate', '__setup', 'loadConf', 'dumpConf'],
            {'os': os, 'Path': Path, 'json': json},
        )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.baseline-test-', dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.previous_cwd = Path.cwd()
        self.addCleanup(os.chdir, self.previous_cwd)
        os.chdir(self.work)
        (self.work / 'conf').mkdir()

    def make_experiment(self, schedule, start_epoch=0):
        experiment_type = load_class_methods(
            ROOT / 'util/experiment.py', 'Experiment', ['run'], {
                'torch': SimpleNamespace(
                    randperm=lambda count: Permutation(reversed(range(count))),
                    no_grad=contextlib.nullcontext,
                ),
                'math': math, 'os': os, 'random': random.Random(123),
                'timer': time.perf_counter, 'DATASET_CONFIGS': [0, 1],
                'batches_per_epoch': batches_per_epoch,
                'candidate_batch_indices': candidate_batch_indices,
                'select_batch_indices': select_batch_indices,
            },
        )
        experiment = experiment_type()
        values = {
            'mode': 'pretrain', 'train_type': 'linearlycombine',
            'training_schedule': schedule, 'alpha': 0.8, 'to_embed': True,
            'pickle': str(self.work / 'exports' / 'encoder.pkl'),
        }
        experiment._Experiment__conf = SimpleNamespace(getHP=values.__getitem__)
        experiment.has_setup = True
        experiment.epoch = start_epoch
        experiment.max_epoch = start_epoch + 2
        experiment.model = ModelDouble()
        experiment.train_total_loader = list(range(20))
        experiment.index_list = [index % 2 for index in range(20)]
        experiment.val_db_loader = []
        experiment.val_query_loader1 = []
        experiment.val_query_loader2 = []
        experiment.orth_regularizer = 'none'
        experiment.logger = logging.getLogger('baseline_control_flow_test')
        experiment.shuffle_batch_inter = lambda batches: batches
        experiment._Experiment__adjust_lr = lambda: None
        experiment._Experiment__adjust_wd = lambda: None
        experiment._Experiment__model_change = lambda: None
        self.trained = []
        self.validated = []
        self.scored = []
        self.checkpoints = []
        experiment._Experiment__checkpoint = lambda persist_model=True: self.checkpoints.append(
            (experiment.epoch, persist_model))
        experiment._Experiment__train = lambda alpha: self.trained.append(
            (experiment.epoch, list(experiment.train_db_loader)))
        experiment._Experiment__validate = lambda alpha: self.validated.append(experiment.epoch)
        score_order = [10, 9, 8] + [index for index in range(20) if index not in (10, 9, 8)]
        scores = {index: rank for rank, index in enumerate(score_order)}

        def score(batch, alpha):
            self.assertEqual(schedule, 'loss_curriculum', 'Random baseline must not evaluate difficulty')
            self.assertFalse(experiment.model.training)
            self.assertEqual(experiment.epoch, start_epoch)
            self.assertEqual(self.trained, [], 'Difficulty scoring must precede all optimizer updates')
            self.scored.append(batch)
            return float(scores[batch])

        experiment._Experiment__calculate_curriculum_loss = score
        return experiment

    def run_silently(self, experiment):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            experiment.run()
        return output.getvalue()

    def test_random_baseline_skips_scoring_and_trains_from_full_pool(self):
        experiment = self.make_experiment('random_baseline')
        self.run_silently(experiment)
        self.assertEqual(self.scored, [])
        self.assertEqual(self.trained, [(1, [19, 18]), (2, [19, 18])])
        self.assertEqual(self.validated, [1, 2])
        self.assertEqual(self.checkpoints, [(0, False), (1, True), (2, True)])

    def test_random_baseline_resumed_epoch_still_uses_full_pool(self):
        experiment = self.make_experiment('random_baseline', start_epoch=50)
        self.run_silently(experiment)
        self.assertEqual(self.trained, [(51, [19, 18]), (52, [19, 18])])

    def test_curriculum_preserves_initial_ranking_and_training_budget(self):
        experiment = self.make_experiment('loss_curriculum')
        self.run_silently(experiment)
        self.assertEqual(self.scored, list(range(20)))
        self.assertEqual(self.trained, [(1, [8, 9]), (2, [8, 9])])
        self.assertTrue(experiment.model.training)
        self.assertEqual(experiment.train_list, [[0], [1]])

    def test_encoder_export_obeys_configured_path(self):
        experiment = self.make_experiment('loss_curriculum')
        self.run_silently(experiment)
        export = self.work / 'exports' / 'encoder.pkl'
        self.assertTrue(export.is_file(), 'Encoder must be saved at the configured pickle path')
        with export.open('rb') as file:
            self.assertEqual(pickle.load(file), {'test_encoder': True})
        self.assertFalse((self.work / 'conf' / 'our_pretrain.pkl').exists())

    def test_legacy_configuration_still_selects_curriculum(self):
        configuration = self.Configuration()
        try:
            schedule = configuration.getHP('training_schedule')
        except ValueError:
            self.fail('Legacy configuration needs a default training schedule')
        self.assertEqual(schedule, 'loss_curriculum')

    def test_unknown_schedule_is_rejected_during_configuration_load(self):
        config_file = self.work / 'invalid.json'
        config_file.write_text(json.dumps({'training_schedule': 'typo'}), encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            self.Configuration(str(config_file))

    def test_pair_configs_preserve_hyperparameters_and_separate_outputs(self):
        baseline_file = ROOT / 'conf/random_baseline/example.json'
        self.assertTrue(baseline_file.is_file(), 'Independent random baseline config is missing')
        with contextlib.redirect_stdout(io.StringIO()):
            course = self.Configuration(str(ROOT / 'conf/example.json'))
            baseline = self.Configuration(str(baseline_file))
        self.assertEqual(baseline.getHP('training_schedule'), 'random_baseline')
        allowed_differences = {
            'training_schedule', 'name', 'conf_path', 'log_filepath', 'record_folder',
            'checkpoint_folder', 'pickle', 'result_path',
        }
        keys = set(course.defaults) | set(course.settings) | set(baseline.defaults) | set(baseline.settings)
        for key in keys - allowed_differences:
            with self.subTest(key=key):
                self.assertEqual(course.getHP(key), baseline.getHP(key))
        for key in ['checkpoint_folder', 'log_filepath', 'pickle', 'result_path']:
            with self.subTest(output=key):
                self.assertNotEqual(course.getHP(key), baseline.getHP(key))
        self.assertEqual(Path(baseline.getHP('checkpoint_folder')), ROOT / 'conf/random_baseline')

    def test_elapsed_time_log_follows_configured_log_directory(self):
        output_dir = self.work / 'random_outputs'
        output_dir.mkdir()
        config = SimpleNamespace(getHP=lambda key: str(output_dir / 'fit.log'))
        main = load_function(ROOT / 'run.py', 'main', {
            'argparse': argparse, 'Path': Path,
            'Configuration': lambda path, dump: config,
            'Experiment': lambda conf: SimpleNamespace(run=lambda: None),
        })
        main(['run.py', '-C', 'unused.json'])
        timing_log = output_dir / '0time.log'
        self.assertTrue(timing_log.is_file(), 'Elapsed-time log must follow the experiment output directory')
        self.assertGreaterEqual(float(timing_log.read_text(encoding='utf-8').strip()), 0)
        self.assertFalse((self.work / 'conf/0time.log').exists())


if __name__ == '__main__':
    unittest.main()
