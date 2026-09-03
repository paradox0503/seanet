import importlib
import unittest


class BatchScheduleTests(unittest.TestCase):
    def setUp(self):
        try:
            self.schedule = importlib.import_module('util.batch_schedule')
        except ModuleNotFoundError as error:
            if error.name != 'util.batch_schedule':
                raise
            self.fail('The pure batch scheduling implementation is missing')

    def test_budget_preserves_existing_ten_percent_floor(self):
        for total, expected in [(1, 1), (9, 1), (10, 1), (19, 1), (20, 2), (15630, 1563)]:
            with self.subTest(total=total):
                self.assertEqual(self.schedule.batches_per_epoch(total), expected)

    def test_budget_rejects_empty_or_invalid_counts(self):
        for total in [0, -1, 2.5, True]:
            with self.subTest(total=total), self.assertRaises(ValueError):
                self.schedule.batches_per_epoch(total)
        for divisor in [0, -1, 1.5, True]:
            with self.subTest(divisor=divisor), self.assertRaises(ValueError):
                self.schedule.batches_per_epoch(20, divisor)

    def test_curriculum_boundaries_remain_cumulative(self):
        ordered = list(range(80, 0, -1))
        for epoch, length in [(0, 10), (9, 10), (10, 20), (19, 20), (20, 30),
                              (30, 40), (40, 50), (50, 60), (60, 70), (69, 70),
                              (70, 80), (79, 80), (80, 80), (99, 80)]:
            with self.subTest(epoch=epoch):
                self.assertEqual(
                    self.schedule.candidate_batch_indices('loss_curriculum', ordered, epoch),
                    ordered[:length],
                )

    def test_curriculum_keeps_remainders_and_tiny_pools(self):
        self.assertEqual(self.schedule.candidate_batch_indices('loss_curriculum', [91], 0), [91])
        ordered = [91, 42, 5, 18, 77, 31, 9, 6, 0]
        self.assertEqual(self.schedule.candidate_batch_indices('loss_curriculum', ordered, 0), [91, 42])
        self.assertEqual(self.schedule.candidate_batch_indices('loss_curriculum', ordered, 70), ordered)

    def test_random_baseline_never_restricts_the_candidate_pool(self):
        ordered = [80, 7, 42, 3, 199]
        for epoch in range(100):
            with self.subTest(epoch=epoch):
                self.assertEqual(
                    self.schedule.candidate_batch_indices('random_baseline', ordered, epoch), ordered)

    def test_candidate_pool_does_not_mutate_input(self):
        ordered = [7, 2, 99]
        result = self.schedule.candidate_batch_indices('random_baseline', ordered, 0)
        result.reverse()
        self.assertEqual(ordered, [7, 2, 99])

    def test_candidate_pool_rejects_invalid_schedule_or_epoch(self):
        with self.assertRaises(ValueError):
            self.schedule.candidate_batch_indices('typo', [1], 0)
        with self.assertRaises(ValueError):
            self.schedule.candidate_batch_indices('random_baseline', [], 0)
        for epoch in [-1, 0.5, True]:
            with self.subTest(epoch=epoch), self.assertRaises(ValueError):
                self.schedule.candidate_batch_indices('random_baseline', [1], epoch)

    def test_candidate_pool_rejects_invalid_stage_settings(self):
        for epochs, stages in [(0, 8), (80, 0), (81, 8), (4, 8), (80.0, 8), (80, True)]:
            with self.subTest(epochs=epochs, stages=stages), self.assertRaises(ValueError):
                self.schedule.candidate_batch_indices('loss_curriculum', [1], 0, epochs, stages)

    def test_selection_maps_positions_back_to_original_batch_ids(self):
        self.assertEqual(self.schedule.select_batch_indices([7, 2, 99, 6], 2, [2, 0]), [99, 7])

    def test_selection_uses_only_budget_from_full_permutation(self):
        self.assertEqual(self.schedule.select_batch_indices([7, 2, 99, 6], 2, [2, 0, 3, 1]), [99, 7])

    def test_selection_uses_whole_pool_when_it_fits(self):
        for budget in [2, 3]:
            self.assertEqual(self.schedule.select_batch_indices([7, 2], budget, []), [7, 2])

    def test_selection_rejects_duplicate_or_out_of_range_positions(self):
        for positions in [[0, 0], [-1, 0], [0, 4], [0], [0, 1.5], [0, True]]:
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                self.schedule.select_batch_indices([7, 2, 99, 6], 2, positions)

    def test_selection_rejects_empty_or_duplicate_candidates(self):
        for candidates in [[], [7, 7]]:
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                self.schedule.select_batch_indices(candidates, 1, [0])

    def test_selection_rejects_invalid_budget(self):
        for budget in [0, -1, 1.5, True]:
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.schedule.select_batch_indices([7, 2], budget, [0])

    def test_equal_budget_for_both_schedules(self):
        ordered = list(range(80))
        for mode in ['loss_curriculum', 'random_baseline']:
            for epoch in [0, 10, 50, 99]:
                pool = self.schedule.candidate_batch_indices(mode, ordered, epoch)
                selected = self.schedule.select_batch_indices(
                    pool, self.schedule.batches_per_epoch(80), list(reversed(range(len(pool)))))
                self.assertEqual(len(selected), 8)
                self.assertEqual(len(set(selected)), 8)


if __name__ == '__main__':
    unittest.main()
