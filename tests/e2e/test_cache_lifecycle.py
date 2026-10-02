import copy
import unittest

from cache_lifecycle import conversation, extend, run_lifecycle


class CacheLifecycleTest(unittest.TestCase):
    def test_growth_preserves_prior_messages_and_tool_pair(self):
        before = conversation(7, 2048)
        frozen = copy.deepcopy(before)
        after = extend(before, 7, 0)
        self.assertEqual(before, frozen)
        self.assertEqual(after[:len(before)], before)
        self.assertEqual(after[2]["tool_calls"][0]["id"], after[3]["tool_call_id"])
        self.assertEqual(after[-1]["role"], "user")

    def test_seed_reproduces_fixture_and_separates_roots(self):
        self.assertEqual(conversation(7, 2048), conversation(7, 2048))
        self.assertNotEqual(conversation(7, 2048)[0], conversation(8, 2048)[0])

    def test_no_cache_hit_cannot_pass_lifecycle(self):
        class NoCacheSuite:
            class args:
                window = 32
                reference_outputs = {'7': {'cold': 'READY'}}

            def request(self, *args, **kwargs):
                return {"text": "READY", "seconds": 0, "usage": {
                    "prompt_tokens": 3000, "completion_tokens": 1,
                    "prompt_tokens_details": {"cached_tokens": 0}}}

            def liveness(self):
                pass

        with self.assertRaisesRegex(AssertionError, "no long retained prefix"):
            run_lifecycle(NoCacheSuite(), 7)

    def test_successful_generation_with_wrong_cached_output_is_failure(self):
        class DirtyCacheSuite:
            class args:
                window = 32
                reference_outputs = {'7': {'cold': 'READY', 'warm': 'READY'}}
            calls = 0
            def request(self, *args, **kwargs):
                self.calls += 1
                return {'text': 'READY' if self.calls == 1 else 'DIRTY', 'seconds': 0, 'usage': {
                    'prompt_tokens': 3000, 'completion_tokens': 1,
                    'prompt_tokens_details': {'cached_tokens': 0 if self.calls == 1 else 2500}}}
            def liveness(self):
                pass
        with self.assertRaisesRegex(AssertionError, 'differs from the same-input cold control'):
            run_lifecycle(DirtyCacheSuite(), 7)


if __name__ == '__main__':
    unittest.main()
