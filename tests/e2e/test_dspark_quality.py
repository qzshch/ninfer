import unittest
from dspark_quality import CASES, grade
class QualityOracleTest(unittest.TestCase):
    def test_math_exact_value_and_format(self):
        self.assertTrue(grade(CASES[3],"FINAL: 338350")["passed"])
        self.assertFalse(grade(CASES[3],"FINAL: 338351")["passed"])
    def test_code_functional_lower_bound(self):
        code="def lower_bound(values,target):\n    for i,v in enumerate(values):\n        if v>=target:return i\n    return len(values)\n"
        self.assertTrue(grade(CASES[2],code)["passed"])
        self.assertFalse(grade(CASES[2],"def lower_bound(values,target): return 0")["passed"])
    def test_reject_io_code(self):
        self.assertFalse(grade(CASES[2],"import os\ndef lower_bound(values,target): return len(values)")["passed"])
    def test_merge_accepts_safe_lambda_and_checks_mutation(self):
        code="def merge_intervals(intervals):\n    merged=[]\n    for start,end in sorted(intervals,key=lambda pair: pair[0]):\n        if merged and start<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],end)\n        else: merged.append([start,end])\n    return merged\n"
        self.assertTrue(grade(CASES[0],code)["passed"])
        self.assertFalse(grade(CASES[0],"def merge_intervals(intervals):\n    intervals.sort()\n    return intervals")["passed"])
    def test_symmetric_expected_value(self):
        self.assertEqual(CASES[5][3],13**2-2*40)
if __name__=="__main__":unittest.main()
