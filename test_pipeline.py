import unittest

from pipeline import run_research_pipeline


class PipelineValidationTests(unittest.TestCase):
    def test_empty_topic_raises_value_error(self):
        with self.assertRaises(ValueError):
            run_research_pipeline("   ")


if __name__ == "__main__":
    unittest.main()
