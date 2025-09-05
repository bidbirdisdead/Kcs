"""Unit tests for position_manager and trade_logic modules."""


import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import position_manager
import trade_logic

class TestPositionManager(unittest.TestCase):
    def test_check_exposure_limit(self):
        # Should always be True if no positions (mocked)
        self.assertTrue(position_manager.check_exposure_limit(1000.0))

    def test_get_open_positions(self):
        # Should return a dict (mocked)
        result = position_manager.get_open_positions()
        self.assertIsInstance(result, dict)

class TestTradeLogic(unittest.TestCase):
    def test_is_trade_opportunity(self):
        self.assertTrue(trade_logic.is_trade_opportunity(50, 55, 20, 2, 10, 60))
        self.assertFalse(trade_logic.is_trade_opportunity(50, 51, 5, 2, 10, 60))

    def test_score_opportunity(self):
        score = trade_logic.score_opportunity(50, 55, 20, 2, 10, 60)
        self.assertGreater(score, 0)
        score2 = trade_logic.score_opportunity(50, 51, 5, 2, 10, 60)
        self.assertEqual(score2, 0.0)

if __name__ == "__main__":
    unittest.main()
