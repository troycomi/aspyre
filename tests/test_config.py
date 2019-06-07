from unittest import TestCase
from aspyre.utils.config import Config


class ConfigTest(TestCase):
    def setUp(self):
        # A config object is initialized by a valid json string which may have nested attributes and lists,
        # and provides access to values as recursive attributes
        self.config = Config("""
            {
                "foo": 42,
                "bar": {
                    "names": ["alice", "bob"],
                    "pi": 3.14
                }
            }    
        """)

    def tearDown(self):
        pass

    def testInt(self):
        self.assertEqual(42, self.config.foo)

    def testFloat(self):
        self.assertAlmostEqual(3.14, self.config.bar.pi)

    def testList(self):
        self.assertEqual("alice", self.config.bar.names[0])
        self.assertEqual("bob", self.config.bar.names[1])

    def testOverride(self):
        # Values can be overridden in a context block by providing a override dictionary
        with self.config.override({'bar.pi': 3}):
            self.assertEqual(3, self.config.bar.pi)

        # Outside the block, we still have the original value
        self.assertAlmostEqual(3.14, self.config.bar.pi)
