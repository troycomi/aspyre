from unittest import TestCase
from aspyre.utils.config import Config, ConfigArgumentParser


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
                },
                "baz": null
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

    def testFlatten(self):
        # The 'flatten' method on the Config object gives us a flat dictionary of key=>value pairs
        # where keynames are flattened using a '.' delimited
        d = self.config.flatten()
        self.assertDictEqual(
            d,
            {
                'foo': 42,
                'bar.names.0': 'alice',
                'bar.names.1': 'bob',
                'bar.pi': 3.14,
                'baz': None
            }
        )

    def testConfigArgumentParser(self):
        # A ConfigArgumentParser can be instantiated from a Config object
        # which provides an override mechanism for the Config object through a context manager

        # If the 'config' kwarg is unspecified in the constructor, the 'config' object in the aspyre package
        # is (temporarily) overridden.
        # This allows scripts to support all 'config.*' options that are found in the aspyre 'config' object

        # Here we test our custom 'self.config' object since we can't make any guarantees about keys present in
        # the aspyre config object
        parser = ConfigArgumentParser(config=self.config)

        # 'foo' has the expected value here
        self.assertEqual(42, self.config.foo)

        with parser.parse_args(['--config.foo', '99']):
            # 'foo' value overridden!
            self.assertEqual(99, self.config.foo)

        # 'foo' reverts to its expected value here
        self.assertEqual(42, self.config.foo)
