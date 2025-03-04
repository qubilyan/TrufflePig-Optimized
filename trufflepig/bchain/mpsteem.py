
import logging
import urllib.request as rqst

from steem import Steem
from steem.commit import Commit
from steem.steemd import Steemd


logger = logging.getLogger(__name__)


class MPSteem(Steem):
    """Multiprocessing safe Steem"""
    def __init__(self, nodes: list, no_broadcast=False, **kwargs):

        self.nodes = self.get_up_nodes(nodes)
        self.no_broadcast = no_broadcast
        self.kwargs = kwargs.copy()
        super().__init__(nodes=nodes, no_broadcast=no_broadcast, **kwargs)
        logger.info('Steem is ready, I am connected to {}.'.format(self.nodes))

    def reconnect(self):
        """Creates a new Steemd and Commit"""
        self.steemd = Steemd(
            nodes=self.nodes.copy(),
            **self.kwargs.copy()
        )
        self.commit = Commit(
            steemd_instance=self.steemd,
            no_broadcast=self.no_broadcast,
            **self.kwargs.copy()
        )

    def get_up_nodes(self, nodes):
        """Checks and returns nodes that are up and running"""
        logger.info('Checking {} nodes: {}'.format(len(nodes), nodes))
        node_codes = []
        for node in nodes:
            try:
                if not node.startswith('http'):
                    test_node = 'https://' + node
                else:
                    test_node = node
                node_codes.append((node, rqst.urlopen(test_node,
                                                      timeout=1.0).getcode()))
            except Exception as e:
                node_codes.append((node, 400))
        up_nodes = [x[0] for x in node_codes if x[1] == 200]
        if not up_nodes:
            raise RuntimeError('No Steem nodes available')
        logger.info('Found {} UP nodes: {}'.format(len(up_nodes), up_nodes))
        return up_nodes

    def __getstate__(self):
        result = self.__dict__.copy()
        del result['steemd']
        del result['commit']
        return result

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        self.reconnect()