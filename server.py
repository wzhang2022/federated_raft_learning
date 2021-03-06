import argparse

import rpyc

from utils import Net
from pysyncobj import SyncObj, replicated, SyncObjConf, SyncObjException

import numpy as np
import time


net = Net()


class SyncNet(SyncObj):
    def __init__(self, raft_port, raft_peers, conf=None):
        super(SyncNet, self).__init__(
            f'localhost:{raft_port}',
            list(map(lambda x: f'localhost:{x}', raft_peers)),
            conf=conf
        )
        self.params = list(map(lambda x: x.data.numpy(), net.parameters()))
        self.lr = 0.001
        self.epoch = 0
        self.time = time.time()

    @replicated
    def update_all_gradients(self, gradients):
        for i, grad in enumerate(gradients):
            assert isinstance(grad, np.ndarray)
            self.params[i] -= grad.clip(-0.1, 0.1) * self.lr

    @replicated
    def update_params(self, params):
        for i, param in enumerate(params):
            assert isinstance(param, np.ndarray)
            self.params[i] = param

    @replicated
    def update_epoch(self):
        self.epoch += 1
        print(f"Epoch {self.epoch} completed in {time.time() - self.time} seconds")
        self.time = time.time()

    @replicated
    def get_model_params(self):
        return self.params


class FederatedLearningService(rpyc.Service):
    def __init__(self, synced_net: SyncNet):
        super(FederatedLearningService, self).__init__()
        self.synced_net = synced_net

    def exposed_send_gradient(self, param_gradients):
        gradients = list(map(lambda x: np.array(x), param_gradients))
        self.synced_net.update_all_gradients(gradients, sync=True)

    def exposed_get_model_params(self):
        isReady = self.synced_net.isReady()
        if isReady:
            params = self.synced_net.get_model_params(sync=True)
        else:
            raise SyncObjException
        return params

    def exposed_reset_model_params(self, params):
        gradients = list(map(lambda x: np.array(x), params))
        self.synced_net.update_params(gradients, sync=True)

    def exposed_mark_epoch_done(self):
        self.synced_net.update_epoch(sync=True)

    def exposed_get_server_availability(self):
        return self.synced_net.isReady()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raft_port", dest="raft_port", type=int, required=True)
    parser.add_argument("--client_port", dest="client_port", type=int, required=True)
    parser.add_argument("--raft_peers", dest="raft_peers", type=int, nargs='+', required=True)
    args = parser.parse_args()

    conf = SyncObjConf(
        sendBufferSize=2**19,
        recvBufferSize=2**19,
        appendEntriesBatchSizeBytes=2**19,
        commandsQueueSize=10**5,
        connectionTimeout=7,
    )
    # print(conf)
    # conf = None
    syncnet = SyncNet(args.raft_port, args.raft_peers, conf=conf)
    print("Synced raft object created")
    service = FederatedLearningService(syncnet)
    server = rpyc.utils.server.ThreadedServer(
        service,
        hostname="localhost",
        port=args.client_port,
        protocol_config={
            "allow_pickle": True,
        })
    server.start()
