
import sys
import logging
import argparse
from omegaconf import OmegaConf

from dna import config, initialize_logger
from dna.node.pika.pika_execution import PikaExecutionFactory, PikaConnector
from dna.node.pika.pika_execution_server import PikaExecutionServer

LOGGER = logging.getLogger('dna.node.pika')


def parse_args():
    parser = argparse.ArgumentParser(description="Run a node processor.")
    parser.add_argument("--conf_root", metavar="dir", help="Root directory for configurations", default="conf")
    parser.add_argument("--kafka_brokers", nargs='+', metavar="hosts", help="Kafka broker hosts list", default=None)
    parser.add_argument("--rabbitmq_url", metavar="URL", help="RabbitMQ broker URL",
                        default="rabbitmq://admin:admin@localhost:5672/track_requests")
    parser.add_argument("--show", "-s", nargs='?', const='0x0', default='0x0')

    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")
    return parser.parse_known_args()

def main():
    args, _ = parse_args()
    initialize_logger(args.logger)
    
    connector, request_qname = PikaConnector.parse_url(args.rabbitmq_url)
    fact = PikaExecutionFactory(args=args)
    server = PikaExecutionServer(connector=connector,
                                 execution_factory=fact,
                                 request_qname=request_qname,
                                 logger=LOGGER)
    server.run()

if __name__ == '__main__':
	main()