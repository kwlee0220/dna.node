import json

from dna import config
from dna.execution import ExecutionState

from dna.node.pika.pika_execution import PikaConnector
from dna.node.pika.pika_execution_client import PikaExecutionClient


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Detect Objects in a video file")
    parser.add_argument("req_message", help="request file path")
    parser.add_argument("--kafka_brokers", nargs='+', metavar="hosts", help="Kafka broker hosts list", default=None)
    parser.add_argument("--rabbitmq_url", metavar="URL", help="RabbitMQ broker URL",
                        default="rabbitmq://admin:admin@localhost:5672/track_requests")
    return parser.parse_known_args()


def main():
    args, unknown = parse_args()
    
    json_str = ''
    with open(args.req_message, 'r') as file:
        json_str = file.read()
        
    client = PikaExecutionClient.from_url(args.rabbitmq_url)
    try:
        client.start(json_str)
        for resp in client.report_progress():
            resp = config.to_conf(resp)
            print(resp)
            # if resp.state == 'RUNNING' and resp.progress.frame_index >= 40:
            #     client.stop(resp.id, 'timeout')
    finally:
        client.close()

if __name__ == '__main__':
	main()