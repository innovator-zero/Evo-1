"""Run with python -m CALVIN_evaluation.server --checkpoint PATH."""
import argparse
import asyncio
import json
import logging

from websockets.asyncio.server import serve


async def handle_connection(connection, policy):
    async for message in connection:
        try:
            response = policy.infer(json.loads(message))
        except Exception as exc:
            logging.exception("CALVIN inference failed")
            await connection.send(json.dumps({"error": str(exc)}))
            await connection.close(code=1011, reason="Inference failed")
            return
        await connection.send(json.dumps(response, allow_nan=False))


async def run_server(policy, host, port):
    async with serve(lambda connection: handle_connection(connection, policy), host, port,
                     max_size=100_000_000, ping_interval=None):
        logging.info("CALVIN policy ready at ws://%s:%s", host, port)
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--inference_steps", type=int)
    parser.add_argument("--arm", default="calvin_franka_delta")
    parser.add_argument("--dataset", default="InternData-Calvin_ABC")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    from accelerate.utils import set_seed
    from CALVIN_evaluation.policy import CalvinPolicy
    set_seed(args.seed)
    policy = CalvinPolicy.load(args.checkpoint, args.device, args.model_path,
                               args.inference_steps, args.arm, args.dataset)
    asyncio.run(run_server(policy, args.host, args.port))


if __name__ == "__main__":
    main()
