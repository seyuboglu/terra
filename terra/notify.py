import socket

from slack import WebClient

from terra.settings import TERRA_CONFIG

run_to_ts = {}


def init_task_notifications(run_dir: str):
    if not TERRA_CONFIG["notify"]:
        return
    client = WebClient(TERRA_CONFIG["slack_web_client_id"])

    message = client.chat_postMessage(
        channel="#experiments",
        text="🎬 Task starting...```{}```".format(run_dir),
    )
    run_to_ts[run_dir] = message.data["ts"]

    client.chat_postMessage(
        channel="#experiments",
        text="This process is running on: {}".format(socket.gethostname()),
        thread_ts=run_to_ts[run_dir],
    )


def notify_task_completed(run_dir: str):
    if not TERRA_CONFIG["notify"]:
        return
    client = WebClient(TERRA_CONFIG["slack_web_client_id"])
    client.chat_postMessage(
        channel="#experiments",
        text="✅ Process Completed.",
        thread_ts=run_to_ts[run_dir],
    )


def notify_task_error(run_dir: str, msg: str):
    if not TERRA_CONFIG["notify"]:
        return
    client = WebClient(TERRA_CONFIG["slack_web_client_id"])
    client.chat_postMessage(
        channel="#experiments",
        text="⛔️ Process Error: `{}`".format(run_dir),
        thread_ts=run_to_ts[run_dir],
    )
    client.chat_postMessage(
        channel="#experiments",
        text="Check out the error message: \n```{}```".format(msg),
        thread_ts=run_to_ts[run_dir],
    )


def notify_task_checkpoint(run_dir: str, msg: str):
    if not TERRA_CONFIG["notify"]:
        return
    client = WebClient(TERRA_CONFIG["slack_web_client_id"])
    client.chat_postMessage(
        channel="#experiments",
        text="🚩Checkpoint: \n {}".format(msg),
        thread_ts=run_to_ts[run_dir],
    )
