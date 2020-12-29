import socket

from slack import WebClient

from terra.settings import TERRA_CONFIG

run_to_ts = {}


def init_task_notifications(run_id: int):
    return
    if not TERRA_CONFIG["notify"]:
        return

    try:
        client = WebClient(TERRA_CONFIG["slack_web_client_id"])
        message = client.chat_postMessage(
            channel="#experiments",
            text="🎬 Task starting:`run_id={}`".format(run_id),
        )
        run_to_ts[run_id] = message.data["ts"]

        client.chat_postMessage(
            channel="#experiments",
            text="This task is running on: {}".format(socket.gethostname()),
            thread_ts=run_to_ts[run_id],
        )

    # key error to catch missing run_id
    except (OSError, KeyError) as e:
        print(f"Failed to post slack message: {e}")


def notify_task_completed(run_id: str):
    return
    if not TERRA_CONFIG["notify"]:
        return

    try:
        client = WebClient(TERRA_CONFIG["slack_web_client_id"])
        client.chat_postMessage(
            channel="#experiments",
            text="✅ Task Completed.",
            thread_ts=run_to_ts[run_id],
        )

    # key error to catch missing run_id
    except (OSError, KeyError) as e:
        print(f"Failed to post slack message: {e}")


def notify_task_error(run_id: str, msg: str):
    return
    if not TERRA_CONFIG["notify"]:
        return

    try:
        client = WebClient(TERRA_CONFIG["slack_web_client_id"])
        client.chat_postMessage(
            channel="#experiments",
            text="⛔️ Task error: `run_id={}`".format(run_id),
            thread_ts=run_to_ts[run_id],
        )
        client.chat_postMessage(
            channel="#experiments",
            text="Check out the error message: \n```{}```".format(msg),
            thread_ts=run_to_ts[run_id],
        )
    except (OSError, KeyError) as e:
        print(f"Failed to post slack message: {e}")


def notify_task_checkpoint(run_id: str, msg: str):
    return
    if not TERRA_CONFIG["notify"]:
        return

    try:
        client = WebClient(TERRA_CONFIG["slack_web_client_id"])
        client.chat_postMessage(
            channel="#experiments",
            text="🚩Checkpoint: \n {}".format(msg),
            thread_ts=run_to_ts[run_id],
        )
    except (OSError, KeyError) as e:
        print(f"Failed to post slack message: {e}")
