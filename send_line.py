# -*- coding: utf-8 -*-
# pylint: disable=C0114,C0115,C0116
import requests

LINE_TOKEN_TEST = "Bearer <TOKEN for TEST>"
LINE_TOKEN = "Bearer <TOKEN>"


def send(message, real=True):
    if real is False:
        line_token = LINE_TOKEN_TEST
        print("Send LINE to TEST Channel")
    else:
        line_token = LINE_TOKEN
        print("Send LINE to REAL Channel")
    headers = {
        "Authorization": line_token,
    }
    files = {
        "message": (None, message),
    }
    requests.post(
        "https://notify-api.line.me/api/notify",
        headers=headers,
        files=files,
        timeout=30,
    )
    return
