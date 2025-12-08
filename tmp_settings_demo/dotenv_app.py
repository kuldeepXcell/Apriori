import os
from dotenv import load_dotenv


def load_settings():
    load_dotenv()
    host = os.getenv("HOST")
    timeout = os.getenv("TIMEOUT")
    debug = os.getenv("DEBUG")

    print("dotenv approach:")
    print("  HOST   :", host, type(host))
    print("  TIMEOUT:", timeout, type(timeout))
    print("  DEBUG  :", debug, type(debug))

    # Common footgun: forgetting to cast strings from env.
    # This will blow up if TIMEOUT is not an int already.
    wait_ms = timeout + 5  # TypeError when timeout is a str
    print("  wait_ms:", wait_ms)


if __name__ == "__main__":
    load_settings()
