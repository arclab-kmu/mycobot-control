"""``mycobot-camera``: watch the cameras live in a browser instead of recording the screen.

Read-only by construction: nothing here imports the arm's backend, and the page
has no route that could command anything. OpenCV (the ``camera`` extra) is
imported only inside the functions that open a camera or encode a frame, so
the server, the page and the tests run without it.
"""
