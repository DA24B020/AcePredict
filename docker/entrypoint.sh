#!/bin/sh
# Spark (Hadoop) and Ray look the current user up in /etc/passwd. The backend and the
# pipeline runner run as your VM user's UID (so files they write stay yours); add a
# passwd entry for that UID when the image does not have one.
if ! whoami >/dev/null 2>&1; then
  echo "acepredict:x:$(id -u):$(id -g):acepredict:/tmp:/bin/sh" >> /etc/passwd
fi
exec "$@"
